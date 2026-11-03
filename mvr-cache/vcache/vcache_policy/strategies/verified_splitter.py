import logging
import queue
import random
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np

from vcache.config import VCacheConfig
from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.similarity_evaluator import SimilarityEvaluator
from vcache.vcache_policy.strategies.verified import _Action, _Algorithm
from vcache.vcache_policy.vcache_policy import VCachePolicy
# [판정자] 팀원 판정자 코드(code_changes.patch)의 후보 답 합의 · 확률 계산 함수
from vcache.vcache_core.splitter.consensus import (
    predict_correct_probability,
    top1_excluded_agreement,
)

PAD_PARENT_ID = np.iinfo(np.uint64).max


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 2D float32 array row-wise (safe for zero rows).
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


class _MultiVectorHNSWIndex:
    """
    Policy-local multivector HNSW index using the custom hnswlib fork that supports:
      - add_items(..., parent_ids=...)
      - knn_query_skipping_duplicates_with_parent(...)

    We index many vectors per cache entry (parent_id = embedding_id) and return candidate parent IDs.
    """

    def __init__(
        self,
        *,
        max_elements: int = 2_000_000,
        ef_construction: int = 256,
        M: int = 64,
        ef: int = 400,
        num_threads: int = 32,
    ):
        self.max_elements = int(max_elements)
        self.ef_construction = int(ef_construction)
        self.M = int(M)
        self.ef = int(ef)
        self.num_threads = int(num_threads)

        self._index = None
        self._dim: Optional[int] = None
        self._next_vector_id: int = 0
        self._indexed_parents: set[int] = set()
        self._lock = threading.Lock()

    def _ensure_init(self, dim: int) -> None:
        if self._index is not None:
            return
        import hnswlib

        self._dim = int(dim)
        # Use inner product on L2-normalized vectors so dot == cosine similarity.
        self._index = hnswlib.Index(space="ip", dim=self._dim)
        self._index.init_index(
            max_elements=self.max_elements, M=self.M, ef_construction=self.ef_construction
        )
        self._index.set_ef(self.ef)
        try:
            self._index.set_num_threads(self.num_threads)
        except Exception:
            pass

        # Fail fast if the custom multivector APIs are missing.
        if not hasattr(self._index, "knn_query_skipping_duplicates_with_parent"):
            raise RuntimeError(
                "Custom hnswlib is required: missing method "
                "`Index.knn_query_skipping_duplicates_with_parent`. "
                "Install your fork with: "
                "python -m pip install -e /home/ali/vcahce/vcache/vcache_core/cache/embedding_store/hnswlib"
            )

    def add_parent_vectors(self, *, parent_id: int, vectors: np.ndarray) -> None:
        """
        Add a [N, D] float array of vectors for one parent_id.
        No-op if this parent_id was already indexed.
        """
        if vectors is None:
            return
        if vectors.ndim != 2:
            raise ValueError(f"Expected vectors [N,D], got shape={vectors.shape}")
        if vectors.shape[0] == 0:
            return

        with self._lock:
            if int(parent_id) in self._indexed_parents:
                return
            self._ensure_init(dim=int(vectors.shape[1]))

            x = vectors.astype(np.float32, copy=False)
            x = _l2_normalize_rows(x)

            n = int(x.shape[0])
            ids = np.arange(self._next_vector_id, self._next_vector_id + n, dtype=np.uint64)
            parent_ids = np.full((n,), np.uint64(parent_id), dtype=np.uint64)

            # Requires custom hnswlib fork supporting parent_ids.
            self._index.add_items(x, ids=ids, parent_ids=parent_ids, num_threads=self.num_threads)
            self._next_vector_id += n
            self._indexed_parents.add(int(parent_id))

    def query_candidate_parents(
        self, *, query_vectors: np.ndarray, k_per_vector: int
    ) -> list[int]:
        """
        Query with [Nq, D] vectors; union candidate parent IDs across per-vector KNN.
        """
        if query_vectors is None:
            return []
        if self._index is None:
            return []
        if query_vectors.ndim != 2 or query_vectors.shape[0] == 0:
            return []

        q = query_vectors.astype(np.float32, copy=False)
        q = _l2_normalize_rows(q)

        k = max(1, int(k_per_vector))
        out: set[int] = set()
        # Query each vector separately; keep internal threads at 1 (matches your test script).
        for i in range(int(q.shape[0])):
            qi = q[i : i + 1, :]
            _labels, _dists, parents = self._index.knn_query_skipping_duplicates_with_parent(
                qi, k=k, num_threads=1
            )
            if parents is None:
                continue
            for t in range(int(parents.shape[1])):
                pid = int(parents[0, t])
                if pid == int(PAD_PARENT_ID):
                    continue
                out.add(pid)
        return list(out)

class CallbackQueue(queue.Queue):
    """
    Same helper as in VerifiedDecisionPolicy: serialize cache updates in one worker thread.
    """

    def __init__(self, callback_function):
        super().__init__()
        self.callback_function = callback_function
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)

    def _worker(self):
        while True:
            if self._stop_event.is_set():
                break
            try:
                item = self.get(timeout=1)
                if item is None:
                    break
                self.callback_function(item)
                self.task_done()
            except queue.Empty:
                continue

    def start(self):
        self.worker_thread.start()

    def stop(self):
        if self.worker_thread.is_alive():
            self.put(None)
            self.worker_thread.join()


class VerifiedSplitterDecisionPolicy(VCachePolicy):
    """
    VerifiedDecisionPolicy variant where the similarity score is computed via:

      (query_prompt, cached_prompt) -> RL splitter (MaxSimSplitter) -> MaxSim similarity.

    The explore/exploit decision + Bayesian thresholding logic is reused from the original
    verified policy via `_Algorithm`.
    """

    def __init__(
        self,
        delta: float = 0.01,
        splitter=None,
        device: str = "cpu",
        candidate_selection: str = "top_k",
        candidate_k: int = 10,
        use_cached_candidate_segments: bool = False,
        multivector_max_elements: int = 2_000_000,
        mix_fullcos: bool = False,
        timing_collector=None,
        timing_cuda_sync: bool = False,
        consensus_model: Optional[dict] = None,
        consensus_scope: str = "all",
        collect_consensus: bool = False,
        consensus_answer_threshold: float = 0.85,
        consensus_min_candidates: int = 5,
        consensus_online_budget: Optional[float] = None,
        consensus_online_mode: str = "calibrated",
        consensus_online_bin_width: float = 0.02,
        consensus_online_prior: float = 20.0,
        consensus_online_probe: float = 0.1,
        consensus_online_z: float = 0.0,
        consensus_online_label_only: bool = False,
        consensus_seed: int = 0,
    ):
        self.bayesian = _Algorithm(delta=delta)
        # [판정자] consensus_model이 있으면 vCache가 재사용하지 않기로 한 요청 중
        # P(1등 정답 | s, a_excl, 후보 수) >= min_probability 인 것을 재사용으로 바꾼다.
        #   scope "all"  : vCache가 EXPLORE로 정한 모든 요청 (learned_all)
        #   scope "cold" : 그중 1등 항목의 관측이 6개 미만인 요청만 (learned_cold)
        if consensus_scope not in ("all", "cold"):
            raise ValueError(f"Unsupported consensus_scope={consensus_scope!r}")
        self.consensus_model = consensus_model
        self.consensus_scope = consensus_scope
        self.collect_consensus = bool(collect_consensus or consensus_model is not None)
        self.consensus_answer_threshold = float(consensus_answer_threshold)
        self.consensus_min_candidates = int(consensus_min_candidates)
        # [실행 중 한도] 팀원 online_budget.patch(b07856972ddc) · probe_label_only.patch(92ca6d07db0a)의
        # 규칙을 이 정책(판정자 = _apply_consensus_judge, p* = consensus_model["min_probability"])에 옮김.
        #   cap        : 비용 = 1 - p
        #   calibrated : 비용 = 확률 구간별 오답률 추정치 (사전값 1 - 구간 중앙, 사전 관측 prior 건, 라벨로 갱신,
        #                z > 0 이면 사후 평균 + z x 사후 표준편차). 추정 오답률이 낮은 구간부터 "도착률 x 오답률" 합이
        #                한도에 찰 때까지 열고, 통과시킨 요청의 비용 합(최신 추정치로 다시 계산)이 "요청 수 x 한도"를
        #                넘으면 막는다. 통과 요청의 probe 비율은 라벨을 얻으려 LLM 으로 보낸다.
        #   label_only : 한도가 LLM 으로 돌린 요청(막힘 · 구간 닫힘 · probe)은 라벨을 한도 추정에만 쓰고
        #                vCache 관측 · 캐시 추가를 하지 않는다 → 캐시 상태 경로가 판정자만 쓴 경우(CJ)와 같다
        self.consensus_online_budget = (
            float(consensus_online_budget) if consensus_online_budget is not None else None
        )
        self.consensus_online_mode = str(consensus_online_mode)
        self._ob_label_only = bool(consensus_online_label_only)
        if self.consensus_online_budget is not None:
            if consensus_model is None:
                raise ValueError("consensus_online_budget needs consensus_model (judge with p*)")
            if not 0.0 < self.consensus_online_budget < 1.0:
                raise ValueError("consensus_online_budget must be in (0, 1)")
            if self.consensus_online_mode not in ("cap", "calibrated"):
                raise ValueError(f"Unsupported consensus_online_mode={consensus_online_mode!r}")
            if (not 0.0 < float(consensus_online_bin_width) <= 1.0 or float(consensus_online_prior) < 0.0
                    or not 0.0 <= float(consensus_online_probe) < 1.0 or float(consensus_online_z) < 0.0):
                raise ValueError("bad consensus_online_bin_width / prior / probe / z")
            self._ob_pstar = float(consensus_model["min_probability"])
            self._ob_probe = float(consensus_online_probe)
            self._ob_z = float(consensus_online_z)
            self._ob_skipped_updates = 0
            # 전용 생성기: vCache 의 전역 random 과 섞이면 기준선 판정까지 바뀐다
            self._ob_rng = np.random.default_rng(int(consensus_seed) + 7919)
            width = float(consensus_online_bin_width)
            n_bins = max(1, int(np.ceil((1.0 - self._ob_pstar) / width - 1e-9)))
            self._ob_width = width
            self._ob_prior = float(consensus_online_prior)
            self._ob_lo = self._ob_pstar + width * np.arange(n_bins)
            self._ob_mid = np.minimum(self._ob_lo + width / 2.0, (self._ob_lo + 1.0) / 2.0)
            self._ob_arrivals = np.zeros(n_bins, dtype=np.int64)   # p >= p* 로 한도 검사에 온 요청
            self._ob_accepted = np.zeros(n_bins, dtype=np.int64)
            self._ob_probed = np.zeros(n_bins, dtype=np.int64)     # 허용 구간인데 라벨용으로 LLM 호출
            self._ob_label_n = np.zeros(n_bins, dtype=np.int64)    # LLM 으로 돌려 라벨을 얻은 요청
            self._ob_label_err = np.zeros(n_bins, dtype=np.int64)
            self._ob_requests = 0          # 지금까지 들어온 요청 수 (후보 없는 요청 포함)
            self._ob_spent = 0.0           # 통과시킨 요청의 비용 합 (기대 추가 오답)
            self._ob_spent_model = 0.0     # 같은 요청들의 1 - p 합 (비교용)
        elif self._ob_label_only:
            raise ValueError("consensus_online_label_only needs consensus_online_budget")
        self._consensus_answer_embeddings: dict = {}
        self.similarity_evaluator: Optional[SimilarityEvaluator] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.cache: Optional[Cache] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.executor: Optional[ThreadPoolExecutor] = None
        self.callback_queue: Optional[CallbackQueue] = None

        # RL splitter instance (expected: vcache.vcache_core.splitter.MaxSimSplitter.MaxSimSplitter)
        self.splitter = splitter
        self.device = device
        # If True, use 0.5*(MaxSim + cosine(full_embed_no_cls)) as the similarity score (both in [0,1]).
        # If False, use MaxSim only.
        self.mix_fullcos = bool(mix_fullcos)
        # How to choose which cached prompts to score with MaxSim:
        # - "top_k": get k candidates from the vector DB (fast), then rerank by MaxSim (accurate).
        # - "all": score against all cached prompts by MaxSim (slow, but MaxSim everywhere).
        # - "multivector_top_k": use policy-local multivector HNSW over cached segment vectors (parent_id=doc),
        #   then rerank by MaxSim.
        self.candidate_selection = candidate_selection
        self.candidate_k = candidate_k
        # If True, cache a candidate's MaxSim segment-embedding tensor in metadata at insert time
        # (or lazily on first use), and at query time only segment the query (single-text forward).
        self.use_cached_candidate_segments = bool(use_cached_candidate_segments) or (
            self.candidate_selection == "multivector_top_k"
        )

        # Optional policy-local multivector HNSW index (only used when candidate_selection=multivector_top_k).
        self._mv_index: Optional[_MultiVectorHNSWIndex] = None
        self._mv_max_elements: int = int(multivector_max_elements)

        # Optional timing collector. Expected interface:
        #   - add(name: str, dt_seconds: float) -> None
        # When enabled, we record wall times for major stages (HNSW retrieval, splitter/model compute, etc.).
        self._timing = timing_collector
        # If True and device is CUDA, synchronize around timed regions for more accurate GPU timing.
        self._timing_cuda_sync = bool(timing_cuda_sync)

    @contextmanager
    def _time_block(self, name: str):
        """
        Lightweight timing context. No-op if no collector is set.
        """
        if self._timing is None:
            yield
            return

        def _maybe_sync():
            if not self._timing_cuda_sync:
                return
            try:
                import torch

                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.synchronize()
            except Exception:
                pass

        _maybe_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _maybe_sync()
            dt = max(0.0, time.perf_counter() - t0)
            try:
                self._timing.add(str(name), float(dt))
            except Exception:
                pass

    # [추가: prefix-stress-test] 분할기가 segment_weights(텐서)를 주면 조각별 가중치로 MaxSim을 계산한다.
    #   없으면(원래 RL 분할기) 원본과 똑같이 균등 가중치.
    def _score_tensors(self, query_tensor, corpus_tensor) -> float:
        weights_fn = getattr(self.splitter, "segment_weights", None)
        if weights_fn is None:
            return self._maxsim_from_tensors(query_tensor, corpus_tensor)
        return self._maxsim_from_tensors(
            query_tensor, corpus_tensor,
            query_weights=weights_fn(query_tensor), corpus_weights=weights_fn(corpus_tensor),
        )

    @staticmethod
    def _maxsim_from_tensors(query_tensor, corpus_tensor, query_weights=None, corpus_weights=None) -> float:
        """
        Compute symmetric MaxSim similarity and return it in **[0, 1]** given:
          - query_tensor:  [S_q, H] (segment embeddings)
          - corpus_tensor: [S_c, H] (segment embeddings)
        """
        import torch
        import torch.nn.functional as F

        dev = query_tensor.device

        qt = query_tensor.to(dtype=torch.float32)
        ct = corpus_tensor.to(dtype=torch.float32, device=qt.device)

        # MaxSim over all segment rows.
        query_vectors = qt
        corpus_vectors = ct
        if query_vectors.shape[0] > 0 and corpus_vectors.shape[0] > 0:
            qn = F.normalize(query_vectors, p=2, dim=-1)
            cn = F.normalize(corpus_vectors, p=2, dim=-1)
            cos = torch.mm(qn, cn.T)
            max_cos_sim_row = torch.max(cos, dim=1).values  # [Q]
            max_cos_sim_col = torch.max(cos, dim=0).values  # [C]

            # Weighted version (training-style); uniform unless the caller supplies weights.
            # [수정: prefix-stress-test] 가중치가 주어지면 그 값을 쓴다 (가중 평균이라 결과는 그대로 [0, 1])
            if query_weights is None:
                query_weights = torch.ones(max_cos_sim_row.shape[0], device=dev, dtype=torch.float32)
            if corpus_weights is None:
                corpus_weights = torch.ones(max_cos_sim_col.shape[0], device=dev, dtype=torch.float32)
            query_weights = query_weights.to(device=dev, dtype=torch.float32)
            corpus_weights = corpus_weights.to(device=dev, dtype=torch.float32)
            row_score = torch.sum(max_cos_sim_row * query_weights) / (torch.sum(query_weights) + 1e-8)
            col_score = torch.sum(max_cos_sim_col * corpus_weights) / (torch.sum(corpus_weights) + 1e-8)
        else:
            row_score = torch.tensor(0.0, device=dev)
            col_score = torch.tensor(0.0, device=dev)

        # Weighted mix between row/col (uniform by default).
        mix = torch.softmax(torch.tensor([0.0, 0.0], device=dev, dtype=torch.float32), dim=0)
        raw = mix[0] * row_score + mix[1] * col_score
        # Map cosine-like similarity from [-1, 1] to [0, 1] to match the policy/Bayesian assumptions.
        s01 = ((raw + 1.0) * 0.5).clamp(0.0, 1.0)
        return float(s01.item())

    @staticmethod
    def _cos01(a: "torch.Tensor", b: "torch.Tensor") -> float:
        """Cosine similarity mapped to [0,1]. Accepts [H] or [1,H] tensors."""
        import torch
        import torch.nn.functional as F

        aa = a
        bb = b
        if aa.dim() == 1:
            aa = aa.unsqueeze(0)
        if bb.dim() == 1:
            bb = bb.unsqueeze(0)
        # Compute cosine in [-1,1], then map to [0,1]
        cos = F.cosine_similarity(F.normalize(aa.float(), p=2, dim=-1), F.normalize(bb.float(), p=2, dim=-1), dim=-1)
        s01 = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)
        return float(s01.mean().item())

    def _maybe_cache_candidate_segments(self, embedding_id: int) -> None:
        """
        Ensure metadata for `embedding_id` has `cached_maxsim_tensor` populated.
        Safe no-op if caching is disabled or prerequisites are missing.
        """
        if not self.use_cached_candidate_segments:
            return
        if self.cache is None or self.splitter is None:
            return
        try:
            meta = self.cache.get_metadata(embedding_id=embedding_id)
        except Exception:
            return
        if meta is None:
            return
        if getattr(meta, "cached_maxsim_tensor", None) is not None:
            return

        cached_prompt = getattr(meta, "prompt", "") or ""
        if not cached_prompt:
            return
        try:
            # One-time compute per candidate prompt
            cand_tensor = self.splitter.split_text_return_maxsim_tensor(cached_prompt)
            meta.cached_maxsim_tensor = cand_tensor.detach()
            self.cache.update_metadata(embedding_id=embedding_id, embedding_metadata=meta)
            self._maybe_index_candidate_multivector(embedding_id=embedding_id, meta=meta)
        except Exception as e:
            # Keep serving even if caching fails for an entry
            self.logger.warning(f"Candidate segment caching failed for embedding_id={embedding_id}: {e}")

    def _maybe_index_candidate_multivector(
        self, *, embedding_id: int, meta: Optional[EmbeddingMetadataObj] = None
    ) -> None:
        """
        If multivector candidate selection is enabled, index this candidate's cached vectors.
        Requires `meta.cached_maxsim_tensor` to be populated.
        """
        if self.candidate_selection != "multivector_top_k":
            return
        if self._mv_index is None:
            return
        if self.cache is None:
            return
        if meta is None:
            try:
                meta = self.cache.get_metadata(embedding_id=embedding_id)
            except Exception:
                return
        if meta is None:
            return
        cand_tensor = getattr(meta, "cached_maxsim_tensor", None)
        if cand_tensor is None:
            return
        try:
            if hasattr(cand_tensor, "detach"):
                x = cand_tensor.detach().float().cpu().numpy()
            else:
                x = np.asarray(cand_tensor, dtype=np.float32)
            self._mv_index.add_parent_vectors(parent_id=int(embedding_id), vectors=x)
        except Exception as e:
            self.logger.warning(f"Multivector indexing failed for embedding_id={embedding_id}: {e}")

    def __cache_add(self, prompt: str, response: str, id_set: int) -> int:
        """
        Add (prompt,response) to cache and optionally precompute candidate segment embeddings.
        Returns embedding_id.
        """
        if self.cache is None:
            return -1
        embedding_id = self.cache.add(prompt=prompt, response=response, id_set=id_set)
        if embedding_id >= 0:
            self._maybe_cache_candidate_segments(embedding_id=embedding_id)
        return embedding_id

    def setup(self, config: VCacheConfig):
        self.similarity_evaluator = config.similarity_evaluator
        self.inference_engine = config.inference_engine
        self.cache = Cache(
            embedding_engine=config.embedding_engine,
            embedding_store=EmbeddingStore(
                embedding_metadata_storage=config.embedding_metadata_storage,
                vector_db=config.vector_db,
            ),
            eviction_policy=config.eviction_policy,
        )

        self.callback_queue = CallbackQueue(callback_function=self.__perform_cache_update)
        self.callback_queue.start()
        self.executor = ThreadPoolExecutor(max_workers=64)

        if self.candidate_selection == "multivector_top_k":
            self._mv_index = _MultiVectorHNSWIndex(max_elements=self._mv_max_elements)

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.callback_queue:
            self.callback_queue.stop()

    def _maxsim_similarity(self, query: str, candidate: str) -> float:
        """
        Compute MaxSim similarity in [0, 1] using the provided splitter + its embedding model.
        """
        if self.splitter is None:
            raise ValueError(
                "VerifiedSplitterDecisionPolicy requires `splitter` (MaxSimSplitter) to be provided."
            )

        try:
            import torch
            import torch.nn.functional as F
        except Exception as e:
            raise RuntimeError(
                "VerifiedSplitterDecisionPolicy requires torch to compute MaxSim similarity."
            ) from e

        if not bool(getattr(self, "mix_fullcos", False)):
            query_tensor, corpus_tensor = self.splitter.split_pair_return_maxsim_tensors(
                query, candidate
            )
            return self._score_tensors(query_tensor, corpus_tensor)

        # Mixed mode: compute MaxSim + cosine(full_embed_no_cls)
        qenc = self.splitter.encode_text(query)
        cenc = self.splitter.encode_text(candidate)
        query_tensor, corpus_tensor = self.splitter.split_pair_return_maxsim_tensors_from_encoded(qenc, cenc)
        maxsim01 = self._score_tensors(query_tensor, corpus_tensor)
        fullcos01 = self._cos01(qenc["pooled_no_cls"], cenc["pooled_no_cls"])
        return 0.5 * (float(maxsim01) + float(fullcos01))

    def _maxsim_similarity_from_encoded(self, query_enc: dict, candidate_prompt: str) -> float:
        """
        MaxSim similarity where the query has already been encoded once (token embeddings + pooled embedding).
        We still need to encode the candidate prompt once to get its token embeddings for the RL splitter.
        """
        if self.splitter is None:
            raise ValueError("VerifiedSplitterDecisionPolicy requires `splitter` (MaxSimSplitter) to be provided.")
        import torch
        import torch.nn.functional as F

        # Encode candidate once (token-level). This is the part you may want to report separately.
        t0 = time.time()
        cand_enc = self.splitter.encode_text(candidate_prompt)
        dt = time.time() - t0
        # Best-effort accumulate timing for reporting (does not affect correctness)
        try:
            self._timing_candidate_encode_s = getattr(self, "_timing_candidate_encode_s", 0.0) + float(dt)
            self._timing_candidate_encode_n = getattr(self, "_timing_candidate_encode_n", 0) + 1
        except Exception:
            pass

        query_tensor, corpus_tensor = self.splitter.split_pair_return_maxsim_tensors_from_encoded(query_enc, cand_enc)
        maxsim01 = self._score_tensors(query_tensor, corpus_tensor)
        if not bool(getattr(self, "mix_fullcos", False)):
            return float(maxsim01)
        fullcos01 = self._cos01(query_enc["pooled_no_cls"], cand_enc["pooled_no_cls"])
        return 0.5 * (float(maxsim01) + float(fullcos01))

    def _select_nn_by_maxsim(
        self, prompt: str
    ) -> Tuple[Optional[EmbeddingMetadataObj], Optional[float]]:
        """
        Select the nearest-neighbor metadata object using MaxSim similarity.

        Returns:
            (best_metadata, best_similarity) where similarity is in [0, 1].
        """
        if self.cache is None:
            return None, None

        candidates: list[EmbeddingMetadataObj] = []

        if self.candidate_selection == "all":
            candidates = self.cache.get_all_embedding_metadata_objects()
        elif self.candidate_selection == "top_k":
            knn = self.cache.get_knn(prompt=prompt, k=max(1, int(self.candidate_k)))
            for _db_score, embedding_id in knn:
                try:
                    candidates.append(self.cache.get_metadata(embedding_id=embedding_id))
                except Exception:
                    continue
        else:
            raise ValueError(
                f"Unknown candidate_selection={self.candidate_selection!r}. Use 'top_k' or 'all'."
            )

        if not candidates:
            return None, None

        best_meta: Optional[EmbeddingMetadataObj] = None
        best_s: float = -1.0

        for meta in candidates:
            cached_prompt = getattr(meta, "prompt", "") or ""
            if not cached_prompt:
                # Can't compute MaxSim without cached prompt text; skip.
                continue
            try:
                s = self._maxsim_similarity(prompt, cached_prompt)
            except Exception as e:
                self.logger.warning(f"MaxSim similarity failed for one candidate: {e}")
                continue
            if s > best_s:
                best_s = s
                best_meta = meta

        if best_meta is None:
            return None, None
        return best_meta, best_s

    def process_request(
        self, prompt: str, system_prompt: Optional[str], id_set: int
    ) -> Tuple[bool, str, EmbeddingMetadataObj]:
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")

        # Optimized path: encode query ONCE and reuse it for:
        # - KNN selection (pooled embedding)
        # - MaxSim/RL (token embeddings)
        if self.splitter is None:
            raise ValueError("VerifiedSplitterDecisionPolicy requires `splitter` (MaxSimSplitter) to be provided.")
        if self.consensus_online_budget is not None:
            self._ob_requests += 1

        # "embedding time": query encoding (LM forward + pooling) inside MaxSimSplitter
        with self._time_block("splitter.encode_text"):
            query_enc = self.splitter.encode_text(prompt)
        query_knn_emb = query_enc["pooled_knn"]  # torch tensor on device
        # HNSW runs on CPU; pass a CPU list[float]
        query_knn_emb_cpu = query_knn_emb.detach().float().cpu().tolist()

        # NOTE: don't wrap this call in its own timing block because it already contains
        # fine-grained timers (wrapping would double-count in totals).
        nn_metadata, similarity_score = self._select_nn_by_maxsim_with_query(
            prompt, query_enc, query_knn_emb_cpu
        )
        if nn_metadata is None or similarity_score is None:
            response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
            self.__cache_add(prompt=prompt, response=response, id_set=id_set)
            return False, response, EmbeddingMetadataObj(embedding_id=-1, response="")

        action = self.bayesian.select_action(similarity_score=similarity_score, metadata=nn_metadata)
        if self.consensus_model is not None:
            action = self._apply_consensus_judge(action, similarity_score, nn_metadata)

        match action:
            case _Action.EXPLOIT:
                return True, nn_metadata.response, nn_metadata
            case _Action.EXPLORE:
                response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
                pending = getattr(nn_metadata, "_ob_pending_bin", None)
                if pending is not None and self.similarity_evaluator is not None:
                    # 한도가 LLM 으로 돌린 요청: LLM 응답으로 1등 후보가 맞았는지 안다 (vCache 관측과 같은 판정)
                    correct = self.similarity_evaluator.answers_similar(
                        a=response, b=nn_metadata.response,
                        id_set_a=id_set, id_set_b=nn_metadata.id_set,
                    )
                    self._ob_label_n[pending] += 1
                    self._ob_label_err[pending] += int(not correct)
                    nn_metadata._ob_pending_bin = None
                    if self._ob_label_only:
                        self._ob_skipped_updates += 1
                        return False, response, nn_metadata
                self.__update_cache(
                    response=response,
                    nn_metadata=nn_metadata,
                    similarity_score=similarity_score,
                    embedding_id=nn_metadata.embedding_id,
                    prompt=prompt,
                    label_id_set=id_set,
                )
                return False, response, nn_metadata

    def _select_nn_by_maxsim_with_query(
        self, prompt: str, query_enc: dict, query_knn_emb_cpu: list[float]
    ) -> Tuple[Optional[EmbeddingMetadataObj], Optional[float]]:
        """
        Like `_select_nn_by_maxsim`, but reuses the already-encoded query:
          - uses pooled embedding for vector DB KNN
          - uses query token embeddings for MaxSim/RL scoring
        """
        if self.cache is None:
            return None, None

        candidates: list[EmbeddingMetadataObj] = []
        if self.candidate_selection == "all":
            with self._time_block("candidate_selection.all_scan"):
                candidates = self.cache.get_all_embedding_metadata_objects()
        elif self.candidate_selection == "top_k":
            with self._time_block("candidate_selection.hnsw_top_k"):
                knn = self.cache.get_knn_from_embedding(
                    embedding=query_knn_emb_cpu, k=max(1, int(self.candidate_k))
                )
            for _db_score, embedding_id in knn:
                try:
                    candidates.append(self.cache.get_metadata(embedding_id=embedding_id))
                except Exception:
                    continue
        elif self.candidate_selection == "multivector_top_k":
            if self.splitter is None or self._mv_index is None:
                return None, None
            # Build query multivector tensor (segments + full) once, then query parents per vector.
            try:
                with self._time_block("splitter.query_multivector_build"):
                    query_tensor_mv = self.splitter.split_text_return_maxsim_tensor_from_encoded(query_enc)
                    qv = query_tensor_mv.detach().float().cpu().numpy()
            except Exception as e:
                self.logger.warning(f"Query multivector tensor build failed: {e}")
                return None, None

            with self._time_block("candidate_selection.multivector_hnsw"):
                parent_ids = self._mv_index.query_candidate_parents(
                    query_vectors=qv, k_per_vector=max(1, int(self.candidate_k))
                )
            for pid in parent_ids:
                try:
                    candidates.append(self.cache.get_metadata(embedding_id=int(pid)))
                except Exception:
                    continue
        else:
            raise ValueError(
                f"Unknown candidate_selection={self.candidate_selection!r}. "
                "Use 'top_k', 'all', or 'multivector_top_k'."
            )

        if not candidates:
            return None, None

        # If we are using cached candidate segment tensors, compute the query tensor ONCE.
        query_tensor = None
        if self.use_cached_candidate_segments and self.splitter is not None:
            try:
                with self._time_block("splitter.query_tensor_build"):
                    query_tensor = self.splitter.split_text_return_maxsim_tensor_from_encoded(query_enc)
            except Exception as e:
                self.logger.warning(f"Query single-text segmentation failed; falling back to pair MaxSim path: {e}")
                query_tensor = None

        best_meta: Optional[EmbeddingMetadataObj] = None
        best_s: float = -1.0
        scored_candidates: list = []  # [판정자] 점수를 매긴 후보 전부 (합의 계산용)

        for meta in candidates:
            cached_prompt = getattr(meta, "prompt", "") or ""
            if not cached_prompt:
                continue
            try:
                if query_tensor is not None:
                    cand_tensor = getattr(meta, "cached_maxsim_tensor", None)
                    if cand_tensor is None:
                        # Lazy-fill cache for this candidate (still avoids re-encoding it next time)
                        try:
                            with self._time_block("splitter.candidate_tensor_build"):
                                cand_tensor = self.splitter.split_text_return_maxsim_tensor(cached_prompt)
                            meta.cached_maxsim_tensor = cand_tensor.detach()
                            self.cache.update_metadata(embedding_id=meta.embedding_id, embedding_metadata=meta)
                            self._maybe_index_candidate_multivector(
                                embedding_id=meta.embedding_id, meta=meta
                            )
                        except Exception:
                            cand_tensor = None
                    if cand_tensor is None:
                        # Fall back to existing encoded-pair path for correctness
                        with self._time_block("maxsim.score_pair"):
                            s = self._maxsim_similarity_from_encoded(query_enc, cached_prompt)
                    else:
                        with self._time_block("maxsim.score_tensors"):
                            s = self._score_tensors(query_tensor, cand_tensor)
                else:
                    with self._time_block("maxsim.score_pair"):
                        s = self._maxsim_similarity_from_encoded(query_enc, cached_prompt)
            except Exception as e:
                self.logger.warning(f"MaxSim similarity failed for one candidate: {e}")
                continue
            scored_candidates.append(meta)
            if s > best_s:
                best_s = s
                best_meta = meta

        if best_meta is None:
            return None, None
        if self.collect_consensus:
            self._attach_consensus_features(best_meta, best_s, scored_candidates)
        return best_meta, best_s

    # ------------------------------------------------------------------ [판정자]
    def _attach_consensus_features(self, best_meta, best_s, scored_candidates) -> None:
        """후보 수와 1등을 뺀 후보들의 답 합의 a_excl을 1등 메타데이터에 붙인다 (라벨 미사용)."""
        try:
            count, agreement = top1_excluded_agreement(
                best_meta.response,
                [m.response for m in scored_candidates],
                embed_texts=self._embed_consensus_answers,
                embedding_threshold=self.consensus_answer_threshold,
            )
        except Exception as e:  # 합의 계산 실패 시 판정자는 기준선 판정을 그대로 둔다
            self.logger.warning(f"consensus feature failed: {e}")
            count, agreement = len(scored_candidates), None
        best_meta._consensus_raw_score = float(best_s)
        best_meta._consensus_candidate_count = int(count)
        best_meta._consensus_agreement_excl = agreement

    def _embed_consensus_answers(self, texts: list) -> np.ndarray:
        """80자 넘는 답끼리 비교할 때만 쓰는 BGE 문장 벡터 (답 문자열별로 기억)."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._consensus_answer_embeddings]
        for t in missing:
            v = self.splitter.encode_text(t)["pooled_knn"]
            v = v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=np.float32)
            self._consensus_answer_embeddings[t] = np.asarray(v, dtype=np.float32).reshape(-1)
        return np.stack([self._consensus_answer_embeddings[t] for t in texts])

    def _apply_consensus_judge(self, action, similarity_score, nn_metadata):
        """vCache 판정(action)을 판정자 확률로 덮어쓴다. 판정 이유는 메타데이터에 남긴다."""
        count = int(getattr(nn_metadata, "_consensus_candidate_count", 0) or 0)
        agreement = getattr(nn_metadata, "_consensus_agreement_excl", None)
        n_obs = len(getattr(nn_metadata, "observations", []) or [])
        is_cold = n_obs < 6
        reason = "baseline_exploit" if action == _Action.EXPLOIT else "baseline_fallback"
        probability = None
        nn_metadata._ob_pending_bin = None      # 메타데이터 객체는 요청 간에 재사용된다
        if action != _Action.EXPLOIT:
            if self.consensus_scope == "cold" and not is_cold:
                reason = "not_cold"
            elif count < self.consensus_min_candidates or agreement is None:
                reason = "candidate_count_lt_min"
            else:
                probability = predict_correct_probability(
                    similarity_score, agreement, count, self.consensus_model
                )
                accept = probability >= float(self.consensus_model["min_probability"])
                if accept and self.consensus_online_budget is not None:
                    accept, k, why = self._online_budget_admit(probability)
                    if not accept:
                        reason = why
                        nn_metadata._ob_pending_bin = k
                if accept:
                    action = _Action.EXPLOIT
                    reason = "judge_cold" if is_cold else "judge_explore"
        nn_metadata._consensus_probability = probability
        nn_metadata._consensus_decision_score = probability
        nn_metadata._consensus_action = action.name
        nn_metadata._consensus_reason = reason
        return action

    def _ob_error_estimate(self) -> np.ndarray:
        # 구간별 오답률 추정: 판정자 예측(1 - 구간 중앙)을 prior 건수만큼의 사전 관측으로 두고 라벨로 갱신
        m = self._ob_prior
        prior = 1.0 - self._ob_mid
        n = self._ob_label_n + m
        mean = (self._ob_label_err + m * prior) / np.maximum(n, 1e-12)
        if self._ob_z <= 0.0:
            return mean
        # Beta 사후 표준편차 근사: sqrt(mean (1 - mean) / (n + 1))
        sd = np.sqrt(np.clip(mean * (1.0 - mean), 0.0, None) / (n + 1.0))
        return np.minimum(1.0, mean + self._ob_z * sd)

    def _ob_admitted_bins(self, est: np.ndarray) -> np.ndarray:
        # 추정 오답률이 낮은 구간부터, 요청당 기대 추가 오답(도착률 x 오답률)의 합이 한도에 찰 때까지.
        # 경계 구간도 넣는다 — 넘치는 만큼은 누적 한도가 막는다.
        t = max(1, self._ob_requests)
        rate = self._ob_arrivals / t
        admitted = np.zeros(len(est), dtype=bool)
        cum = 0.0
        for k in np.argsort(est, kind="stable"):
            admitted[k] = True
            cum += rate[k] * est[k]
            if cum > self.consensus_online_budget:
                break
        return admitted

    def _online_budget_admit(self, probability: float) -> Tuple[bool, int, str]:
        k = int(min(len(self._ob_lo) - 1, max(0, (probability - self._ob_pstar) // self._ob_width)))
        self._ob_arrivals[k] += 1
        if self.consensus_online_mode == "cap":
            cost = 1.0 - float(probability)
        else:
            est = self._ob_error_estimate()
            if not self._ob_admitted_bins(est)[k]:
                return False, k, "online_budget_bin_closed"
            cost = float(est[k])
            # 이미 통과시킨 요청의 비용도 현재 추정치로 다시 센다
            self._ob_spent = float(np.dot(self._ob_accepted, est))
        if self._ob_spent + cost > self.consensus_online_budget * self._ob_requests:
            return False, k, "online_budget_blocked"
        if (self.consensus_online_mode == "calibrated" and self._ob_probe > 0.0
                and self._ob_rng.random() < self._ob_probe):
            self._ob_probed[k] += 1
            return False, k, "online_budget_probe"
        self._ob_spent += cost
        self._ob_spent_model += 1.0 - float(probability)
        self._ob_accepted[k] += 1
        return True, k, "accepted"

    def online_budget_state(self) -> Optional[dict]:
        if self.consensus_online_budget is None:
            return None
        est = self._ob_error_estimate()
        if self.consensus_online_mode == "calibrated":
            self._ob_spent = float(np.dot(self._ob_accepted, est))
        return {
            "budget": self.consensus_online_budget,
            "mode": self.consensus_online_mode,
            "pstar": self._ob_pstar,
            "bin_width": self._ob_width,
            "prior": self._ob_prior,
            "probe": self._ob_probe,
            "z": self._ob_z,
            "label_only": self._ob_label_only,
            "skipped_updates": int(self._ob_skipped_updates),
            "requests": int(self._ob_requests),
            "spent": float(self._ob_spent),
            "spent_model": float(self._ob_spent_model),
            "allowed": float(self.consensus_online_budget * self._ob_requests),
            "bins": [
                {
                    "lo": float(lo), "arrivals": int(a), "accepted": int(c), "probed": int(q),
                    "label_n": int(n), "label_err": int(e), "est_err": float(x),
                }
                for lo, a, c, q, n, e, x in zip(
                    self._ob_lo, self._ob_arrivals, self._ob_accepted, self._ob_probed,
                    self._ob_label_n, self._ob_label_err, est,
                )
            ],
            "admitted_final": [bool(v) for v in self._ob_admitted_bins(est)],
        }

    def __update_cache(
        self,
        response: str,
        nn_metadata: EmbeddingMetadataObj,
        similarity_score: float,
        embedding_id: int,
        prompt: str,
        label_id_set: int,
    ) -> None:
        if self.executor is None:
            raise ValueError("Executor not initialized. Call setup() first.")

        self.executor.submit(
            self.__submit_for_background_update,
            response,
            similarity_score,
            embedding_id,
            prompt,
            nn_metadata.response,
            label_id_set,
            nn_metadata.id_set,
        )

    def __submit_for_background_update(
        self,
        new_response: str,
        similarity_score: float,
        embedding_id: int,
        prompt: str,
        cached_response: str,
        label_id_set: int,
        nn_id_set: int,
    ):
        if self.similarity_evaluator is None or self.callback_queue is None:
            return

        should_have_exploited = self.similarity_evaluator.answers_similar(
            a=new_response, b=cached_response, id_set_a=label_id_set, id_set_b=nn_id_set
        )

        self.callback_queue.put(
            (
                should_have_exploited,
                new_response,
                similarity_score,
                embedding_id,
                prompt,
                label_id_set,
            )
        )

    def __perform_cache_update(self, update_args: tuple) -> None:
        (
            should_have_exploited,
            new_response,
            similarity_score,
            embedding_id,
            prompt,
            id_set,
        ) = update_args

        if self.cache is None:
            return

        try:
            latest_metadata_object = self.cache.get_metadata(embedding_id=embedding_id)
        except (ValueError, KeyError):
            return

        if latest_metadata_object is None:
            return

        try:
            self.bayesian.add_observation_to_metadata(
                similarity_score=similarity_score,
                is_correct=should_have_exploited,
                metadata=latest_metadata_object,
            )
        except (ValueError, KeyError):
            return

        if not should_have_exploited:
            self.__cache_add(prompt=prompt, response=new_response, id_set=id_set)

        try:
            self.cache.update_metadata(
                embedding_id=embedding_id, embedding_metadata=latest_metadata_object
            )
        except (ValueError, KeyError):
            return


