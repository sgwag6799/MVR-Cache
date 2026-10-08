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

PAD_PARENT_ID = np.iinfo(np.uint64).max


class _LoggedAlgorithm(_Algorithm):
    """The verified explore/exploit rule of _Algorithm.select_action, unchanged, but it keeps
    what the decision used in `last_info` (log B7) and can draw u from a seeded generator.

    With seed=None, u comes from the global `random` module exactly as in _Algorithm.
    """
    # [로그] vCache 판정(로지스틱 함수)의 내부 값을 남기기 위한 클래스.
    #        판정 규칙은 verified.py의 _Algorithm.select_action과 똑같이 옮겨 왔고, 값만 기록한다.

    def __init__(self, delta: float, seed: Optional[int] = None):
        super().__init__(delta=delta)
        # seed를 주면 탐색/활용 난수 u를 고정된 생성기에서 뽑아 실행을 재현할 수 있다
        self._rng = random.Random(seed) if seed is not None else None
        self.last_info: Optional[dict] = None

    def select_action(self, similarity_score: float, metadata: EmbeddingMetadataObj) -> _Action:
        similarity_score = round(similarity_score, 3)
        # 판정 전 값 (이전 판정에서 적합된 t̂, γ)
        info = {
            "s_rounded": similarity_score,
            "n_obs": len(metadata.observations),
            "t_hat_before": metadata.t_hat,
            "gamma_before": metadata.gamma,
            "var_t_before": getattr(metadata, "var_t", None),
            "t_prime_before": metadata.t_prime,
            "t_hat": None, "gamma": None, "var_t": None, "t_prime": None,
            "alpha": None, "alpha_tprime": None, "tau": None, "u": None, "reason": None,
        }
        self.last_info = info
        similarities = np.array([obs[0] for obs in metadata.observations])
        labels = np.array([obs[1] for obs in metadata.observations])

        if len(similarities) < 6 or len(labels) < 6:
            info["reason"] = "few_observations"  # 관측 6개 미만 → 무조건 탐색
            return _Action.EXPLORE

        t_hat, gamma, var_t = self._estimate_parameters(similarities=similarities, labels=labels)
        if t_hat == -1:
            info["reason"] = "fit_failed"  # 로지스틱 적합 실패 → 탐색
            return _Action.EXPLORE
        metadata.gamma = gamma
        metadata.t_hat = t_hat
        metadata.var_t = var_t

        start_time = time.time()
        tau = self._get_tau(var_t=var_t, s=similarity_score, t_hat=t_hat, metadata=metadata)
        self.tau_latencies.append(time.time() - start_time)

        u = self._rng.uniform(0, 1) if self._rng is not None else random.uniform(0, 1)
        # 판정 후 값: 이번에 적합한 t̂, γ, 분산, 신뢰구간 보정 임계값 t', τ, u
        #   alpha        = σ(γ(s − t̂))  (보정 전 임계값 기준)
        #   alpha_tprime = σ(γ(s − t′))  (τ 계산에 실제로 쓰인 보정 임계값 기준)
        t_prime = None if metadata.t_prime is None else float(metadata.t_prime)
        info.update(
            t_hat=float(t_hat), gamma=float(gamma), var_t=float(var_t),
            t_prime=t_prime,
            alpha=float(1.0 / (1.0 + np.exp(-gamma * (similarity_score - t_hat)))),
            alpha_tprime=None if t_prime is None else float(1.0 / (1.0 + np.exp(-gamma * (similarity_score - t_prime)))),
            tau=float(tau), u=float(u), reason="tau",
        )
        if u <= tau:
            return _Action.EXPLORE
        return _Action.EXPLOIT


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
        detail_log: bool = False,
        seed: Optional[int] = None,
    ):
        # [로그] _LoggedAlgorithm은 판정 규칙이 _Algorithm과 같고, 판정에 쓴 값만 추가로 기록한다
        self.bayesian = _LoggedAlgorithm(delta=delta, seed=seed)
        self.similarity_evaluator: Optional[SimilarityEvaluator] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.cache: Optional[Cache] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.executor: Optional[ThreadPoolExecutor] = None
        self.callback_queue: Optional[CallbackQueue] = None
        # Set by process_request: the similarity score and fit behind the latest decision.
        # [추가] 가장 최근 요청에서 vCache가 본 값(s, 판정, t̂, γ 등)을 담아두는 변수.
        #        평가 스크립트가 --log-scores 옵션을 쓰면 요청마다 이 값을 읽어 파일로 저장한다.
        self.last_decision: Optional[dict] = None
        # [로그] detail_log=True면 요청마다 후보별 점수·판정 내부값·단계별 시간을 last_detail에 남긴다 (실험 로그 B)
        self.detail_log = bool(detail_log)
        self.last_detail: Optional[dict] = None
        self._req_timing: dict = {}
        self._cand_log = None
        self._query_tensor_log = None
        # [로그] 평가 스크립트가 True로 바꾸면 다음 요청 하나에 대해 샘플 진단(C)을 계산해 last_diag에 남긴다
        self.diag_next = False
        self.last_diag: Optional[dict] = None
        # [로그] 이번 요청의 진단(C) 소요 시간(초). 평가 스크립트가 요청 시간(total)에서 빼고 timing_ms.diag로 따로 기록
        self.last_diag_s: float = 0.0
        self._in_diag = False
        self._diag_vecs = np.zeros((0, 0), dtype=np.float32)  # 진단용: HNSW에 저장된 문장 벡터 복사본
        # [로그] 관측 추가 기록(updates.jsonl). 평가 스크립트가 요청 번호(request_order)와 기록 함수(update_hook)를 넣는다.
        #        update_hook은 백그라운드 스레드에서 불린다.
        self.request_order: Optional[int] = None
        self.update_hook = None
        self._main_thread_id: Optional[int] = None

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
        if self._timing is None and not self.detail_log:
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
            self._add_req_timing(str(name), float(dt))
            if self._timing is not None and not self._in_diag:
                try:
                    self._timing.add(str(name), float(dt))
                except Exception:
                    pass

    def _add_req_timing(self, name: str, dt: float) -> None:
        # [로그] 이번 요청 하나의 단계별 시간 합계 (초).
        #        진단(C) 중의 시간과, 백그라운드 스레드(캐시 추가 시 조각 계산)의 시간은 이 요청의 단계별 시간에 넣지 않는다.
        if not self.detail_log or self._in_diag:
            return
        if self._main_thread_id is not None and threading.get_ident() != self._main_thread_id:
            return
        self._req_timing[name] = self._req_timing.get(name, 0.0) + dt

    # [추가] 점수 계산의 공통 입구. 기존에 _maxsim_from_tensors를 직접 부르던 4곳이 모두 이 함수를 거친다.
    #        분할기가 조각별 가중치 함수(segment_weights)를 가지고 있으면 가중치를 넣어 MaxSim을 계산하고,
    #        없으면(원래 RL 분할기 등) 기존과 똑같이 균등 가중치로 계산한다.
    def _score_tensors(self, query_tensor, corpus_tensor) -> float:
        """MaxSim with per-row weights when the splitter provides them (uniform otherwise)."""
        # 분할기에 segment_weights 함수가 있는지 확인 (RulePunctuationSplitter / SegmentRowWeighter에 있음)
        weights_fn = getattr(self.splitter, "segment_weights", None)
        if weights_fn is None:
            # 가중치 기능이 없는 분할기 → 원래 코드와 완전히 같은 계산
            return self._maxsim_from_tensors(query_tensor, corpus_tensor)
        # 질문(query)과 후보(corpus) 각각의 조각 텐서로 가중치를 조회해서 함께 넘긴다
        return self._maxsim_from_tensors(
            query_tensor,
            corpus_tensor,
            query_weights=weights_fn(query_tensor),
            corpus_weights=weights_fn(corpus_tensor),
        )

    @staticmethod
    # [수정] query_weights / corpus_weights 인자를 추가. 안 넘기면(None) 원래처럼 균등 가중치.
    def _maxsim_from_tensors(query_tensor, corpus_tensor, query_weights=None, corpus_weights=None) -> float:
        """
        Compute symmetric MaxSim similarity and return it in **[0, 1]** given:
          - query_tensor:  [S_q, H] (segment embeddings)
          - corpus_tensor: [S_c, H] (segment embeddings)
          - query_weights / corpus_weights: optional non-negative per-row weights [S_q] / [S_c];
            uniform when omitted. Scores are weighted means, so the result stays in [0, 1].
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
            # [수정] 원래는 여기서 항상 torch.ones(전부 1)로 균등 가중치를 썼다.
            #        이제는 가중치가 들어오지 않았을 때만 1로 채우고, 들어오면 그 값을 쓴다.
            # row_score = 질문 조각마다 "후보에서 가장 비슷한 조각과의 코사인"을 가중 평균
            # col_score = 후보 조각마다 "질문에서 가장 비슷한 조각과의 코사인"을 가중 평균
            if query_weights is None:
                query_weights = torch.ones(max_cos_sim_row.shape[0], device=dev, dtype=torch.float32)
            if corpus_weights is None:
                corpus_weights = torch.ones(max_cos_sim_col.shape[0], device=dev, dtype=torch.float32)
            # 가중치를 계산 텐서와 같은 장치(CPU/GPU)와 자료형(float32)으로 맞춘다
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
        # [로그] 분할기가 단계별 시간 훅을 지원하면(규칙 분할기) 요청별 시간 기록에 연결
        if self.detail_log and hasattr(self.splitter, "timing_hook"):
            self.splitter.timing_hook = self._add_req_timing

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
            # [수정] _maxsim_from_tensors 직접 호출 → _score_tensors 경유 (가중치 적용)
            return self._score_tensors(query_tensor, corpus_tensor)

        # Mixed mode: compute MaxSim + cosine(full_embed_no_cls)
        qenc = self.splitter.encode_text(query)
        cenc = self.splitter.encode_text(candidate)
        query_tensor, corpus_tensor = self.splitter.split_pair_return_maxsim_tensors_from_encoded(qenc, cenc)
        # [수정] 가중치 적용을 위해 _score_tensors 경유
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
        # [수정] 가중치 적용을 위해 _score_tensors 경유
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

        # [로그] 요청마다 기록 초기화
        self._main_thread_id = threading.get_ident()
        self._req_timing = {}
        self.last_detail = None
        self.last_diag = None
        self.last_diag_s = 0.0
        self._cand_log = None
        self._query_tensor_log = None

        # "embedding time": query encoding (LM forward + pooling) inside MaxSimSplitter
        with self._time_block("splitter.encode_text"):
            query_enc = self.splitter.encode_text(prompt)
        query_knn_emb = query_enc["pooled_knn"]  # torch tensor on device
        # HNSW runs on CPU; pass a CPU list[float]
        query_knn_emb_cpu = query_knn_emb.detach().float().cpu().tolist()

        # NOTE: don't wrap this call in its own timing block because it already contains
        # fine-grained timers (wrapping would double-count in totals).
        cache_size_before = int(self.cache.vector_db_size()) if self.detail_log else None
        nn_metadata, similarity_score = self._select_nn_by_maxsim_with_query(
            prompt, query_enc, query_knn_emb_cpu
        )
        if self.diag_next:
            # [로그] 샘플 진단(C). 이번 요청이 캐시를 바꾸기 전 상태에서 계산한다.
            #        걸린 시간은 last_diag_s로 따로 남겨 요청 시간에서 뺀다 (단계별 시간에도 넣지 않음)
            self.diag_next = False
            diag_t0 = time.perf_counter()
            self._in_diag = True
            try:
                self.last_diag = self._diagnose(query_enc, query_knn_emb_cpu)
            finally:
                self._in_diag = False
                self.last_diag_s = time.perf_counter() - diag_t0
        if nn_metadata is None or similarity_score is None:
            # [추가] 비교할 캐시 항목이 없는 요청(캐시가 비어 있음 등)은 기록할 s가 없으므로 None
            self.last_decision = None
            self.bayesian.last_info = None
            with self._time_block("llm.call"):
                response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
            self.__cache_add(prompt=prompt, response=response, id_set=id_set)
            self._finish_detail(query_enc, cache_size_before, None, None)
            return False, response, EmbeddingMetadataObj(embedding_id=-1, response="")

        # [추가] 판정 직전에 이 캐시 항목이 가진 (s, c) 관측 개수를 저장 (사전값 2개 포함)
        n_obs_before = len(nn_metadata.observations)
        with self._time_block("vcache.decide"):
            action = self.bayesian.select_action(similarity_score=similarity_score, metadata=nn_metadata)
        # Snapshot of what the verified policy saw for this request (read by the eval script
        # when --log-scores is set). t_hat/gamma are the fit used for this decision; they stay
        # None while the entry has too few observations and the policy always explores.
        # [추가] 이번 판정의 기록. 각 항목 의미:
        #   s            : vCache에 넘어간 유사도 점수 (가장 비슷한 후보의 점수)
        #   embedding_id : 그 후보 캐시 항목의 번호
        #   action       : "exploit"(재사용) 또는 "explore"(LLM 새로 호출)
        #   n_obs        : 판정 시점의 관측 개수
        #   t_hat, gamma : 이 항목의 로지스틱 모델 임계값과 기울기 (관측이 부족하면 None)
        self.last_decision = {
            "s": float(similarity_score),
            "embedding_id": int(nn_metadata.embedding_id),
            "action": action.value,
            "n_obs": n_obs_before,
            "t_hat": None if nn_metadata.t_hat is None else float(nn_metadata.t_hat),
            "gamma": None if nn_metadata.gamma is None else float(nn_metadata.gamma),
        }

        self._finish_detail(query_enc, cache_size_before, nn_metadata, action)

        match action:
            case _Action.EXPLOIT:
                return True, nn_metadata.response, nn_metadata
            case _Action.EXPLORE:
                with self._time_block("llm.call"):
                    response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
                if self.last_detail is not None:
                    self.last_detail["timing_s"]["llm.call"] = self._req_timing.get("llm.call", 0.0)
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
        # [로그] 후보별 (메타, 최종 점수 s, 후보 조각 텐서)
        cand_log: list = [] if self.detail_log else None
        self._query_tensor_log = query_tensor

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
                            # [수정] 가중치 적용을 위해 _score_tensors 경유.
                            # 주의: multivector_top_k 실행은 이 경로를 타며, 여기서는 --mix-fullcos가 적용되지 않는다
                            #       (순수 가중 MaxSim). 원본 코드도 동일한 동작.
                            s = self._score_tensors(query_tensor, cand_tensor)
                else:
                    with self._time_block("maxsim.score_pair"):
                        s = self._maxsim_similarity_from_encoded(query_enc, cached_prompt)
            except Exception as e:
                self.logger.warning(f"MaxSim similarity failed for one candidate: {e}")
                continue
            if cand_log is not None:
                cand_log.append((meta, float(s), cand_tensor if query_tensor is not None else None))
            if s > best_s:
                best_s = s
                best_meta = meta

        self._cand_log = cand_log
        if best_meta is None:
            return None, None
        return best_meta, best_s

    # ------------------------------------------------------------------ logging helpers
    # [로그] 아래 함수들은 detail_log=True일 때만 쓰인다. 점수·판정 결과에는 영향을 주지 않는다.

    def _sentence_vectors(self, ids: list) -> np.ndarray:
        """Unit vectors the sentence-level HNSW index stores for these embedding ids."""
        vdb = self.cache.embedding_store.vector_db
        if not ids or getattr(vdb, "index", None) is None:
            return np.zeros((0, 0), dtype=np.float32)
        x = np.asarray(vdb.index.get_items(list(ids)), dtype=np.float32)
        return _l2_normalize_rows(x)

    @staticmethod
    def _unit(v) -> np.ndarray:
        x = np.asarray(v, dtype=np.float32).reshape(1, -1)
        return _l2_normalize_rows(x)[0]

    def _segment_match(self, query_tensor, cand_tensor) -> dict:
        """Per-row best match in both directions (log B6). Last row of each side = whole sentence."""
        import torch.nn.functional as F

        q = F.normalize(query_tensor.detach().float().cpu(), dim=-1)
        c = F.normalize(cand_tensor.detach().float().cpu(), dim=-1)
        cos = q @ c.T
        rmax, rarg = cos.max(dim=1)  # 쿼리 조각마다 후보에서 가장 비슷한 조각
        cmax, carg = cos.max(dim=0)  # 후보 조각마다 쿼리에서 가장 비슷한 조각
        return {
            "query_row_max": [round(float(v), 4) for v in rmax],
            "query_row_match": [int(v) for v in rarg],
            "cand_row_max": [round(float(v), 4) for v in cmax],
            "cand_row_match": [int(v) for v in carg],
        }

    def _finish_detail(self, query_enc: dict, cache_size_before, nn_metadata, action) -> None:
        """Assemble last_detail for this request (log B2-B4, B6, B7, B9)."""
        if not self.detail_log:
            return
        cand_log = self._cand_log or []
        query_tensor = self._query_tensor_log
        qvec = self._unit(query_enc["pooled_knn"].detach().float().cpu().numpy())
        vecs = self._sentence_vectors([int(m.embedding_id) for m, _, _ in cand_log])
        weights_fn = getattr(self.splitter, "segment_weights", None)
        candidates = []
        for i, (meta, s, ct) in enumerate(cand_log):
            candidates.append({
                "id": int(meta.embedding_id),
                "prompt": getattr(meta, "prompt", ""),
                "response": getattr(meta, "response", ""),
                "id_set": int(getattr(meta, "id_set", -1)),
                # 코사인: HNSW가 쓰는 문장 벡터 코사인 (재순위 전 기준)
                "cos": round(float(vecs[i] @ qvec), 5) if len(vecs) else None,
                # 균등 SMaxSim: 같은 조각, 가중치 없이 / 가중 SMaxSim: 실제로 쓰인 점수
                "s_uniform": round(float(self._maxsim_from_tensors(query_tensor, ct)), 5) if ct is not None else None,
                "s_weighted": round(s, 5),
            })
        detail = {
            "cache_size": cache_size_before,
            "n_query_tokens": int(query_enc["length"]),
            "candidates": candidates,
            "selected_id": None if nn_metadata is None else int(nn_metadata.embedding_id),
            "action": None if action is None else action.value,
            "decision": self.bayesian.last_info,
            "split": self.splitter.describe_split(query_enc) if hasattr(self.splitter, "describe_split") else None,
            "query_weights": None,
            "segment_match": None,
            "timing_s": dict(self._req_timing),
        }
        if query_tensor is not None and weights_fn is not None:
            detail["query_weights"] = [round(float(w), 5) for w in weights_fn(query_tensor)]
        if nn_metadata is not None and query_tensor is not None:
            best = next((ct for m, _, ct in cand_log if m.embedding_id == nn_metadata.embedding_id), None)
            if best is not None:
                detail["segment_match"] = self._segment_match(query_tensor, best)
        self.last_detail = detail

    def _diagnose(self, query_enc: dict, query_knn_emb_cpu: list) -> Optional[dict]:
        """Sample diagnostics (log C): HNSW recall, brute-force MVR best, candidate sim matrices."""
        import torch.nn.functional as F

        n = int(self.cache.vector_db_size())
        if n == 0:
            return None
        k = max(1, int(self.candidate_k))
        # 1) 정확 검색 Top-K vs HNSW Top-K (같은 문장 벡터 코사인 기준) → HNSW recall
        have = self._diag_vecs.shape[0]
        if have < n:
            new = self._sentence_vectors(list(range(have, n)))
            self._diag_vecs = new if have == 0 else np.vstack([self._diag_vecs, new])
        qvec = self._unit(query_knn_emb_cpu)
        exact = [int(i) for i in np.argsort(-(self._diag_vecs[:n] @ qvec))[:k]]
        hnsw = [int(i) for _, i in self.cache.get_knn_from_embedding(embedding=query_knn_emb_cpu, k=k)]
        cand_ids = [int(m.embedding_id) for m, _, _ in (self._cand_log or [])]

        # 2) 캐시 전체를 가중 SMaxSim으로 전수 비교했을 때의 1등 (Top-K 후보 방식이 타당한지 검증)
        query_tensor = self.splitter.split_text_return_maxsim_tensor_from_encoded(query_enc)
        best_meta, best_s = None, -1.0
        for meta in self.cache.get_all_embedding_metadata_objects():
            ct = getattr(meta, "cached_maxsim_tensor", None)
            if ct is None:
                ct = self.splitter.split_text_return_maxsim_tensor(meta.prompt)
                meta.cached_maxsim_tensor = ct.detach()
            s = float(self._score_tensors(query_tensor, ct))
            if s > best_s:
                best_meta, best_s = meta, s

        # 3) 후보 전체의 조각×조각 코사인 행렬 (행 = 쿼리 조각, 열 = 후보 조각, 마지막 = 문장 전체)
        q = F.normalize(query_tensor.detach().float().cpu(), dim=-1)
        matrices = {}
        for meta, _, ct in self._cand_log or []:
            if ct is not None:
                c = F.normalize(ct.detach().float().cpu(), dim=-1)
                matrices[str(int(meta.embedding_id))] = [[round(float(v), 3) for v in row] for row in (q @ c.T)]
        return {
            "k": k,
            "cache_size": n,
            "exact_topk": exact,
            "hnsw_topk": hnsw,
            "hnsw_recall": len(set(exact) & set(hnsw)) / max(1, len(exact)),
            "candidate_ids": cand_ids,
            "candidates_in_exact_topk": len(set(exact) & set(cand_ids)) / max(1, len(exact)),
            "bruteforce_best": None if best_meta is None else {
                "id": int(best_meta.embedding_id),
                "s": round(best_s, 5),
                "prompt": best_meta.prompt,
                "response": best_meta.response,
                "id_set": int(getattr(best_meta, "id_set", -1)),
            },
            "sim_matrices": matrices,
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
            self.request_order,
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
        request_order: Optional[int] = None,
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
                request_order,
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
            request_order,
        ) = update_args

        if self.cache is None:
            return

        try:
            latest_metadata_object = self.cache.get_metadata(embedding_id=embedding_id)
        except (ValueError, KeyError):
            return

        if latest_metadata_object is None:
            return

        n_obs_before = len(latest_metadata_object.observations)
        try:
            self.bayesian.add_observation_to_metadata(
                similarity_score=similarity_score,
                is_correct=should_have_exploited,
                metadata=latest_metadata_object,
            )
        except (ValueError, KeyError):
            return

        inserted_id = None
        if not should_have_exploited:
            inserted_id = self.__cache_add(prompt=prompt, response=new_response, id_set=id_set)

        # [로그] 관측 추가 기록 (updates.jsonl). 판정 규칙·캐시 상태에는 영향 없음
        if self.update_hook is not None:
            try:
                self.update_hook({
                    "order": request_order,  # 이 관측을 만든 요청 번호 (requests.jsonl의 order)
                    "applied_at_order": self.request_order,  # 실제로 반영된 시점에 처리 중이던 요청 번호
                    "entry_id": int(embedding_id),
                    "s": round(float(similarity_score), 3),  # 저장되는 값과 같은 반올림
                    "c": int(bool(should_have_exploited)),
                    "n_obs_before": n_obs_before,  # 사전값 2개 포함
                    "n_obs_after": len(latest_metadata_object.observations),
                    "inserted_entry_id": None if inserted_id is None else int(inserted_id),
                })
            except Exception as e:
                self.logger.warning(f"update log failed: {e}")

        try:
            self.cache.update_metadata(
                embedding_id=embedding_id, embedding_metadata=latest_metadata_object
            )
        except (ValueError, KeyError):
            return


