"""Experiment logs for eval_sembenchmark_verified_splitter.py (--run-log-dir).

Files written to the run directory:

  run.json         A. one record per run: ids, code/library/hardware, parameters, data
  requests.jsonl   B. one record per test prompt: input, split, candidates and their
                      scores, rank diagnostics, selected neighbour, vCache decision,
                      outcome, timings (timing_ms.total excludes the C diagnostics,
                      which are in timing_ms.diag)
  diag.jsonl       C. sampled prompts only: HNSW recall, brute-force MVR best, candidate
                      segment x segment cosine matrices
  progress.jsonl   D. every --progress-every prompts: running hit / error rate (overall
                      and per task), stored vectors, memory
  updates.jsonl    one record per observation (s, c) added to a cache entry by the
                      background update: originating request, request being processed
                      when it landed, entry, observation counts, newly inserted entry

Definitions
  candidates  what the cache retrieved for the prompt (--candidate-selection / --candidate-k)
  c           1 when a candidate's response is correct for the prompt (same rule as tp/fn)
  cosine #1   best candidate by sentence-vector cosine (before reranking)
  uniform #1  best by unweighted SMaxSim;  weighted #1  best by the score actually used
"""
# [새 파일] 실험 로그 (A~D) 기록 도구. 정책(verified_splitter.py)이 모은 값으로 c·순위 진단을 계산해 파일로 쓴다.
# 사후 집계(F)는 analyze_run_log.py.

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import defaultdict
from typing import Optional


def _sha256(path: Optional[str]) -> Optional[str]:
    # 가중치 모델 파일 해시 (어떤 파일로 돌렸는지 확인용)
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def split_fingerprint(df) -> Optional[str]:
    """Same fingerprint as the split manifests: sha1 of "dataset_name:id" joined by commas."""
    if "dataset_name" not in df.columns or "id" not in df.columns:
        return None
    key = ",".join(f"{a}:{b}" for a, b in zip(df["dataset_name"], df["id"]))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _git(args: list) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() or None
    except Exception:
        return None


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "numpy", "pandas", "sklearn", "hnswlib", "scipy"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def _hardware() -> dict:
    hw = {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()}
    hw["colab"] = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))
    try:
        import torch

        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        else:
            hw["gpu"] = None
    except Exception:
        hw["gpu"] = None
    return hw


def _memory() -> dict:
    mem = {}
    try:
        import psutil

        mem["ram_rss_gb"] = round(psutil.Process().memory_info().rss / 2**30, 3)
    except Exception:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS는 바이트, Linux는 KB 단위
        mem["ram_peak_gb"] = round(rss / (2**30 if sys.platform == "darwin" else 2**20), 3)
    try:
        import torch

        if torch.cuda.is_available():
            mem["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
            mem["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception:
        pass
    return mem


def _rank_of_first_correct(order: list, c: list) -> Optional[int]:
    for rank, i in enumerate(order, start=1):
        if c[i]:
            return rank
    return None


def _change(c_from: int, c_to: int, changed: bool) -> str:
    # 1등이 바뀌었을 때의 유형: 오답→정답(fixed, 교정) / 정답→오답(broke, 악화) / 오답→오답 / 정답→정답
    if not changed:
        return "unchanged"
    return {(0, 1): "fixed", (1, 0): "broke", (0, 0): "wrong_to_wrong", (1, 1): "right_to_right"}[(c_from, c_to)]


def rank_diagnostics(cands: list) -> dict:
    """B5: cosine / uniform / weighted #1, top-1 change flags and types, ceiling, case."""
    if not cands:
        return {"n_candidates": 0, "n_correct": 0, "ceiling": False, "case": "no_candidates"}
    c = [x["c"] for x in cands]

    def order(key):
        vals = [x[key] if x[key] is not None else -1e9 for x in cands]
        return sorted(range(len(cands)), key=lambda i: -vals[i])  # 동점이면 먼저 나온 후보

    o_cos, o_uni, o_w = order("cos"), order("s_uniform"), order("s_weighted")
    top = {"cos": o_cos[0], "uniform": o_uni[0], "weighted": o_w[0]}
    n_correct = int(sum(c))
    flag1 = cands[top["cos"]]["id"] != cands[top["uniform"]]["id"]  # ① 재순위 효과
    flag2 = cands[top["uniform"]]["id"] != cands[top["weighted"]]["id"]  # ② 가중치 효과
    if n_correct == 0:
        case = "1_no_correct_candidate"  # ① 후보에 정답 없음 (검색 천장 밖)
    elif c[top["weighted"]]:
        case = "2_top1_correct"  # ② 정답 있고 1등도 정답
    else:
        case = "3_top1_wrong"  # ③ 정답 있는데 1등은 오답 (선택 실패)
    return {
        "n_candidates": len(cands),
        "top1": {k: {"id": cands[i]["id"], "c": c[i]} for k, i in top.items()},
        "flag_rerank_changed": flag1,
        "flag_weight_changed": flag2,
        "rerank_change": _change(c[top["cos"]], c[top["uniform"]], flag1),
        "weight_change": _change(c[top["uniform"]], c[top["weighted"]], flag2),
        "n_correct": n_correct,
        "first_correct_rank": {
            "cos": _rank_of_first_correct(o_cos, c),
            "uniform": _rank_of_first_correct(o_uni, c),
            "weighted": _rank_of_first_correct(o_w, c),
        },
        "ceiling": n_correct > 0,
        "case": case,
    }


class RunLogger:
    """Writes run.json / requests.jsonl / diag.jsonl / progress.jsonl for one evaluation run."""

    def __init__(self, out_dir: str, *, delta: float, progress_every: int = 1000):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self.delta = float(delta)
        self.progress_every = int(progress_every)
        self.header: dict = {}
        self.req_f = open(os.path.join(out_dir, "requests.jsonl"), "w", encoding="utf-8")
        self.diag_f = open(os.path.join(out_dir, "diag.jsonl"), "w", encoding="utf-8")
        self.prog_f = open(os.path.join(out_dir, "progress.jsonl"), "w", encoding="utf-8")
        # updates.jsonl은 백그라운드 스레드에서 쓰므로 잠금을 둔다
        self.upd_f = open(os.path.join(out_dir, "updates.jsonl"), "w", encoding="utf-8")
        self._upd_lock = threading.Lock()
        self.n = 0
        self.hits = 0
        self.false_hits = 0
        self.by_task = defaultdict(lambda: [0, 0, 0])  # 과제별 [요청 수, 히트, 오답 히트]

    # ---------------------------------------------------------------- A. 실행 단위
    def write_header(self, *, args, df, train_prompts: Optional[set], extra: dict) -> None:
        tasks = df["dataset_name"].value_counts().to_dict() if "dataset_name" in df.columns else None
        dup = None
        if train_prompts is not None:
            dup = int(df["prompt"].astype(str).isin(train_prompts).sum())
        # 데이터 폴더에 manifest.json이 있으면 함께 기록 (분할 인덱스 지문, leak 수 등)
        manifest = None
        mpath = os.path.join(os.path.dirname(os.path.abspath(str(args.dataset))), "manifest.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            manifest = {k: v for k, v in m.items() if k != "leak_positions"}
        punct = getattr(sys.modules.get("vcache.vcache_core.splitter.RuleSplitter"), "PUNCT_CHARS", None)
        self.header = {
            "run_id": args.run_id,
            "condition": args.condition,
            "seed": args.seed,
            "repeat": args.repeat,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
            "versions": _versions(),
            "hardware": _hardware(),
            "delta": self.delta,
            "candidate_selection": args.candidate_selection,
            "candidate_k": args.candidate_k,
            "splitter": {
                "mode": args.splitter_mode,
                "punct_chars": punct if args.splitter_mode == "rule" else None,
                "rules": "1-6 on" if args.splitter_mode == "rule" else None,
                "max_splits": args.splitter_max_segments,
                "min_tokens": None,
                "include_full_embedding": bool(args.include_full_embedding),
                "overlap_tokens": int(args.splitter_overlap_tokens),
                "checkpoint": args.splitter_checkpoint,
            },
            "weighting": args.segment_weighting,
            "weight_stats": args.segment_weight_stats,
            "weight_stats_sha256": _sha256(args.segment_weight_stats),
            "dataset": args.dataset,
            "dataset_fingerprint": split_fingerprint(df),
            "dataset_manifest": manifest,
            "train_file": args.train_file,
            "n_prompts": int(len(df)),
            "task_counts": tasks,
            "train_test_duplicate_prompts": dup,
            "sleep": args.sleep,
            "cache_on_miss": True,
            "diag_frac": args.diag_frac,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **extra,
        }
        self._dump_header()

    def _dump_header(self) -> None:
        with open(os.path.join(self.dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(self.header, f, ensure_ascii=False, indent=1, default=str)

    def finish(self, extra: dict) -> None:
        self.header["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.header.update(extra)
        self._dump_header()
        with self._upd_lock:
            for f in (self.req_f, self.diag_f, self.prog_f, self.upd_f):
                f.close()

    # ---------------------------------------------------------------- B. 프롬프트 단위
    def log_request(self, *, order: int, row: dict, prompt: str, in_train: Optional[bool],
                    detail: Optional[dict], correct_fn, prompt_info: dict,
                    is_hit: bool, false_hit: bool, total_s: float, diag_s: float = 0.0) -> None:
        # total_s: 요청 처리 시간(진단 제외), diag_s: 이번 요청에서 샘플 진단(C)에 쓴 시간
        task = row.get("dataset_name")
        raw = (detail or {}).get("candidates", [])
        cands = [
            {k: x[k] for k in ("id", "cos", "s_uniform", "s_weighted")}
            | {"c": int(correct_fn(x["response"], x["id_set"]))}
            for x in raw
        ]
        sel = None if detail is None else detail.get("selected_id")
        sel_raw = next((x for x in raw if x["id"] == sel), None)
        sel_c = None if sel_raw is None else int(correct_fn(sel_raw["response"], sel_raw["id_set"]))
        sel_info = prompt_info.get((sel_raw or {}).get("prompt"), {})
        action = None if detail is None else detail.get("action")
        timing = {k: round(v * 1000, 3) for k, v in ((detail or {}).get("timing_s") or {}).items()}
        timing["total"] = round(total_s * 1000, 3)
        if diag_s:
            timing["diag"] = round(diag_s * 1000, 3)
        rec = {
            # B1 식별·입력
            "order": order,
            "prompt_id": row.get("id"),
            "task": task,
            "n_chars": len(prompt),
            "n_tokens": None if detail is None else detail.get("n_query_tokens"),
            "in_train": in_train,
            # B2 분할 (조각 텍스트, 조각별 토큰 수, 1토큰 조각 수, 규칙별 발동 횟수)
            "split": None if detail is None else detail.get("split"),
            # B3 검색
            "cache_size": None if detail is None else detail.get("cache_size"),
            "candidate_ids": [x["id"] for x in cands],
            # B4 후보별 점수
            "candidates": cands,
            # B5 순위 진단
            "rank": rank_diagnostics(cands),
            # B6 선택된 이웃
            "selected": None if sel_raw is None else {
                "id": sel,
                "task": sel_info.get("task"),
                "prompt_id": sel_info.get("id"),
                "s": sel_raw["s_weighted"],
                "c": sel_c,
                "query_weights": detail.get("query_weights"),
                "segment_match": detail.get("segment_match"),
            },
            # B7 vCache 판정 (판정 전후 t̂·γ, α, τ, u, 이유)
            "decision": None if detail is None else detail.get("decision"),
            "action": action,
            # B8 결과
            "hit": bool(is_hit),
            "false_hit": bool(false_hit),
            "llm_called": not bool(is_hit),
            # 캐시 삽입: 이웃이 없거나, 탐색했는데 이웃 답이 틀린 경우 (cache-on-miss)
            "cache_insert": sel is None or (action == "explore" and sel_c == 0),
            # B9 시간 (ms)
            "timing_ms": timing,
        }
        self.req_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        self.n += 1
        self.hits += int(is_hit)
        self.false_hits += int(false_hit)
        t = self.by_task[task]
        t[0] += 1
        t[1] += int(is_hit)
        t[2] += int(false_hit)

    # ---------------------------------------------------------------- C. 샘플 진단
    def log_diag(self, *, order: int, task, diag: Optional[dict], correct_fn) -> None:
        if diag is None:
            return
        best = diag.get("bruteforce_best")
        if best is not None:
            best = {k: best[k] for k in ("id", "s")} | {"c": int(correct_fn(best["response"], best["id_set"]))}
        rec = {k: v for k, v in diag.items() if k != "bruteforce_best"} | {
            "order": order, "task": task, "bruteforce_best": best,
        }
        self.diag_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- 관측 추가 기록
    def log_update(self, rec: dict) -> None:
        """Called from the policy's background update thread (one line per observation)."""
        with self._upd_lock:
            if not self.upd_f.closed:
                self.upd_f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- D. 주기 기록
    def maybe_progress(self, *, cache_vectors: int, multivector_vectors: Optional[int]) -> None:
        if self.progress_every <= 0 or self.n % self.progress_every != 0:
            return
        err = self.false_hits / self.n
        rec = {
            "n": self.n,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hit_rate": round(self.hits / self.n, 5),
            "error_rate": round(err, 5),
            "error_within_delta": err <= self.delta,
            "by_task": {
                str(k): {
                    "n": v[0],
                    "hit_rate": round(v[1] / v[0], 5),
                    "error_rate": round(v[2] / v[0], 5),
                    "error_within_delta": v[2] / v[0] <= self.delta,
                }
                for k, v in self.by_task.items()
            },
            "cache_vectors": cache_vectors,
            "multivector_vectors": multivector_vectors,
            **_memory(),
        }
        self.prog_f.write(json.dumps(rec) + "\n")
        # 중간 저장: 파일 버퍼를 디스크에 내려 둔다
        for f in (self.req_f, self.diag_f, self.prog_f):
            f.flush()
        with self._upd_lock:
            self.upd_f.flush()
