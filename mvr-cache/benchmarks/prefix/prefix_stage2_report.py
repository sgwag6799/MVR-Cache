"""Experiment 8: summarise the stage 2 stream runs (hit rate, error rate) from the per-sample logs.

Each run JSON (eval_sembenchmark_verified*.py --output-json) holds one tp / fp / tn / fn indicator (0/1)
per request. The official count skips the first --warmup requests (the cache is still filling):

  hit rate    = hits after warm-up / requests after warm-up        (hits = tp + fp)
  error rate  = fp after warm-up / requests after warm-up          (wrong answers served; target <= delta)
  precision   = tp / hits after warm-up

Per sub-task (--task-col, default dataset_name): each run JSON stores its --dataset path and
--filter-true-column, and the eval scripts process the filtered rows in order without skipping, so
request i is row i of the filtered dataset. The sub-task counts use the same warm-up cut as the totals
(the first --warmup requests of the whole stream). Runs whose dataset file is not found get no sub-task rows.

Run names follow scripts/run_stage2_lmarena.sh: <condition>_L<level>_s<seed>. Also prints the paired
(same seed) differences LoRA - base for B-A, D-C, G-F, and the hit-rate recovery
(H_L8 - H_L8,base) / (H_L0,base - H_L8,base).

Example:
  python -m benchmarks.prefix.prefix_stage2_report --runs ../data/exp8/runs --out ../data/exp8/report
"""
# [새 파일] 실험 8: 2단계 스트림 결과 집계 (히트율 · 오류율, 앞 2,000행 제외)

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

PAIRS = {"vCache": ("A", "B"), "MVR 규칙 균등": ("C", "D"), "MVR-Cache (논문, RL 분할기)": ("F", "G")}


def _rates(recs):
    d = {k: sum(r[k] for r in recs) for k in ("tp", "fp", "tn", "fn")}
    n = sum(d.values())
    hits = d["tp"] + d["fp"]
    return {"n": n, "hits": hits, "hit_rate": hits / n if n else float("nan"),
            "error_rate": d["fp"] / n if n else float("nan"), "precision": d["tp"] / hits if hits else float("nan")}


_TASKS: dict = {}


def task_labels(run_args, task_col):
    """Sub-task of every request: the run's dataset filtered as the eval script did (None if not found)."""
    path, col = run_args.get("dataset"), run_args.get("filter_true_column")
    key = (path, col, task_col)
    if key not in _TASKS:
        _TASKS[key] = None
        if path and os.path.exists(path):
            df = pd.read_parquet(path, columns=[c for c in (task_col, col) if c])
            if col:
                df = df[df[col].astype(bool)].reset_index(drop=True)
            _TASKS[key] = df[task_col].astype(str).to_numpy()
    return _TASKS[key]


def summarise(path, warmup, task_col):
    j = json.load(open(path))
    ps = j["per_sample"]
    out = {"n_total": len(ps), **_rates(ps[warmup:]), "delta": j["summary"].get("delta"),
           "total_time_s": j["summary"].get("total_time")}
    by_task = []
    tasks = task_labels(j.get("args", {}), task_col)
    if tasks is not None and len(tasks) >= len(ps):
        for t in sorted(set(tasks[warmup:len(ps)])):
            recs = [r for r, tt in zip(ps[warmup:], tasks[warmup:len(ps)]) if tt == t]
            by_task.append({"task": t, **_rates(recs)})
    return out, by_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--task-col", default="dataset_name", help="Sub-task column of the dataset (per-task tables)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, task_rows = [], []
    for f in sorted(glob.glob(os.path.join(args.runs, "*.json"))):
        m = re.match(r"([A-Z]+)_L(\d+)_s(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        key = {"cond": m[1], "level": int(m[2]), "seed": int(m[3])}
        total, by_task = summarise(f, args.warmup, args.task_col)
        rows.append({**key, **total})
        task_rows += [{**key, **t} for t in by_task]
    df = pd.DataFrame(rows).sort_values(["level", "cond", "seed"])
    df.to_csv(os.path.join(args.out, "stage2_runs.csv"), index=False)
    print(df[["cond", "level", "seed", "n", "hit_rate", "error_rate", "precision"]].to_string(index=False))

    agg = df.groupby(["cond", "level"]).agg(hit_rate=("hit_rate", "mean"), hit_sd=("hit_rate", "std"),
                                            error_rate=("error_rate", "mean"), err_max=("error_rate", "max"),
                                            seeds=("seed", "count")).reset_index()
    agg.to_csv(os.path.join(args.out, "stage2_summary.csv"), index=False)
    print("\n" + agg.to_string(index=False))

    if task_rows:
        tdf = pd.DataFrame(task_rows).sort_values(["task", "level", "cond", "seed"])
        tdf.to_csv(os.path.join(args.out, "stage2_runs_by_task.csv"), index=False)
        tagg = tdf.groupby(["task", "cond", "level"]).agg(
            n=("n", "mean"), hit_rate=("hit_rate", "mean"), hit_sd=("hit_rate", "std"),
            error_rate=("error_rate", "mean"), err_max=("error_rate", "max"), seeds=("seed", "count")).reset_index()
        tagg.to_csv(os.path.join(args.out, "stage2_summary_by_task.csv"), index=False)
        print("\n[하위 과제별]\n" + tagg.to_string(index=False))

    out = {"warmup": args.warmup, "pairs": {}}
    for name, (base, lora) in PAIRS.items():
        res = {}
        for level in sorted(df["level"].unique()):
            a = df[(df.cond == base) & (df.level == level)].set_index("seed")
            b = df[(df.cond == lora) & (df.level == level)].set_index("seed")
            seeds = sorted(set(a.index) & set(b.index))
            if not seeds:
                continue
            diff = [float(b.loc[s, "hit_rate"] - a.loc[s, "hit_rate"]) for s in seeds]
            res[f"L{level}"] = {"seeds": seeds, "hit_base": float(a.loc[seeds, "hit_rate"].mean()),
                                "hit_lora": float(b.loc[seeds, "hit_rate"].mean()), "hit_diff_per_seed": diff,
                                "err_base_max": float(a.loc[seeds, "error_rate"].max()),
                                "err_lora_max": float(b.loc[seeds, "error_rate"].max())}
        if "L0" in res and "L8" in res:
            h0, h8, h8l = res["L0"]["hit_base"], res["L8"]["hit_base"], res["L8"]["hit_lora"]
            res["recovery"] = (h8l - h8) / (h0 - h8) if h0 > h8 else float("nan")
        out["pairs"][name] = res
        print(f"\n[{name}] " + json.dumps(res, ensure_ascii=False))
    with open(os.path.join(args.out, "stage2_report.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
