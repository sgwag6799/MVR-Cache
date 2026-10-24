"""Stage 2 report: hit / error rate per condition, seed and task from the --output-json files.

Each run is given as COND:SEED=path (path = --output-json of eval_sembenchmark_verified.py
for M1, eval_sembenchmark_verified_splitter.py for M2/M3). Rows are matched to the
evaluated dataset in order, so pass the same file and filter the runs used.

Official numbers skip the first --skip rows (default 2,000; cache warm-up).
Paired differences are taken between conditions that share a seed.

Example:
  python -m benchmarks.prefix.prefix_stream_report \
    --dataset /content/drive/MyDrive/mvr_eval/prefix/data/variant/L8/test.parquet \
    --filter-true-column fits_all_levels \
    M1:1=runs/L8_M1_s1.json M2:1=runs/L8_M2_s1.json M3:1=runs/L8_M3_s1.json ...
"""
# [새 파일] 2단계 집계: 조건 × 시드 × 과제별 히트율·오류율, 같은 시드끼리 짝지은 차이

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="COND:SEED=path/to/output.json")
    p.add_argument("--dataset", required=True)
    p.add_argument("--filter-true-column", default=None)
    p.add_argument("--skip", type=int, default=2000)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--out", default=None, help="CSV path")
    args = p.parse_args()

    df = pd.read_parquet(args.dataset)
    if args.filter_true_column:
        df = df[df[args.filter_true_column].astype(bool)].reset_index(drop=True)
    tasks = df["dataset_name"].to_numpy()

    rows, by = [], defaultdict(dict)
    for spec in args.runs:
        key, path = spec.split("=", 1)
        cond, seed = key.split(":")
        per = json.load(open(path))["per_sample"]
        if len(per) != len(df):
            raise SystemExit(f"{path}: {len(per)} samples but dataset has {len(df)} rows (same file/filter?)")
        hit = np.array([bool(r["is_hit"]) for r in per])
        fp = np.array([int(r["fp"]) for r in per])
        keep = np.arange(len(per)) >= args.skip
        for task in ["ALL"] + sorted(set(tasks)):
            m = keep & (True if task == "ALL" else tasks == task)
            rec = {"condition": cond, "seed": int(seed), "task": task, "n": int(m.sum()),
                   "hit_rate": float(hit[m].mean()), "error_rate": float(fp[m].mean())}
            rec["error_within_delta"] = rec["error_rate"] <= args.delta
            rows.append(rec)
            by[(task, cond)][int(seed)] = rec

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    # 같은 시드끼리 짝지은 차이 (M3 − M2, M3 − M1, M2 − M1)
    print("\npaired differences (same seed), %p:")
    for task in ["ALL"] + sorted(set(tasks)):
        for a, b in (("M3", "M2"), ("M3", "M1"), ("M2", "M1")):
            sa, sb = by.get((task, a), {}), by.get((task, b), {})
            seeds = sorted(set(sa) & set(sb))
            if not seeds:
                continue
            d = np.array([sa[s]["hit_rate"] - sb[s]["hit_rate"] for s in seeds]) * 100
            e = np.array([sa[s]["error_rate"] - sb[s]["error_rate"] for s in seeds]) * 100
            print(f"  {task:22s} {a}-{b}: hit {d.mean():+.2f} ± {d.std(ddof=1) if len(d) > 1 else 0:.2f} "
                  f"(seeds {', '.join(f'{x:+.2f}' for x in d)}) | error {e.mean():+.3f}")
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
