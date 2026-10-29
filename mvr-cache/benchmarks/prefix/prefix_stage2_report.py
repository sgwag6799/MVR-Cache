"""Experiment 8: summarise the stage 2 stream runs (hit rate, error rate) from the per-sample logs.

Each run JSON (eval_sembenchmark_verified*.py --output-json) holds one tp / fp / tn / fn indicator (0/1)
per request. The official count skips the first --warmup requests (the cache is still filling):

  hit rate    = hits after warm-up / requests after warm-up        (hits = tp + fp)
  error rate  = fp after warm-up / requests after warm-up          (wrong answers served; target <= delta)
  precision   = tp / hits after warm-up

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


def summarise(path, warmup):
    j = json.load(open(path))
    ps = j["per_sample"]
    d = {k: sum(r[k] for r in ps[warmup:]) for k in ("tp", "fp", "tn", "fn")}
    n = sum(d.values())
    hits = d["tp"] + d["fp"]
    return {"n_total": len(ps), "n": n, "hits": hits, "hit_rate": hits / n, "error_rate": d["fp"] / n,
            "precision": d["tp"] / hits if hits else float("nan"), "delta": j["summary"].get("delta"),
            "total_time_s": j["summary"].get("total_time")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=2000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.runs, "*.json"))):
        m = re.match(r"([A-J])_L(\d+)_s(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        rows.append({"cond": m[1], "level": int(m[2]), "seed": int(m[3]), **summarise(f, args.warmup)})
    df = pd.DataFrame(rows).sort_values(["level", "cond", "seed"])
    df.to_csv(os.path.join(args.out, "stage2_runs.csv"), index=False)
    print(df[["cond", "level", "seed", "n", "hit_rate", "error_rate", "precision"]].to_string(index=False))

    agg = df.groupby(["cond", "level"]).agg(hit_rate=("hit_rate", "mean"), hit_sd=("hit_rate", "std"),
                                            error_rate=("error_rate", "mean"), err_max=("error_rate", "max"),
                                            seeds=("seed", "count")).reset_index()
    agg.to_csv(os.path.join(args.out, "stage2_summary.csv"), index=False)
    print("\n" + agg.to_string(index=False))

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
