"""Experiment 3 (judge on shared-prefix Classification): build the judge data files.

Input: Classification shared-paragraph data made by benchmarks.prefix.make_prefix_data
  shared pool : data/prefix/data/variant/L{0,8}/{train,val,test}.parquet   (prefix_pool.json)
  task pool   : data/prefix_task/data/variant/L{0,8}/{train,val,test}.parquet (prefix_pool_task.json, optional)

Output (git-ignored, under data/judge_cls/<pool>/):
  L{L}_trainval.parquet  judge training stream = train rows then val rows, fits_all_levels only
                         (2,791 + 935 = 3,726 rows). Used only to collect features and fix p* before test.
  L{L}_test.parquet      test rows of the level (41,000; eval scripts filter fits_all_levels -> 38,304)
Also writes mvr-cache/benchmarks/prefix/prefix_pool_task_flat.json: every task-pool sentence in one pool,
so the rule splitter can find where the task-pool prefix ends (cap_scope original).

Usage (repo root):  python scripts/judge/make_judge_cls_data.py [--pools shared task] [--levels 0 8]
"""
# [새 파일] 실험 3: 판정자 학습 스트림(train+val) · test 파일 만들기

import argparse
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = {"shared": os.path.join(REPO, "data", "prefix", "data", "variant"),
       "task": os.path.join(REPO, "data", "prefix_task", "data", "variant")}
OUT = os.path.join(REPO, "data", "judge_cls")
POOL_DIR = os.path.join(REPO, "mvr-cache", "benchmarks", "prefix")


def flat_task_pool() -> str:
    with open(os.path.join(POOL_DIR, "prefix_pool_task.json"), encoding="utf-8") as f:
        raw = json.load(f)
    n_slots = len(next(iter(raw["tasks"].values()))["slots"])
    slots = [{"name": f"slot{i}", "variants": [v for t in raw["tasks"].values() for v in t["slots"][i]["variants"]]}
             for i in range(n_slots)]
    path = os.path.join(POOL_DIR, "prefix_pool_task_flat.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"description": "All sentences of prefix_pool_task.json merged per slot (prefix-end detection only).",
                   "slots": slots}, f, ensure_ascii=False, indent=1)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["shared", "task"], choices=["shared", "task"])
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 8])
    args = ap.parse_args()
    if "task" in args.pools:
        print("flat task pool ->", flat_task_pool())
    for pool in args.pools:
        if not os.path.isdir(SRC[pool]):
            print(f"[skip] {pool}: {SRC[pool]} not found (run make_prefix_data first)")
            continue
        out = os.path.join(OUT, pool)
        os.makedirs(out, exist_ok=True)
        for L in args.levels:
            d = os.path.join(SRC[pool], f"L{L}")
            parts = [pd.read_parquet(os.path.join(d, f"{s}.parquet")) for s in ("train", "val")]
            tv = pd.concat([p[p.fits_all_levels] for p in parts], ignore_index=True)
            tv.to_parquet(os.path.join(out, f"L{L}_trainval.parquet"), index=False)
            te = pd.read_parquet(os.path.join(d, "test.parquet"))
            te.to_parquet(os.path.join(out, f"L{L}_test.parquet"), index=False)
            print(f"{pool} L{L}: trainval {len(tv)} rows, test {len(te)} rows (fits {int(te.fits_all_levels.sum())})")


if __name__ == "__main__":
    main()
