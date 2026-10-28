"""Experiment 2 sensitivity (PREFIX_EXPERIMENT.md 13.3): merge ID_Sets that share an identical prompt.

SemBenchmarkLmArena has 186 prompts that appear word for word in more than one ID_Set. Such pairs
score 1.0 but count as wrong answers, which is label noise. This script joins those sets with a
union-find over the full raw file and writes the result as a new column `id_set_merged` into every
parquet under --data (split files and make_prefix_data output alike), keyed by the row id.

Usage:
  python scripts/lmarena_merge_duplicate_sets.py --raw data/raw_lmarena/train.parquet \
    --data data/lmarena/prefix/data
"""

import argparse
import glob
import json
import os

import pandas as pd
import pyarrow.parquet as pq


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_lmarena/train.parquet")
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    raw = pq.read_table(args.raw, columns=["id", "prompt", "ID_Set"]).to_pandas()
    raw["ID_Set"] = raw["ID_Set"].astype(str)
    parent = {s: s for s in raw["ID_Set"].unique()}
    for _, sets in raw.groupby("prompt")["ID_Set"]:
        sets = sets.unique()
        for s in sets[1:]:
            a, b = find(parent, sets[0]), find(parent, s)
            if a != b:
                parent[max(a, b, key=int)] = min(a, b, key=int)
    merged = {s: find(parent, s) for s in parent}
    by_id = dict(zip(raw["id"].astype(str), raw["ID_Set"].map(merged)))
    n_groups = len(set(merged.values()))
    print(f"ID_Sets {len(parent)} -> merged groups {n_groups}")

    files = sorted(glob.glob(os.path.join(args.data, "**", "*.parquet"), recursive=True))
    for f in files:
        df = pd.read_parquet(f)
        df["id_set_merged"] = df["id"].astype(str).map(by_id)
        assert df["id_set_merged"].notna().all(), f"{f}: ids missing from raw"
        df.to_parquet(f, index=False)
    with open(os.path.join(args.data, "id_set_merged_meta.json"), "w") as fh:
        json.dump({"raw_sets": len(parent), "merged_groups": n_groups, "files": len(files)}, fh, indent=1)
    print(f"wrote id_set_merged into {len(files)} files")


if __name__ == "__main__":
    main()
