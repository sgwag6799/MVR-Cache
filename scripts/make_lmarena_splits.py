"""Experiment 2: train / val / test splits of SemBenchmarkLmArena, grouped by ID_Set.

LmArena has 63,796 prompts in 3,500 paraphrase sets (ID_Set). Two prompts count as "same answer"
when they share an ID_Set, so the label is decided by meaning, unlike the Classification tasks
where a few answer strings are shared by unrelated prompts. Splitting by set keeps every set in
one split, so a test anchor's paraphrases are in the test bank and none leak into M3 training.

Rule: shuffle the set ids with random.Random(42), then give whole sets to train until it has at
least 3,000 rows, to val until 1,000, to test until 41,000 (same sizes as cls_seed42); the
remaining sets are not used. Rows keep the HF order inside each split. Added columns:
  id_set_label   str(ID_Set), passed to the offline scripts as --resp-col
  dataset_name   "lm arena" (already present; one task, so pooled = within-task)

Usage:
  python scripts/make_lmarena_splits.py --raw data/raw_lmarena/train.parquet --out data/lmarena
  -> data/lmarena/{train,val,test}.parquet, data/lmarena/splits_meta.json
"""

import argparse
import hashlib
import json
import os
import random

import pyarrow.parquet as pq

SEED = 42
SPLITS = [("train", 3000), ("val", 1000), ("test", 41000)]


def fingerprint(df) -> str:
    key = ",".join(f"{a}:{b}" for a, b in zip(df["dataset_name"], df["id"]))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_lmarena/train.parquet")
    ap.add_argument("--out", default="data/lmarena")
    args = ap.parse_args()

    names = pq.ParquetFile(args.raw).schema_arrow.names
    columns = [c for c in names if not c.startswith("emb_")]  # embeddings are 5GB, unused
    df = pq.read_table(args.raw, columns=columns).to_pandas()
    df["id_set_label"] = df["ID_Set"].astype(str)
    sizes = df.groupby("id_set_label").size()
    sets = sorted(sizes.index, key=int)
    random.Random(SEED).shuffle(sets)

    os.makedirs(args.out, exist_ok=True)
    meta = {"seed": SEED, "rule": "whole ID_Sets in random.Random(42) order until each split reaches its size",
            "raw_rows": len(df), "raw_sets": len(sets), "splits": {}}
    pos = 0
    for name, n in SPLITS:
        chosen, rows = [], 0
        while rows < n and pos < len(sets):
            chosen.append(sets[pos])
            rows += int(sizes[sets[pos]])
            pos += 1
        part = df[df["id_set_label"].isin(set(chosen))].reset_index(drop=True)
        path = os.path.join(args.out, f"{name}.parquet")
        part.to_parquet(path, index=False)
        meta["splits"][name] = {"rows": len(part), "sets": len(chosen), "fingerprint": fingerprint(part),
                                "set_size_mean": round(float(sizes[chosen].mean()), 2)}
        print(f"{name}: {len(part)} rows, {len(chosen)} sets, fingerprint {fingerprint(part)} -> {path}")
    meta["unused_sets"] = len(sets) - pos

    with open(os.path.join(args.out, "splits_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
