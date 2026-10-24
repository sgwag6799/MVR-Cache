"""Rebuild the team's Classification splits (cls_seed42) from the HF parquet.

Same rule as the team files: df.sample(frac=1, random_state=42), then the first 3,000
rows are train, the next 1,000 val, the remaining 41,000 test, each in shuffled order.
The fingerprints (sha1 of "dataset_name:id" in row order, from the gpu-production
split manifests) are checked so a wrong HF revision or rule fails loudly.

Usage:
  python scripts/make_cls_splits.py --raw data/raw/train.parquet --out data
  -> data/cls_seed42/{train,val}.parquet, data/cls_seed42_nodedup/test41k.parquet
"""

import argparse
import hashlib
import json
import os

import pyarrow.parquet as pq

SEED = 42
SPLITS = [  # (name, rows, out path relative to --out, team fingerprint)
    ("train", 3000, "cls_seed42/train.parquet", "35365cc8e27f"),
    ("val", 1000, "cls_seed42/val.parquet", "a472edba40b2"),
    ("test", 41000, "cls_seed42_nodedup/test41k.parquet", "f215310197be"),
]


def fingerprint(df) -> str:
    key = ",".join(f"{a}:{b}" for a, b in zip(df["dataset_name"], df["id"]))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/train.parquet")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    names = pq.ParquetFile(args.raw).schema_arrow.names
    columns = [c for c in names if not c.startswith("emb_")]  # embeddings are 3GB+, unused
    df = pq.read_table(args.raw, columns=columns).to_pandas()
    shuffled = df.sample(frac=1, random_state=SEED)

    pos, meta = 0, {"seed": SEED, "rule": "df.sample(frac=1, random_state=42)", "splits": {}}
    for name, n, rel, expected in SPLITS:
        part = shuffled.iloc[pos:pos + n].reset_index(drop=True)
        pos += n
        fp = fingerprint(part)
        if fp != expected:
            raise SystemExit(f"{name}: fingerprint {fp} != team {expected}")
        path = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part.to_parquet(path, index=False)
        meta["splits"][name] = {"rows": len(part), "path": rel, "fingerprint": fp,
                                "tasks": part.dataset_name.value_counts().to_dict()}
        print(f"{name}: {len(part)} rows, fingerprint {fp} ok -> {path}")

    with open(os.path.join(args.out, "cls_seed42", "splits_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
