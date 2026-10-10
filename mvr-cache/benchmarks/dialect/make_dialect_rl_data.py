"""Training data for the MVR-cache RL splitter (rl-training-algorithm/RL4COTrainer.py) from the dialect data.

Each sampled meaning (standard-sentence group) contributes its standard sentence and every distinct
dialect form of it, all with the same `id_set`. With `--label_mode id_set --train_sampling_mode
anchor_nn` the trainer pairs each prompt with its nearest neighbour. On this data that neighbour is
almost always the other form of the same meaning, so nearly every training pair would be positive.
`--hard-negatives` adds, for every sampled meaning, the most similar *wrong* cached sentence found by
eval_dialect_retrieval.py (column best_wrong_text of retrieval_<model>.csv) as a prompt of its own
meaning: its nearest neighbour is then usually a sampled meaning, which yields hard negative pairs.

Groups that appear in the vCache streams (stream_repeat / stream_zipf) are left out, so the splitter
is never trained on sentences it is evaluated on. Train groups come from the Training split, val
groups from the Validation split.

Outputs (in --out-dir): rl_train.parquet, rl_val.parquet (columns prompt, id_set), rl_manifest.json

Example:
  python benchmarks/dialect/make_dialect_rl_data.py --data-dir /content/drive/MyDrive/dialects/prepared \
    --out-dir /content/drive/MyDrive/dialects/prepared --train-groups 5000 --val-groups 500 \
    --hard-negatives /content/drive/MyDrive/dialects/results/retrieval_BAAI__bge-m3.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def build(pairs: pd.DataFrame, groups: np.ndarray, negatives: pd.DataFrame | None) -> pd.DataFrame:
    sub = pairs[pairs["group_id"].isin(set(groups.tolist()))]
    parts = [
        sub[["standard", "group_id"]].rename(columns={"standard": "prompt"}),
        sub[["dialect", "group_id"]].rename(columns={"dialect": "prompt"}),
    ]
    if negatives is not None:
        neg = negatives[negatives["query_group"].isin(set(groups.tolist()))]
        parts.append(neg[["best_wrong_text", "group_id"]].rename(columns={"best_wrong_text": "prompt"}))
    rows = pd.concat(parts)
    rows = rows.drop_duplicates("prompt").rename(columns={"group_id": "id_set"})
    rows["id_set"] = rows["id_set"].astype("int64")
    return rows.sample(frac=1.0, random_state=0).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Output folder of prepare_dialect_dataset.py")
    p.add_argument("--out-dir", default=None, help="Default: --data-dir")
    p.add_argument("--train-groups", type=int, default=5000, help="Meanings sampled from the Training split.")
    p.add_argument("--val-groups", type=int, default=500, help="Meanings sampled from the Validation split.")
    p.add_argument("--streams", nargs="+", default=["stream_repeat", "stream_zipf"],
                   help="Streams whose meanings are excluded from training.")
    p.add_argument("--hard-negatives", default=None,
                   help="retrieval_<model>.csv from eval_dialect_retrieval.py (use the splitter's encoder).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    pairs = pd.read_parquet(os.path.join(args.data_dir, "pairs.parquet"))
    held_out = set()
    for s in args.streams:
        path = os.path.join(args.data_dir, f"{s}.parquet")
        if os.path.exists(path):
            held_out |= set(pd.read_parquet(path, columns=["id_set"])["id_set"].astype("int64").tolist())
    pairs = pairs[~pairs["group_id"].isin(held_out)]

    negatives = None
    if args.hard_negatives:
        index = pd.read_parquet(os.path.join(args.data_dir, "index.parquet"), columns=["group_id", "text"])
        gid_of = dict(zip(index["text"], index["group_id"]))
        neg = pd.read_csv(args.hard_negatives, usecols=["group_id", "best_wrong_text"])
        neg = neg.rename(columns={"group_id": "query_group"})
        neg["group_id"] = neg["best_wrong_text"].map(gid_of)
        # 평가 스트림에 나오는 문장은 함정으로도 쓰지 않는다
        negatives = neg.dropna(subset=["group_id"])
        negatives = negatives[~negatives["group_id"].isin(held_out)].copy()
        negatives["group_id"] = negatives["group_id"].astype("int64")

    rng = np.random.default_rng(args.seed)
    manifest = {"data_dir": args.data_dir, "held_out_groups": len(held_out), "streams": args.streams,
                "hard_negatives": args.hard_negatives}
    for split, n, name in (("train", args.train_groups, "rl_train"), ("val", args.val_groups, "rl_val")):
        cand = pairs.loc[pairs["split"] == split, "group_id"].unique()
        groups = rng.choice(cand, size=min(n, len(cand)), replace=False)
        df = build(pairs[pairs["split"] == split], groups, negatives)
        df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
        manifest[name] = {"groups": int(len(groups)), "prompts": int(len(df)),
                          "prompts_per_group": round(len(df) / max(1, len(groups)), 2)}
        print(f"{name}: {len(groups)} meanings -> {len(df)} prompts")
    with open(os.path.join(out_dir, "rl_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
