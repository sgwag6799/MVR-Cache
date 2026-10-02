"""Build group-disjoint RL training splits and an evaluation stream from LmArena.

Every ID_Set group is assigned to exactly one of train / val / test / eval, so the
trained splitter never sees paraphrases of prompts in the evaluation stream.
The evaluation stream keeps the original dataset order.

anchor_nn sampling only trains on (prompt, nearest neighbour) pairs, so a negative
pair only appears when a prompt's nearest neighbour belongs to another group.
Training groups are therefore capped at `--max-per-group` prompts, and a
`--singleton-frac` share of them keeps a single prompt (no paraphrase available),
which mirrors a cache stream where most first-seen prompts have no match.

Example:
  python scripts/prepare_lmarena_splits.py \
    --input ../mvr-cache/data/lmarena.parquet --out-dir ../mvr-cache/data/lmarena_rl
"""

import argparse
import os

import numpy as np
import pandas as pd


def sample_split(df: pd.DataFrame, groups: list, label_col: str, target: int,
                 max_per_group: int, singleton_frac: float, rng: np.random.Generator,
                 seed: int) -> tuple[pd.DataFrame, int]:
    rows, total, n_single = [], 0, 0
    for g in groups:
        if total >= target:
            break
        members = df[df[label_col] == g]
        k = 1 if rng.random() < singleton_frac else min(len(members), max_per_group)
        n_single += int(k == 1)
        rows.append(members.sample(n=k, random_state=seed))
        total += k
    return pd.concat(rows).sample(frac=1.0, random_state=seed), n_single


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label-col", default="ID_Set")
    p.add_argument("--train-group-frac", type=float, default=0.25)
    p.add_argument("--val-group-frac", type=float, default=0.05)
    p.add_argument("--test-group-frac", type=float, default=0.05)
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--val", type=int, default=300)
    p.add_argument("--test", type=int, default=300)
    p.add_argument("--max-per-group", type=int, default=2)
    p.add_argument("--singleton-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.input)
    groups = df[args.label_col].unique().tolist()
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * args.train_group_frac)
    n_val = int(n * args.val_group_frac)
    n_test = int(n * args.test_group_frac)
    split_groups = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val : n_train + n_val + n_test],
    }
    eval_groups = set(groups[n_train + n_val + n_test :])

    os.makedirs(args.out_dir, exist_ok=True)
    for name, target in [("train", args.train), ("val", args.val), ("test", args.test)]:
        part, n_single = sample_split(
            df, split_groups[name], args.label_col, target,
            args.max_per_group, args.singleton_frac, rng, args.seed,
        )
        path = os.path.join(args.out_dir, f"{name}.parquet")
        part.reset_index(drop=True).to_parquet(path, index=False)
        print(f"{name}: {len(part)} prompts, {part[args.label_col].nunique()} groups "
              f"({n_single} singletons) -> {path}")

    stream = df[df[args.label_col].isin(eval_groups)].reset_index(drop=True)
    path = os.path.join(args.out_dir, "eval_stream.parquet")
    stream.to_parquet(path, index=False)
    print(f"eval_stream: {len(stream)} prompts, {len(eval_groups)} groups (original order) -> {path}")


if __name__ == "__main__":
    main()
