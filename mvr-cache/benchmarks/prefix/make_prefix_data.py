"""Stage 0: build the shared-prefix datasets (prefix stress test).

For every split (train / val / test) and level L in {0, 2, 4, 8}, prepend the first L
slots of prefix_pool.json to each prompt. Labels and every other column stay as they are.

  variant mode: each prompt draws its own sentence per slot (seed + (dataset_name, id)),
                so the prefix differs a little between requests
  fixed mode:   the same sentences for every prompt (reference, measured at L4 only)

--pool may also be a per-task pool ({"tasks": {dataset_name: {"slots": [...]}}}, e.g.
prefix_pool_task.json): each row then takes its sentences from the pool of its own sub-task.

Output: <out>/<mode>/L<L>/<split>.parquet with extra columns
  orig_prompt, prefix_text, prefix_chars, prefix_choice, level, n_tokens, truncated,
  fits_all_levels (row is <= 512 tokens at the largest level, so no level truncates it)

Example:
  python -m benchmarks.prefix.make_prefix_data \
    --train /content/drive/MyDrive/mvr_splits/cls_seed42/train.parquet \
    --val   /content/drive/MyDrive/mvr_splits/cls_seed42/val.parquet \
    --test  /content/drive/MyDrive/mvr_splits/cls_seed42_nodedup/test41k.parquet \
    --out   /content/drive/MyDrive/mvr_eval/prefix/data
"""
# [새 파일] 0단계: 단계별(L0·L2·L4·L8) 공통 문단 데이터 만들기

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from benchmarks.prefix._shared import POOL_PATH, encode, load_tokenizer, ps


def load_pools(path: str) -> dict:
    """{dataset_name: slots} for a per-task pool, or {None: slots} for one shared pool."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "tasks" in raw:
        return {t: [slot["variants"] for slot in v["slots"]] for t, v in raw["tasks"].items()}
    return {None: ps.load_pool(path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pool", default=POOL_PATH)
    p.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4, 8])
    p.add_argument("--modes", nargs="+", default=["variant", "fixed"])
    p.add_argument("--fixed-levels", type=int, nargs="+", default=[4],
                   help="Levels built in fixed mode (reference only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--position", default="prefix", choices=["prefix", "suffix"],
                   help="Experiment 6: put the shared sentences before (default) or after the prompt")
    args = p.parse_args()

    pools = load_pools(args.pool)
    join = ps.join_suffix if args.position == "suffix" else ps.join_prompt
    tok = load_tokenizer(args.tokenizer)
    max_level = max(args.levels)
    splits = {"train": args.train, "val": args.val, "test": args.test}
    meta = {"pool": args.pool, "seed": args.seed, "levels": args.levels, "files": {}}

    for mode in args.modes:
        levels = args.levels if mode == "variant" else [L for L in args.fixed_levels if L > 0]
        for name, path in splits.items():
            df = pd.read_parquet(path).reset_index(drop=True)
            keys = (df["dataset_name"].astype(str) + ":" + df["id"].astype(str)).tolist()
            tasks = df["dataset_name"].astype(str).tolist()
            pool_of = [pools.get(t, pools.get(None)) for t in tasks]
            missing = sorted({t for t, pl in zip(tasks, pool_of) if pl is None})
            assert not missing, f"no prefix pool for tasks {missing}"
            choices = [ps.prefix_choice(k, len(pl), len(pl[0]), args.seed, mode) for k, pl in zip(keys, pool_of)]
            # 모든 단계에서 같은 행 집합을 쓰기 위해, 가장 긴 단계 기준으로 512토큰 이하인지 표시
            fits = []
            for prompt, ch, pool in zip(df["prompt"].astype(str), choices, pool_of):
                text, _ = join(ps.build_prefix(pool, ch, max_level), prompt)
                fits.append(encode(tok, text)[3] <= ps.MAX_LEN)
            for L in levels:
                rows = []
                for prompt, ch, pool in zip(df["prompt"].astype(str), choices, pool_of):
                    prefix = ps.build_prefix(pool, ch, L)
                    text, start = join(prefix, prompt)
                    n_chars = len(prefix)
                    if args.position == "prefix":
                        start = 0
                    full = encode(tok, text)[3]
                    rows.append((text, prompt, prefix, n_chars, ",".join(map(str, ch[:L])), full, start))
                out = df.copy()
                out["prompt"] = [r[0] for r in rows]
                out["orig_prompt"] = [r[1] for r in rows]
                out["prefix_text"] = [r[2] for r in rows]
                out["prefix_chars"] = [r[3] for r in rows]
                out["prefix_choice"] = [r[4] for r in rows]
                out["prefix_position"] = args.position
                out["prefix_start"] = [r[6] for r in rows]
                out["level"] = L
                out["n_tokens"] = [r[5] for r in rows]
                out["truncated"] = out["n_tokens"] > ps.MAX_LEN
                out["fits_all_levels"] = fits
                d = os.path.join(args.out, mode, f"L{L}")
                os.makedirs(d, exist_ok=True)
                f = os.path.join(d, f"{name}.parquet")
                out.to_parquet(f, index=False)
                meta["files"][f"{mode}/L{L}/{name}"] = {
                    "rows": len(out), "truncated": int(out["truncated"].sum()),
                    "fits_all_levels": int(sum(fits)),
                }
                print(f"{mode} L{L} {name}: {len(out)} rows, truncated {int(out['truncated'].sum())}, "
                      f"fits all levels {sum(fits)}")
    with open(os.path.join(args.out, "data_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
