"""Stage 0 checks for the prefix datasets (tokenizer only, no model).

Per mode / level / split:
  - token counts (mean, p50, p90, max) and how many rows exceed 512 tokens (truncated)
  - segments per prompt under the fixed split rule
  - every prefix sentence is exactly one segment (prefix segments == L)
  - the original part is split exactly as at L0 (same token ids per original segment)
  - how many rows the cap changes, for cap_scope original vs all

Writes prefix_sanity.json next to the data.

Example:
  python -m benchmarks.prefix.prefix_sanity --data /content/drive/MyDrive/mvr_eval/prefix/data
"""
# [새 파일] 0단계 점검: 토큰 수·잘림·조각 수·원본 부분 분할 불변 여부

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from benchmarks.prefix._shared import encode, load_tokenizer, ps


def split_rows(tok, df, max_cuts, cap_scope):
    """Per row: (original-part segments as token-id tuples, n prefix segments, n segments, n cuts uncapped)."""
    out = []
    for text, n_chars in zip(df["prompt"].astype(str), df["prefix_chars"].astype(int)):
        ids, tokens, offsets, _ = encode(tok, text)
        length = len(ids)
        pend = ps.prefix_token_end(offsets, n_chars)
        cuts = ps.cut_points(tokens, length, max_cuts=max_cuts, cap_scope=cap_scope, prefix_end=pend)
        spans = ps.segment_spans(length, cuts)
        flags = ps.prefix_segment_flags(spans, pend)
        orig = tuple(tuple(ids[s:e]) for (s, e), f in zip(spans, flags) if not f)
        uncapped = len(ps.cut_points(tokens, length, max_cuts=None, prefix_end=pend))
        out.append((orig, sum(flags), len(spans), uncapped))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--max-cuts", type=int, default=32)
    p.add_argument("--cap-scope", default="original", choices=["original", "all"])
    p.add_argument("--tokenizer", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--max-rows", type=int, default=None, help="Check only the first N rows (quick look).")
    args = p.parse_args()
    tok = load_tokenizer(args.tokenizer)

    report = {"max_cuts": args.max_cuts, "cap_scope": args.cap_scope, "checks": {}}
    base = {}  # (split) -> L0 original segments
    for f in sorted(glob.glob(os.path.join(args.data, "*", "L*", "*.parquet")),
                    key=lambda x: (x.split(os.sep)[-3] != "variant", int(x.split(os.sep)[-2][1:]))):
        mode, level, split = f.split(os.sep)[-3], int(f.split(os.sep)[-2][1:]), os.path.basename(f)[:-8]
        df = pd.read_parquet(f)
        if args.max_rows:
            df = df.head(args.max_rows)
        rows = split_rows(tok, df, args.max_cuts, args.cap_scope)
        rows_all = split_rows(tok, df, args.max_cuts, "all") if level > 0 else rows
        n_tok = df["n_tokens"].to_numpy()
        n_seg = np.array([r[2] for r in rows])
        fits = df["fits_all_levels"].to_numpy(bool)
        c = {
            "rows": int(len(df)),
            "tokens": {"mean": round(float(n_tok.mean()), 1), "p50": int(np.median(n_tok)),
                       "p90": int(np.percentile(n_tok, 90)), "max": int(n_tok.max())},
            "truncated": int((n_tok > ps.MAX_LEN).sum()),
            "truncated_among_fits_all_levels": int(((n_tok > ps.MAX_LEN) & fits).sum()),
            "rows_fit_all_levels": int(fits.sum()),
            "segments": {"mean": round(float(n_seg.mean()), 2), "p90": int(np.percentile(n_seg, 90)),
                         "max": int(n_seg.max())},
            "rows_hitting_cap": int(sum(r[3] - (r[1] if level else 0) > args.max_cuts for r in rows)),
            "prefix_segments_ne_level": int(sum(r[1] != level for r in rows)),
        }
        if mode == "variant" and level == 0:
            base[split] = [r[0] for r in rows]
        if level > 0 and split in base:
            same = [a == b[0] for a, b in zip(base[split], rows)]
            same_all = [a == b[0] for a, b in zip(base[split], rows_all)]
            fit_idx = np.where(fits[: len(same)])[0]
            c["original_split_differs_from_L0"] = int(len(same) - sum(same))
            c["original_split_differs_from_L0_among_fits"] = int(sum(not same[i] for i in fit_idx))
            c["original_split_differs_if_cap_scope_all"] = int(len(same_all) - sum(same_all))
        report["checks"][f"{mode}/L{level}/{split}"] = c
        print(f"{mode} L{level} {split}: {json.dumps(c)}")

    out = os.path.join(args.data, "prefix_sanity.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
