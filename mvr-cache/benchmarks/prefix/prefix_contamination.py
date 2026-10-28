"""Exploratory: is the M4 < R gap caused by contextual contamination or by the candidate set?

At level L, M4 drops the prefix segments but the original segments were still encoded with
the prefix in front of them (attention mixes the prefix into every token). It also scores the
level-L candidates (nearest neighbours by the prefix-contaminated single vector). This script
separates the two effects on the same test anchors with a 2 x 2 per level:

  segments  ctx  original segments from the level-L encoding (= M4)
            iso  original encoded alone (= the L0 segments of the same row; the original part is
                 cut the same way at every level, which is checked here)
  pairs     L    neighbours searched with the level-L single vector
            L0   neighbours searched with the L0 single vector (the pairs R is scored on)

  ctx x L  = M4,   iso x L0 = R
  iso x L  -> only the candidate set changed    ctx x L0 -> only the segment vectors changed

Also reported: M2 on level-L pairs, neighbour overlap with L0, mean cosine between the
contaminated and isolated vector of each original segment, and paired bootstrap CIs (anchors)
for the differences against R.

Example:
  python -m benchmarks.prefix.prefix_contamination --data data/prefix/data --out data/prefix/contamination
"""
# [새 파일] 탐색: M4와 R의 차이를 문맥 오염 / 후보 집합 변화로 나눈다

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from benchmarks.prefix import prefix_offline
from benchmarks.prefix._shared import ps
from benchmarks.prefix.prefix_offline import (auc, clear_pad_cache, encode_split, metrics, neighbours,
                                              score_pairs)

CELLS = ("M2_L", "ctx_L", "iso_L", "ctx_L0", "iso_L0")


def original_only(bank):
    """Drop prefix segments: (segs, flags) -> (segs, all-False flags) with only original segments."""
    segs, flags = bank
    out = [s[~f] for s, f in zip(segs, flags)]
    return out, [torch.zeros(len(s), dtype=torch.bool) for s in out]


def bootstrap(cell, ref, n=2000, seed=0):
    """Paired bootstrap over anchors of auc_selected(cell) - auc_selected(ref). Returns (lo, hi)."""
    def selected(sc, c):
        top = sc.argmax(1)
        return sc[np.arange(len(top)), top], c[np.arange(len(top)), top]

    (s1, c1), (s0, c0) = selected(*cell), selected(*ref)
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for b in range(n):
        j = rng.integers(0, len(s0), len(s0))
        d[b] = auc(s1[j], c1[j]) - auc(s0[j], c0[j])
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Output folder of make_prefix_data.py")
    p.add_argument("--out", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--fixed-levels", type=int, nargs="*", default=[4])
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--anchors", type=int, default=6000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pair-seed", type=int, default=0)
    p.add_argument("--max-cuts", type=int, default=32)
    p.add_argument("--cap-scope", default="original", choices=["original", "all"])
    p.add_argument("--resp-col", default="response_llama_3_8b")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--max-len", type=int, default=ps.MAX_LEN,
                   help="Experiment 6: encoder input limit in its own tokens (e.g. 1024 for BAAI/bge-m3)")
    p.add_argument("--cut-punct", default=None,
                   help="Experiment 5: cut tokens as one string, e.g. '.!?' (default: . , ! ? ; :)")
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer

    prefix_offline.CUT_PUNCT = set(args.cut_punct) if args.cut_punct else None
    ps.MAX_LEN = args.max_len
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModel.from_pretrained(args.model).to(args.device).eval()

    def load(mode, level):
        return pd.read_parquet(os.path.join(args.data, mode, f"L{level}", "test.parquet"))

    def encode(df):
        segs, flags, knn, _ = encode_split(df, tok, lm, args.device, args.max_cuts, args.cap_scope)
        return (segs, flags), knn

    raw0 = load("variant", 0)
    keep0 = raw0["fits_all_levels"].to_numpy()
    df0 = raw0[keep0].reset_index(drop=True)
    pos0 = np.full(len(raw0), -1)
    pos0[keep0] = np.arange(keep0.sum())  # 원본 41K 위치 -> L0 필터 후 위치
    bank0, knn0 = encode(df0)
    labels0 = df0[args.resp_col].map(ps.normalize_answer).to_numpy()

    rows, summary = [], {}
    for mode, level in [("variant", L) for L in args.levels] + [("fixed", L) for L in args.fixed_levels]:
        clear_pad_cache()
        t0 = time.time()
        # 같은 원본 행끼리 맞춘다: 두 데이터 모두 512토큰 안에 드는 행만 (fixed는 행 집합이 조금 다름)
        rawL = load(mode, level)
        assert (rawL["orig_prompt"].to_numpy() == raw0["orig_prompt"].to_numpy()).all()
        keep = rawL["fits_all_levels"].to_numpy() & keep0
        dfL, idx0 = rawL[keep].reset_index(drop=True), pos0[keep]
        assert (dfL[args.resp_col].to_numpy() == df0[args.resp_col].to_numpy()[idx0]).all()

        bankL, knnL = encode(dfL)
        ctx = original_only(bankL)
        iso = ([bank0[0][i] for i in idx0], [bank0[1][i] for i in idx0])
        n_ctx = np.array([len(s) for s in ctx[0]])
        n_iso = np.array([len(s) for s in iso[0]])
        aligned = n_ctx == n_iso
        cos = [torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()
               for a, b, ok in zip(ctx[0], iso[0], aligned) if ok]

        labels = labels0[idx0]
        n = len(dfL)
        anchors = np.sort(np.random.default_rng(args.pair_seed).choice(n, min(args.anchors, n), replace=False))
        pairs = {"L": neighbours(knnL, anchors, args.k, args.device),
                 "L0": neighbours(knn0[torch.as_tensor(idx0)], anchors, args.k, args.device)}
        tasks = dfL["dataset_name"].to_numpy()[anchors]
        overlap = float(np.mean([len(set(a) & set(b)) / args.k for a, b in zip(pairs["L"], pairs["L0"])]))

        res = {}
        for cell in CELLS:
            name, pk = cell.rsplit("_", 1)
            qi, ci = np.repeat(anchors, args.k), pairs[pk].reshape(-1)
            c = (labels[qi] == labels[ci]).astype(int).reshape(-1, args.k)
            cond, bank = ("M2", bankL) if name == "M2" else ("M2", ctx if name == "ctx" else iso)
            sc = score_pairs(cond, bank, bank, qi, ci, args.device).reshape(-1, args.k)
            res[cell] = (sc, c)
            for task in ["ALL"] + sorted(set(tasks)):
                m = np.ones(len(tasks), bool) if task == "ALL" else tasks == task
                rows.append({"mode": mode, "level": level, "cell": cell, "task": task, **metrics(sc[m], c[m])})

        ref = res["iso_L0"]
        r = metrics(*ref)["auc_selected"]
        s = {"n_rows": int(n), "aligned_share": float(aligned.mean()),
             "mean_cos_ctx_vs_iso": float(np.mean(cos)), "neighbour_overlap_with_L0": overlap, "R": r}
        for cell in CELLS:
            v = metrics(*res[cell])["auc_selected"]
            s[cell] = {"auc_selected": v, "minus_R": v - r,
                       "ci95": bootstrap(res[cell], ref, args.bootstrap) if cell != "iso_L0" else None}
        summary[f"{mode}/L{level}"] = s
        print(f"[{mode} L{level}] {time.time() - t0:.0f}s  aligned={s['aligned_share']:.3f} "
              f"cos={s['mean_cos_ctx_vs_iso']:.4f} overlap={overlap:.3f}  " +
              "  ".join(f"{cl}={s[cl]['auc_selected']:.4f}" for cl in CELLS))
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "prefix_contamination.csv"), index=False)

    meta = {"args": vars(args), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                            text=True).stdout.strip(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "summary": summary}
    with open(os.path.join(args.out, "prefix_contamination.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
