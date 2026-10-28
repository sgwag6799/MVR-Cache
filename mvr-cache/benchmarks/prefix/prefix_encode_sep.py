"""Experiment 3: separate the shared prefix at encoding time instead of re-weighting segments.

Sections 10-13 found that most of the loss is contextual contamination: the original segments
change once the prefix is encoded in front of them, so no segment weighting can reach R. This
script keeps the level-L prompt and changes how it is encoded, on the same test anchors and with
the same scoring (uniform SMaxSim = M2, prefix segments weight 0 = M4):

  boundary   oracle  the true prefix length (prefix_chars from make_prefix_data)
             freq    label-free: the leading run of sentences that each occur in >= --freq-min of
                     the level's train prompts (sentences split after . ! ? + space). No pool file
  encoding   full    the current encoding (reference: M2_full = 1단계 M2, M4_full = 1단계 M4)
             mask    one forward pass with a block attention mask: [CLS], the original tokens and
                     [SEP] cannot attend to prefix tokens (prefix tokens see everything). Positions
                     are left as they are. The single vector averages [CLS] + original + [SEP]
             strip   the text after the boundary is encoded alone (strip_oracle would be exactly L0,
                     so only strip_freq is run)

Every condition searches its own 10 nearest neighbours with its own single vector. R = M2 at L0.
Reports auc_selected, auc_pairs, top1_acc, ceiling10 and paired bootstrap CIs against R, plus how
often the freq boundary equals the true one.

Example:
  python -m benchmarks.prefix.prefix_encode_sep --data ../data/prefix/data --out ../data/prefix/encode_sep \
    --resp-col response_llama_3_8b
"""
# [새 파일] 실험 3: 재순위 가중치 대신 인코딩 단계에서 공통 문단을 분리

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from benchmarks.prefix._shared import ps
from benchmarks.prefix.prefix_offline import auc, clear_pad_cache, metrics, neighbours, score_pairs

SENT_END = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------------------- 경계 찾기


def sentence_spans(text):
    """(start, end) character spans of sentences, split after . ! ? followed by whitespace."""
    spans, pos = [], 0
    for m in SENT_END.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return spans


def freq_table(texts):
    """How many prompts contain each sentence (a prompt counts a sentence once)."""
    cnt = Counter()
    for t in texts:
        cnt.update({t[a:b] for a, b in sentence_spans(t)})
    return cnt


def freq_boundary(text, cnt, min_count):
    """End (characters) of the leading run of frequent sentences, 0 if the first one is rare.
    Never removes the whole prompt: the last sentence always stays."""
    spans = sentence_spans(text)
    end = 0
    for a, b in spans[:-1]:
        if cnt.get(text[a:b], 0) < min_count:
            break
        end = b
    return end


# ----------------------------------------------------------------------------- 인코딩


@torch.no_grad()
def encode(texts, bounds, tok, lm, device, max_cuts, cap_scope, block, batch_size=64):
    """Segment vectors / prefix flags / single vectors like prefix_offline.encode_split, with the
    prefix ending at `bounds` characters. block=True applies the block attention mask."""
    order = np.argsort([len(t) for t in texts])
    segs, flags, knn = [None] * len(texts), [None] * len(texts), [None] * len(texts)
    for b in range(0, len(texts), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=ps.MAX_LEN, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        am = enc["attention_mask"].bool()
        pends = []
        for j, i in enumerate(idx):
            length = int(am[j].sum())
            pends.append(ps.prefix_token_end([tuple(o) for o in offsets[j][:length]], bounds[i]))
        enc = {k: v.to(device) for k, v in enc.items()}
        if block:
            n, L = am.shape
            vis = am.unsqueeze(1).expand(n, L, L).clone()  # 패딩 열은 못 봄
            for j, pend in enumerate(pends):
                if pend > 1:
                    vis[j, 0, 1:pend] = False      # [CLS]는 공통 문단을 못 봄
                    vis[j, pend:, 1:pend] = False  # 원본 · [SEP]도 못 봄
            mask = torch.zeros(n, 1, L, L).masked_fill(~vis.unsqueeze(1), -1e4).to(device)
            kwargs = {"input_ids": enc["input_ids"], "attention_mask": mask}
            if "token_type_ids" in enc:
                kwargs["token_type_ids"] = enc["token_type_ids"]
        else:
            kwargs = enc
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            hs = lm(**kwargs).last_hidden_state
        hs = hs.float()
        for j, i in enumerate(idx):
            length = int(am[j].sum())
            tokens = tok.convert_ids_to_tokens(enc["input_ids"][j][:length].tolist())
            pend = pends[j]
            cuts = ps.cut_points(tokens, length, max_cuts=max_cuts, cap_scope=cap_scope, prefix_end=pend)
            spans = ps.segment_spans(length, cuts)
            h = hs[j]
            segs[i] = torch.stack([h[s:e].mean(0) for s, e in spans]).half().cpu()
            flags[i] = torch.tensor(ps.prefix_segment_flags(spans, pend))
            keep = torch.cat([h[:1], h[pend:length]]) if block and pend > 1 else h[:length]
            knn[i] = keep.mean(0).cpu()
    return (segs, flags), F.normalize(torch.stack(knn), dim=-1)


# ----------------------------------------------------------------------------- 부트스트랩


def boot_ci(cur, ref, n_boot, seed=0):
    """Paired bootstrap over anchors: 95% CI of (cur - ref) for auc_selected and auc_pairs."""
    rng = np.random.default_rng(seed)
    out = {"auc_selected": [], "auc_pairs": []}
    (s1, c1), (s0, c0) = cur, ref
    i = np.arange(len(s1))
    for _ in range(n_boot):
        b = rng.integers(0, len(s1), len(s1))
        t1, t0 = s1[b].argmax(1), s0[b].argmax(1)
        out["auc_selected"].append(auc(s1[b][i, t1], c1[b][i, t1]) - auc(s0[b][i, t0], c0[b][i, t0]))
        out["auc_pairs"].append(auc(s1[b].ravel(), c1[b].ravel()) - auc(s0[b].ravel(), c0[b].ravel()))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


# ----------------------------------------------------------------------------- 메인


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Output folder of make_prefix_data.py")
    p.add_argument("--out", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--fixed-levels", type=int, nargs="*", default=[],
                   help="fixed mode keeps a slightly different row set; only valid when it matches variant L0")
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--anchors", type=int, default=6000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pair-seed", type=int, default=0)
    p.add_argument("--max-cuts", type=int, default=32)
    p.add_argument("--cap-scope", default="original", choices=["original", "all"])
    p.add_argument("--resp-col", default="response_llama_3_8b")
    p.add_argument("--freq-min", type=float, default=0.01,
                   help="A sentence is shared when it occurs in at least this share of the level's train prompts")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModel.from_pretrained(args.model).to(args.device).eval()

    def load(mode, level, split="test"):
        df = pd.read_parquet(os.path.join(args.data, mode, f"L{level}", f"{split}.parquet"))
        return df[df["fits_all_levels"]].reset_index(drop=True)

    df0 = load("variant", 0)
    n = len(df0)
    labels = df0[args.resp_col].map(ps.normalize_answer).to_numpy()
    anchors = np.sort(np.random.default_rng(args.pair_seed).choice(n, min(args.anchors, n), replace=False))
    qi = np.repeat(anchors, args.k)

    def evaluate(bank, knn, conds):
        nb = neighbours(knn, anchors, args.k, args.device)
        ci = nb.reshape(-1)
        c = (labels[qi] == labels[ci]).astype(int).reshape(-1, args.k)
        return {cond: (score_pairs(cond, bank, bank, qi, ci, args.device).reshape(-1, args.k), c) for cond in conds}

    t0 = time.time()
    bank0, knn0 = encode(df0["prompt"].astype(str).tolist(), [0] * n, tok, lm, args.device,
                         args.max_cuts, args.cap_scope, block=False)
    ref = evaluate(bank0, knn0, ["M2"])["M2"]
    r = metrics(*ref)
    print(f"[R / L0] {time.time() - t0:.0f}s auc_sel={r['auc_selected']:.4f} auc_pairs={r['auc_pairs']:.4f}")

    rows = [{"mode": "variant", "level": 0, "cell": "R", **r}]
    summary = {"R": r}
    for mode, level in [("variant", L) for L in args.levels] + [("fixed", L) for L in args.fixed_levels]:
        clear_pad_cache()
        t0 = time.time()
        dfL = load(mode, level)
        assert len(dfL) == n and (dfL["orig_prompt"].to_numpy() == df0["prompt"].to_numpy()).all(), \
            f"{mode} L{level}: rows differ from L0 (fixed mode is not aligned with variant L0 here)"
        texts = dfL["prompt"].astype(str).tolist()
        oracle = dfL["prefix_chars"].astype(int).tolist()
        train = load(mode, level, "train")["prompt"].astype(str).tolist()
        cnt = freq_table(train)
        min_count = max(2, int(np.ceil(args.freq_min * len(train))))
        freq = [freq_boundary(t, cnt, min_count) for t in texts]
        exact = float(np.mean([a == b for a, b in zip(freq, oracle)]))
        under = float(np.mean([a < b for a, b in zip(freq, oracle)]))
        over = float(np.mean([a > b for a, b in zip(freq, oracle)]))

        res = {}
        bank, knn = encode(texts, oracle, tok, lm, args.device, args.max_cuts, args.cap_scope, block=False)
        res.update({f"{c}_full": v for c, v in evaluate(bank, knn, ["M2", "M4"]).items()})
        bank, knn = encode(texts, oracle, tok, lm, args.device, args.max_cuts, args.cap_scope, block=True)
        res.update({f"{c}_mask_oracle": v for c, v in evaluate(bank, knn, ["M2", "M4"]).items()})
        bank, knn = encode(texts, freq, tok, lm, args.device, args.max_cuts, args.cap_scope, block=True)
        res.update({f"{c}_mask_freq": v for c, v in evaluate(bank, knn, ["M2", "M4"]).items()})
        stripped = [t[b:].lstrip() for t, b in zip(texts, freq)]
        bank, knn = encode(stripped, [0] * n, tok, lm, args.device, args.max_cuts, args.cap_scope, block=False)
        res["M2_strip_freq"] = evaluate(bank, knn, ["M2"])["M2"]

        s = {"boundary_freq": {"exact": exact, "under": under, "over": over, "min_count": min_count}}
        for cell, v in res.items():
            m = metrics(*v)
            rows.append({"mode": mode, "level": level, "cell": cell, **m})
            ci = boot_ci(v, ref, args.bootstrap)
            s[cell] = {**m, "sel_minus_R": m["auc_selected"] - r["auc_selected"], "sel_ci95": ci["auc_selected"],
                       "pairs_minus_R": m["auc_pairs"] - r["auc_pairs"], "pairs_ci95": ci["auc_pairs"]}
        summary[f"{mode}/L{level}"] = s
        print(f"[{mode} L{level}] {time.time() - t0:.0f}s freq boundary exact={exact:.3f} under={under:.3f} over={over:.3f}  "
              + "  ".join(f"{c}={s[c]['auc_selected']:.4f}/{s[c]['auc_pairs']:.4f}" for c in res))
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "prefix_encode_sep.csv"), index=False)

    meta = {"args": vars(args), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                            text=True).stdout.strip(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "summary": summary}
    with open(os.path.join(args.out, "prefix_encode_sep.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
