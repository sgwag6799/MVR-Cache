"""Build the statistics used by RulePunctuationSplitter segment weighting (Step 3a/3b).

Uses a reference slice of the dataset that lies *outside* the evaluation stream
(default: rows 10000..13000, since evaluations use the first 10000 rows), and writes one
file with:

  - idf:       [vocab] tensor, log((N + 1) / (df + 1)) + 1 over reference prompts
  - centroid:  [H] mean of L2-normalised rule-split segment embeddings
  - mlp_state: SegmentWeightMLP trained with BCE so that the weighted, fullcos-mixed MaxSim
               score (the exact eval score with --mix-fullcos --include-full-embedding)
               separates correct from incorrect candidate pairs

Candidate pairs mirror the cache: each prompt's top-k neighbours by single-vector cosine.
Anchors are split 80/20 so the MLP is scored on pairs it was not trained on.

Example:
  python benchmarks/build_segment_weight_stats.py \
    --dataset data/classification.parquet --response-col response_llama_3_8b \
    --out results/classification_segment_weight_stats.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.RuleSplitter import (
    MIN_WEIGHT,
    RulePunctuationSplitter,
    SegmentWeightMLP,
)


def pair_scores(rows, mask, full_nocls, ai, bi, seg_w):
    """Weighted MaxSim (+ full-cos mix) for pairs (ai, bi), matching the eval scoring.

    rows: [N, S, H] segment rows (last valid row is the full embedding), mask: [N, S] bool,
    seg_w: [N, S] weights for every row (full-row weight already set to the segment mean).
    """
    q, c = F.normalize(rows[ai], dim=-1), F.normalize(rows[bi], dim=-1)
    qm, cm = mask[ai], mask[bi]
    cos = torch.bmm(q, c.transpose(1, 2))
    cos = cos.masked_fill(~cm.unsqueeze(1), -2.0).masked_fill(~qm.unsqueeze(2), -2.0)
    row_max, col_max = cos.max(dim=2).values, cos.max(dim=1).values
    wq, wc = seg_w[ai] * qm, seg_w[bi] * cm
    row = (row_max.clamp_min(-1) * wq).sum(1) / (wq.sum(1) + 1e-8)
    col = (col_max.clamp_min(-1) * wc).sum(1) / (wc.sum(1) + 1e-8)
    maxsim01 = ((0.5 * (row + col) + 1.0) * 0.5).clamp(0, 1)
    fullcos01 = (
        (F.cosine_similarity(full_nocls[ai], full_nocls[bi], dim=-1) + 1.0) * 0.5
    ).clamp(0, 1)
    return 0.5 * (maxsim01 + fullcos01)


def rows_weights(mlp, rows, mask, n_seg):
    """MLP weights for segment rows; the full row gets the mean segment weight."""
    w = mlp(rows)  # [N, S]
    idx = torch.arange(rows.shape[1]).unsqueeze(0)
    seg_mask = idx < n_seg.unsqueeze(1)
    seg_mean = (w * seg_mask).sum(1) / seg_mask.sum(1).clamp_min(1)
    full_pos = idx == n_seg.unsqueeze(1)
    w = torch.where(full_pos, seg_mean.unsqueeze(1), w)
    return torch.where(mask, w, torch.zeros_like(w))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--start",
        type=int,
        default=10000,
        help="First reference row (after the eval stream).",
    )
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--label-col", default="ID_Set")
    p.add_argument("--response-col", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-segments", type=int, default=4)
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    df = (
        pd.read_parquet(args.dataset)
        .iloc[args.start : args.start + args.n]
        .reset_index(drop=True)
    )
    prompts = df["prompt"].astype(str).tolist()
    if args.response_col:
        responses = df[args.response_col].astype(str).tolist()

        def same(i, j):
            return int(answers_have_same_meaning_static(responses[i], responses[j]))
    else:
        labels = df[args.label_col].to_numpy()

        def same(i, j):
            return int(labels[i] == labels[j])

    embedder = EmbeddingModel(device=args.device)
    # 이거 분할기를 고정으로 만들고 있음
    splitter = RulePunctuationSplitter(
        device=args.device,
        embedding_model=embedder,
        max_segments=args.max_segments,
        include_full_embedding=True,
    )

    vocab = len(splitter.embedding_model.tokenizer)
    doc_freq = torch.zeros(vocab)
    tensors, knn, nocls = [], [], []
    for text in tqdm(prompts, desc="encode+split"):
        enc = splitter.encode_text(text)
        doc_freq[torch.unique(enc["input_ids"][: enc["length"]].cpu())] += 1
        tensors.append(splitter.split_text_return_maxsim_tensor_from_encoded(enc).cpu())
        knn.append(enc["pooled_knn"].float().cpu())
        nocls.append(enc["pooled_no_cls"].float().cpu())

    n = len(prompts)
    idf = torch.log((n + 1) / (doc_freq + 1)) + 1.0
    seg_rows = torch.cat([F.normalize(t[:-1], dim=-1) for t in tensors])
    centroid = seg_rows.mean(0)

    if args.epochs == 0:
        # idf/centroid only. The MLP path builds an N x N similarity matrix, which is far
        # too large when the reference corpus is a whole dataset (45k prompts -> 8 GB).
        torch.save(
            {
                "idf": idf,
                "centroid": centroid,
                "meta": {
                    "dataset": args.dataset,
                    "start": args.start,
                    "n": n,
                    "max_segments": args.max_segments,
                    "mlp": False,
                    "min_weight": MIN_WEIGHT,
                },
            },
            args.out,
        )
        print(f"saved (idf + centroid, no MLP) -> {args.out}")
        return

    # Pad rows: [N, S, H]; last valid row of each prompt is its full embedding.
    s_max = max(t.shape[0] for t in tensors)
    H = tensors[0].shape[1]
    rows = torch.zeros(n, s_max, H)
    mask = torch.zeros(n, s_max, dtype=torch.bool)
    n_seg = torch.tensor([t.shape[0] - 1 for t in tensors])
    for i, t in enumerate(tensors):
        rows[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    full_nocls = torch.stack(nocls)

    knn_n = F.normalize(torch.stack(knn), dim=-1)
    sims = knn_n @ knn_n.T
    sims.fill_diagonal_(-2.0)
    topk = torch.topk(sims, k=min(args.k, n - 1), dim=1).indices
    anchors = rng.permutation(n)
    n_train = int(0.8 * n)
    split = {"train": anchors[:n_train], "val": anchors[n_train:]}
    pairs = {}
    for name, idx in split.items():
        ai = torch.tensor([a for a in idx for _ in range(topk.shape[1])])
        bi = topk[torch.tensor(idx)].reshape(-1)
        y = torch.tensor(
            [same(int(a), int(b)) for a, b in zip(ai, bi)], dtype=torch.float32
        )
        pairs[name] = (ai, bi, y)
        print(f"{name}: {len(y)} pairs, positive rate {y.mean():.3f}")

    mlp = SegmentWeightMLP(dim=H, hidden=args.mlp_hidden)
    calib = torch.nn.Parameter(torch.tensor([10.0, -5.0]))  # logit = a * score + b
    opt = torch.optim.Adam(list(mlp.parameters()) + [calib], lr=args.lr)
    ai, bi, y = pairs["train"]
    pos_weight = (1 - y.mean()) / y.mean().clamp_min(1e-6)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(y))
        total = 0.0
        for start in range(0, len(y), 512):
            b = perm[start : start + 512]
            w = rows_weights(mlp, rows, mask, n_seg)
            s = pair_scores(rows, mask, full_nocls, ai[b], bi[b], w)
            loss = F.binary_cross_entropy_with_logits(
                calib[0] * s + calib[1], y[b], pos_weight=pos_weight
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: train BCE {total / len(y):.4f}")

    uniform_w = torch.where(mask, torch.ones(n, s_max), torch.zeros(n, s_max))
    with torch.no_grad():
        mlp_w = rows_weights(mlp, rows, mask, n_seg)
    for name, (a, b, yy) in pairs.items():
        auc_u = roc_auc_score(
            yy.numpy(), pair_scores(rows, mask, full_nocls, a, b, uniform_w).numpy()
        )
        auc_m = roc_auc_score(
            yy.numpy(), pair_scores(rows, mask, full_nocls, a, b, mlp_w).numpy()
        )
        print(f"{name} AUC  uniform {auc_u:.4f}  mlp {auc_m:.4f}")

    torch.save(
        {
            "idf": idf,
            "centroid": centroid,
            "mlp_state": mlp.state_dict(),
            "mlp_hidden": args.mlp_hidden,
            "meta": {
                "dataset": args.dataset,
                "start": args.start,
                "n": n,
                "k": args.k,
                "max_segments": args.max_segments,
                "seed": args.seed,
                "min_weight": MIN_WEIGHT,
            },
        },
        args.out,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
