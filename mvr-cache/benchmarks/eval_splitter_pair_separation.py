"""Score-only comparison of the vCache and MVR-cache similarity functions (no cache policy).

For each prompt in a held-out pool, the cache would score the top-k cached neighbours
found by single-vector retrieval. This script builds exactly those candidate pairs,
labels them by ID_Set equality, and compares how well each similarity separates
same-meaning from different-meaning pairs:

  - vcache:   cosine of the BGE mean-pooled embedding (what the vCache baseline uses)
  - mvr:      0.5 * MaxSim(segments + full row) + 0.5 * full cosine
              (= --include-full-embedding --mix-fullcos in eval_sembenchmark_verified_splitter.py)
  - maxsim:   MaxSim(segments + full row) alone

The mvr/maxsim scores are computed for three segmentations, all restricted to the
punctuation split points the RL policy may choose from:

  - learned:  the trained RL splitter
  - rule:     split at every punctuation mark (excluding the final one), capped at
              --max-segments boundaries
  - random:   per prompt, the same number of boundaries as the learned splitter,
              placed at random punctuation positions (isolates *where* from *how many*)

Reported: ROC-AUC over candidate pairs, top-1 reranking accuracy, and how many
segments each segmentation produces.

Example:
  python benchmarks/eval_splitter_pair_separation.py \
    --dataset data/lmarena_rl/eval_stream.parquet --n 2000 --k 10 \
    --splitter-checkpoint ../rl-training-algorithm/checkpoints/lmarena_heldout_ckpt/epoch=25-step=4160.ckpt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.vcache_core.splitter.RuleSplitter import PUNCT_CHARS, punctuation_positions
from vcache.vcache_policy.strategies.verified_splitter import VerifiedSplitterDecisionPolicy


def tensor_from_pointers(enc: dict, pointers: list) -> torch.Tensor:
    sent, full = MaxSimSplitter._segment_embeds_from_pointers(enc["token_emb"], enc["length"], pointers)
    return torch.cat([sent, full], dim=0).float().cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--label-col", default="ID_Set")
    p.add_argument(
        "--response-col",
        default=None,
        help="If set, a pair is correct when the responses match under the same static string "
        "comparison as --similarity-evaluator string (use for datasets without ID_Set).",
    )
    p.add_argument("--n", type=int, default=2000, help="Prompts taken from the start of the dataset.")
    p.add_argument("--k", type=int, default=10, help="Candidates per prompt (single-vector top-k).")
    p.add_argument("--splitter-checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-segments", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.dataset).iloc[: args.n].reset_index(drop=True)
    prompts = df["prompt"].astype(str).tolist()
    if args.response_col:
        responses = df[args.response_col].astype(str).tolist()

        def same(i: int, j: int) -> int:
            return int(answers_have_same_meaning_static(responses[i], responses[j]))
    else:
        labels = df[args.label_col].to_numpy()

        def same(i: int, j: int) -> int:
            return int(labels[i] == labels[j])

    embedder = EmbeddingModel(device=args.device)
    splitter = MaxSimSplitter(
        checkpoint_path=args.splitter_checkpoint,
        device=args.device,
        embedding_model=embedder,
        max_segments=args.max_segments,
        include_full_embedding=True,
    )

    tok = splitter.generator.tokenizer
    punct_ids = {i for i in tok.convert_tokens_to_ids(PUNCT_CHARS) if i != tok.unk_token_id}

    variants = ("learned", "rule", "random")
    pooled_knn, pooled_no_cls = [], []
    tensors = {v: [] for v in variants}
    n_segments = {v: [] for v in variants}
    for text in tqdm(prompts, desc="encode+split"):
        enc = splitter.encode_text(text)
        pooled_knn.append(enc["pooled_knn"].float().cpu())
        pooled_no_cls.append(enc["pooled_no_cls"].float().cpu())

        learned = splitter.split_text_return_maxsim_tensor_from_encoded(enc).float().cpu()
        positions = punctuation_positions(enc["input_ids"], enc["length"], punct_ids)
        n_boundaries = min(max(int(learned.shape[0]) - 2, 0), len(positions))
        random_ptrs = sorted(rng.choice(positions, size=n_boundaries, replace=False).tolist()) if n_boundaries else []
        by_variant = {
            "learned": learned,
            "rule": tensor_from_pointers(enc, positions[: args.max_segments]),
            "random": tensor_from_pointers(enc, random_ptrs),
        }
        for v in variants:
            tensors[v].append(by_variant[v])
            n_segments[v].append(int(by_variant[v].shape[0]) - 1)  # last row is the full embedding

    knn = F.normalize(torch.stack(pooled_knn), dim=-1)
    sims = knn @ knn.T
    sims.fill_diagonal_(-2.0)
    k = min(args.k, len(prompts) - 1)
    topk = torch.topk(sims, k=k, dim=1).indices

    maxsim_fn = VerifiedSplitterDecisionPolicy._maxsim_from_tensors
    cos01_fn = VerifiedSplitterDecisionPolicy._cos01

    names = ["vcache"] + [f"{kind}_{v}" for v in variants for kind in ("mvr", "maxsim")]
    y = []
    scores = {name: [] for name in names}
    top1 = {name: 0 for name in names}
    n_with_pos = 0
    for i in tqdm(range(len(prompts)), desc="score pairs"):
        cand = topk[i].tolist()
        row_y = [same(i, j) for j in cand]
        fullcos = [cos01_fn(pooled_no_cls[i], pooled_no_cls[j]) for j in cand]
        rows = {"vcache": [float((sims[i, j] + 1.0) * 0.5) for j in cand]}
        for v in variants:
            ms = [maxsim_fn(tensors[v][i], tensors[v][j]) for j in cand]
            rows[f"maxsim_{v}"] = ms
            rows[f"mvr_{v}"] = [0.5 * (m + c) for m, c in zip(ms, fullcos)]
        y += row_y
        for name in names:
            scores[name] += rows[name]
        if any(row_y):
            n_with_pos += 1
            for name in names:
                top1[name] += row_y[int(np.argmax(rows[name]))]

    y_arr = np.array(y)
    result = {
        "n_prompts": len(prompts),
        "k": k,
        "n_pairs": int(len(y_arr)),
        "positive_pair_rate": round(float(y_arr.mean()), 4),
        "auc": {name: round(float(roc_auc_score(y_arr, scores[name])), 4) for name in names},
        "top1_accuracy": {name: round(v / max(n_with_pos, 1), 4) for name, v in top1.items()},
        "n_prompts_with_positive_candidate": n_with_pos,
        "segments_per_prompt": {
            v: {str(n): c for n, c in sorted(Counter(n_segments[v]).items())} for v in variants
        },
        "mean_segments": {v: round(float(np.mean(n_segments[v])), 3) for v in variants},
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
