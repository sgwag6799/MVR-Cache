"""Dialect -> standard retrieval: does a semantic cache find the same-meaning cached sentence?

The "cache" is every unique standard-Korean sentence (index.parquet); each query is the
Gyeongsang-dialect form of one of them (pairs.parquet). The other ~4,400 cached sentences come
from the same casual conversations and act as hard distractors. Sentence vectors use the same
masked mean pooling as vCache (EmbeddingModel), compared by cosine.

Per encoder it reports
  top1_acc / top5_acc / mrr      is the query's own standard sentence the nearest / in the top 5
  auc_hard                       own-sentence cosine vs the best wrong sentence's cosine
  auc_random                     own-sentence cosine vs a random cached sentence's cosine
  static_threshold               one global cosine threshold as the cache rule: reuse the nearest
                                 sentence when cos >= t. For each target error rate e (wrong reuses
                                 / queries, like vCache's delta) the largest hit rate reachable
  by_dialect_ratio, by_surface_sim   top1_acc split by how much of the utterance is dialect /
                                 how different the two forms are character-wise

Example:
  python benchmarks/dialect/eval_dialect_retrieval.py --data-dir /content/drive/MyDrive/dialect/prepared \
    --models BAAI/bge-base-en-v1.5 BAAI/bge-m3 --device cuda --out /content/drive/MyDrive/dialect/retrieval.json
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from vcache.vcache_core.splitter.embedding_model import EmbeddingModel


def embed(model: EmbeddingModel, texts: list, batch_size: int) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(model.get_embeddings_tensor(texts[i : i + batch_size]).float().cpu())
    return F.normalize(torch.cat(out), dim=-1)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5 * P(pos == neg)."""
    scores = np.concatenate([pos, neg])
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def static_threshold(top1_score: np.ndarray, top1_correct: np.ndarray, targets: list) -> dict:
    """Reuse the nearest cached sentence when its cosine >= t. Error = wrong reuses / all queries."""
    order = np.argsort(-top1_score)
    s, c = top1_score[order], top1_correct[order]
    n = len(s)
    hits = np.arange(1, n + 1)
    wrong = np.cumsum(~c)
    out = {}
    for e in targets:
        ok = np.where(wrong / n <= e)[0]
        if len(ok) == 0:
            out[str(e)] = {"hit_rate": 0.0, "error_rate": 0.0, "threshold": None}
            continue
        i = ok[-1]
        # 같은 점수가 이어지면 임계값을 그 점수로 둘 때 함께 히트하므로, 동점 끝까지 포함되는지 확인
        while i + 1 < n and s[i + 1] == s[i] and wrong[i + 1] / n <= e:
            i += 1
        out[str(e)] = {"hit_rate": round(hits[i] / n, 5), "error_rate": round(wrong[i] / n, 5),
                       "threshold": round(float(s[i]), 5)}
    return out


def by_bins(df: pd.DataFrame, col: str, edges: list) -> dict:
    labels = [f"[{a}, {b})" for a, b in zip(edges[:-1], edges[1:])]
    cut = pd.cut(df[col], bins=edges, labels=labels, right=False)
    g = df.groupby(cut, observed=True)["top1_correct"]
    return {str(k): {"n": int(v.size), "top1_acc": round(float(v.mean()), 4)} for k, v in g}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Output folder of prepare_dialect_dataset.py")
    p.add_argument("--models", nargs="+", default=["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-errors", nargs="+", type=float, default=[0.01, 0.05])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-queries", type=int, default=None, help="Use only the first N queries (quick check).")
    p.add_argument("--out", required=True, help="Summary JSON; per-query CSVs are written next to it.")
    args = p.parse_args()

    index = pd.read_parquet(os.path.join(args.data_dir, "index.parquet"))
    pairs = pd.read_parquet(os.path.join(args.data_dir, "pairs.parquet"))
    if args.max_queries:
        pairs = pairs.head(args.max_queries).reset_index(drop=True)
    pos_of_group = {g: i for i, g in enumerate(index["group_id"])}
    own = np.array([pos_of_group[g] for g in pairs["group_id"]])
    rng = np.random.default_rng(args.seed)
    rand = rng.integers(0, len(index), size=len(pairs))
    rand = np.where(rand == own, (rand + 1) % len(index), rand)  # 무작위 오답이 정답과 겹치지 않게

    summary = {"data_dir": args.data_dir, "n_index": len(index), "n_queries": len(pairs), "models": {}}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    for name in args.models:
        t0 = time.time()
        model = EmbeddingModel(model_name=name, device=args.device)
        idx_vec = embed(model, index["text"].tolist(), args.batch_size)
        q_vec = embed(model, pairs["dialect"].tolist(), args.batch_size)
        sims = q_vec @ idx_vec.T  # [queries, index]
        own_s = sims[torch.arange(len(own)), torch.as_tensor(own)].numpy()
        masked = sims.clone()
        masked[torch.arange(len(own)), torch.as_tensor(own)] = -2.0
        wrong_s, wrong_i = masked.max(dim=1)
        top1_s, top1_i = sims.max(dim=1)
        rank = (sims > torch.as_tensor(own_s).unsqueeze(1)).sum(dim=1).numpy() + 1

        df = pairs.copy()
        df["own_cos"] = own_s
        df["best_wrong_cos"] = wrong_s.numpy()
        df["best_wrong_text"] = index["text"].to_numpy()[wrong_i.numpy()]
        df["top1_cos"] = top1_s.numpy()
        df["rank"] = rank
        df["top1_correct"] = top1_i.numpy() == own
        tag = name.replace("/", "__")
        df.to_csv(os.path.join(out_dir, f"retrieval_{tag}.csv"), index=False)

        rand_s = sims[torch.arange(len(own)), torch.as_tensor(rand)].numpy()
        summary["models"][name] = {
            "dim": int(idx_vec.shape[1]),
            "seconds": round(time.time() - t0, 1),
            "top1_acc": round(float(df["top1_correct"].mean()), 4),
            "top5_acc": round(float((df["rank"] <= 5).mean()), 4),
            "mrr": round(float((1.0 / df["rank"]).mean()), 4),
            "own_cos_mean": round(float(own_s.mean()), 4),
            "best_wrong_cos_mean": round(float(wrong_s.mean()), 4),
            "margin_mean": round(float((own_s - wrong_s.numpy()).mean()), 4),
            "auc_hard": round(auc(own_s, wrong_s.numpy()), 4),
            "auc_random": round(auc(own_s, rand_s), 4),
            "static_threshold": static_threshold(df["top1_cos"].to_numpy(), df["top1_correct"].to_numpy(),
                                                 args.target_errors),
            "by_dialect_ratio": by_bins(df, "dialect_ratio", [0, 0.1, 0.2, 1.01]),
            "by_surface_sim": by_bins(df, "surface_sim", [0, 0.9, 0.95, 1.01]),
        }
        print(json.dumps({name: summary["models"][name]}, ensure_ascii=False, indent=1), flush=True)
        del model, sims, masked
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
