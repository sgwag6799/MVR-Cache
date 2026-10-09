"""Dialect -> standard retrieval: does a semantic cache find the same-meaning cached sentence?

The "cache" is every unique standard-Korean sentence of all splits (index.parquet); each query is
the Gyeongsang-dialect form of one of them (pairs.parquet). The other cached sentences come from
the same kind of casual conversation and act as hard distractors. Sentence vectors use the same
masked mean pooling as vCache (EmbeddingModel), compared by exact cosine (no ANN).

Per encoder and per query split (train / val / all) it reports
  top1_acc / top5_acc / mrr      is the query's own standard sentence the nearest / in the top 5
  auc_hard                       own-sentence cosine vs the best wrong sentence's cosine
  auc_random                     own-sentence cosine vs a random cached sentence's cosine
  static_threshold               one global cosine threshold as the cache rule: reuse the nearest
                                 sentence when cos >= t. For each target error rate e (wrong reuses /
                                 queries, like vCache's delta) the largest hit rate reachable when t
                                 is tuned on the same queries (optimistic)
  by_dialect_ratio, by_surface_sim   top1_acc split by how much of the utterance is dialect /
                                 how different the two forms are character-wise
and, when both splits exist, `calibrated`: t picked on train queries for each e, applied to val
queries (the honest estimate of what a deployed threshold would do).

Example:
  python benchmarks/dialect/eval_dialect_retrieval.py --data-dir /content/drive/MyDrive/dialect/prepared \
    --models BAAI/bge-base-en-v1.5 BAAI/bge-m3 --device cuda --out /content/drive/MyDrive/dialect/results/retrieval.json
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


def embed(model: EmbeddingModel, texts: list, batch_size: int, device) -> torch.Tensor:
    """Unit vectors for `texts`; batches are formed by length to waste less padding."""
    order = np.argsort([len(t) for t in texts])
    out = torch.empty(len(texts), model.model.config.hidden_size)
    for i in range(0, len(texts), batch_size):
        idx = order[i : i + batch_size]
        out[torch.as_tensor(idx)] = model.get_embeddings_tensor([texts[j] for j in idx]).float().cpu()
    return F.normalize(out, dim=-1).to(device)


def score_queries(q: torch.Tensor, idx: torch.Tensor, own: np.ndarray, rand: np.ndarray, chunk: int) -> dict:
    """Own / best-wrong / top-1 / rank / random cosine per query, `chunk` queries at a time."""
    res = {k: [] for k in ("own", "wrong", "wrong_i", "top1", "top1_i", "rank", "rand")}
    for s in range(0, len(q), chunk):
        sims = q[s : s + chunk] @ idx.T  # [chunk, index]
        rows = torch.arange(sims.shape[0], device=sims.device)
        o = torch.as_tensor(own[s : s + chunk], device=sims.device)
        own_s = sims[rows, o]
        top1_s, top1_i = sims.max(dim=1)
        rank = (sims > own_s.unsqueeze(1)).sum(dim=1) + 1
        rand_s = sims[rows, torch.as_tensor(rand[s : s + chunk], device=sims.device)]
        sims[rows, o] = -2.0
        wrong_s, wrong_i = sims.max(dim=1)
        for k, v in (("own", own_s), ("wrong", wrong_s), ("wrong_i", wrong_i), ("top1", top1_s),
                     ("top1_i", top1_i), ("rank", rank), ("rand", rand_s)):
            res[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in res.items()}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5 * P(pos == neg)."""
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_threshold(top1: np.ndarray, correct: np.ndarray, e: float):
    """Lowest threshold t (= most hits) with wrong reuses / queries <= e, reusing when cos >= t."""
    order = np.argsort(-top1)
    s, c = top1[order], correct[order]
    wrong = np.cumsum(~c) / len(s)
    # 동점 점수는 임계값 하나로 함께 히트하므로, 같은 점수 묶음의 끝에서만 자를 수 있다
    ends = np.r_[np.where(s[1:] != s[:-1])[0], len(s) - 1]
    ok = ends[wrong[ends] <= e]
    return None if len(ok) == 0 else float(s[ok[-1]])


def apply_threshold(top1: np.ndarray, correct: np.ndarray, t) -> dict:
    if t is None:
        return {"threshold": None, "hit_rate": 0.0, "error_rate": 0.0}
    hit = top1 >= t
    return {"threshold": round(t, 5), "hit_rate": round(float(hit.mean()), 5),
            "error_rate": round(float((hit & ~correct).mean()), 5)}


def by_bins(df: pd.DataFrame, col: str, edges: list) -> dict:
    labels = [f"[{a}, {b})" for a, b in zip(edges[:-1], edges[1:])]
    g = df.groupby(pd.cut(df[col], bins=edges, labels=labels, right=False), observed=True)["top1_correct"]
    return {str(k): {"n": int(v.size), "top1_acc": round(float(v.mean()), 4)} for k, v in g}


def metrics(df: pd.DataFrame, targets: list) -> dict:
    top1, correct = df["top1_cos"].to_numpy(), df["top1_correct"].to_numpy()
    return {
        "n_queries": len(df),
        "top1_acc": round(float(correct.mean()), 4),
        "top5_acc": round(float((df["rank"] <= 5).mean()), 4),
        "mrr": round(float((1.0 / df["rank"]).mean()), 4),
        "own_cos_mean": round(float(df["own_cos"].mean()), 4),
        "best_wrong_cos_mean": round(float(df["best_wrong_cos"].mean()), 4),
        "margin_mean": round(float((df["own_cos"] - df["best_wrong_cos"]).mean()), 4),
        "auc_hard": round(auc(df["own_cos"].to_numpy(), df["best_wrong_cos"].to_numpy()), 4),
        "auc_random": round(auc(df["own_cos"].to_numpy(), df["random_cos"].to_numpy()), 4),
        "static_threshold": {str(e): apply_threshold(top1, correct, best_threshold(top1, correct, e)) for e in targets},
        "by_dialect_ratio": by_bins(df, "dialect_ratio", [0, 0.1, 0.2, 1.01]),
        "by_surface_sim": by_bins(df, "surface_sim", [0, 0.9, 0.95, 1.01]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Output folder of prepare_dialect_dataset.py")
    p.add_argument("--models", nargs="+", default=["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512, help="Queries scored against the whole cache at once.")
    p.add_argument("--target-errors", nargs="+", type=float, default=[0.01, 0.05])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-queries", type=int, default=None, help="Use only the first N queries (quick check).")
    p.add_argument("--out", required=True, help="Summary JSON; per-query CSVs are written next to it.")
    args = p.parse_args()

    index = pd.read_parquet(os.path.join(args.data_dir, "index.parquet"))
    pairs = pd.read_parquet(os.path.join(args.data_dir, "pairs.parquet"))
    if "split" not in pairs.columns:
        pairs["split"] = "all"
    if args.max_queries:
        pairs = pairs.head(args.max_queries).reset_index(drop=True)
    pos_of_group = {g: i for i, g in enumerate(index["group_id"])}
    own = np.array([pos_of_group[g] for g in pairs["group_id"]])
    rng = np.random.default_rng(args.seed)
    rand = rng.integers(0, len(index), size=len(pairs))
    rand = np.where(rand == own, (rand + 1) % len(index), rand)  # 무작위 비교 대상이 정답과 겹치지 않게
    splits = [s for s in ("train", "val") if (pairs["split"] == s).any()]

    summary = {"data_dir": args.data_dir, "n_index": len(index),
               "n_queries": pairs["split"].value_counts().to_dict(), "models": {}}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    for name in args.models:
        t0 = time.time()
        model = EmbeddingModel(model_name=name, device=args.device)
        idx_vec = embed(model, index["text"].tolist(), args.batch_size, args.device)
        q_vec = embed(model, pairs["dialect"].tolist(), args.batch_size, args.device)
        r = score_queries(q_vec, idx_vec, own, rand, args.chunk)

        df = pairs.copy()
        df["own_cos"], df["best_wrong_cos"], df["random_cos"] = r["own"], r["wrong"], r["rand"]
        df["best_wrong_text"] = index["text"].to_numpy()[r["wrong_i"]]
        df["top1_cos"], df["rank"] = r["top1"], r["rank"]
        df["top1_correct"] = r["top1_i"] == own
        df.to_csv(os.path.join(out_dir, f"retrieval_{name.replace('/', '__')}.csv"), index=False)

        res = {"dim": int(idx_vec.shape[1]), "seconds": round(time.time() - t0, 1),
               "all": metrics(df, args.target_errors)}
        for s in splits:
            res[s] = metrics(df[df["split"] == s], args.target_errors)
        if "train" in splits and "val" in splits:
            tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
            res["calibrated"] = {}
            for e in args.target_errors:
                t = best_threshold(tr["top1_cos"].to_numpy(), tr["top1_correct"].to_numpy(), e)
                res["calibrated"][str(e)] = apply_threshold(va["top1_cos"].to_numpy(), va["top1_correct"].to_numpy(), t)
        summary["models"][name] = res
        show = {k: v for k, v in res.items() if k in ("dim", "seconds", "calibrated")}
        for s in ["all"] + splits:
            show[s] = {k: res[s][k] for k in ("n_queries", "top1_acc", "top5_acc", "auc_hard", "margin_mean", "static_threshold")}
        print(json.dumps({name: show}, ensure_ascii=False, indent=1), flush=True)
        del model, idx_vec, q_vec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
