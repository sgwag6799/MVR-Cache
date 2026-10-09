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


def embed(model: EmbeddingModel, texts: list, batch_size: int, device, label: str) -> torch.Tensor:
    """Unit vectors for `texts`, stored as float16 on `device` (2M x 1024 float32 would not fit a T4).

    Encoding itself runs in float32 like vCache; batches are formed by length to waste less padding.
    """
    order = np.argsort([len(t) for t in texts])
    out = torch.empty(len(texts), model.model.config.hidden_size, dtype=torch.float16, device=device)
    t0, step = time.time(), max(1, len(texts) // 10)
    for i in range(0, len(texts), batch_size):
        idx = order[i : i + batch_size]
        v = F.normalize(model.get_embeddings_tensor([texts[j] for j in idx]).float(), dim=-1)
        out[torch.as_tensor(idx, device=device)] = v.to(device=device, dtype=torch.float16)
        if (i // batch_size) % max(1, step // batch_size) == 0:
            print(f"  encode {label}: {min(i + batch_size, len(texts)):,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def score_queries(q: torch.Tensor, idx: torch.Tensor, own: np.ndarray, rand: np.ndarray,
                  chunk: int, block: int) -> dict:
    """Own / best-wrong / top-1 / rank / random cosine per query against the whole cache.

    The cache is walked in `block`-row pieces (cast to float32) and the queries in `chunk`-row pieces,
    so memory stays at about chunk x block floats however large the cache is.
    """
    dev, n_q, n_i = q.device, len(q), len(idx)
    own_t, rand_t = torch.as_tensor(own, device=dev), torch.as_tensor(rand, device=dev)
    own_s = torch.empty(n_q, device=dev)
    rand_s = torch.empty(n_q, device=dev)
    for s in range(0, n_q, chunk):
        qc = q[s : s + chunk].float()
        own_s[s : s + chunk] = (qc * idx[own_t[s : s + chunk]].float()).sum(-1)
        rand_s[s : s + chunk] = (qc * idx[rand_t[s : s + chunk]].float()).sum(-1)
    greater = torch.zeros(n_q, dtype=torch.long, device=dev)
    top1_s = torch.full((n_q,), -9.0, device=dev)
    top1_i = torch.zeros(n_q, dtype=torch.long, device=dev)
    wrong_s = torch.full((n_q,), -9.0, device=dev)
    wrong_i = torch.zeros(n_q, dtype=torch.long, device=dev)
    for b0 in range(0, n_i, block):
        blk = idx[b0 : b0 + block].float()
        for s in range(0, n_q, chunk):
            e = min(s + chunk, n_q)
            sims = q[s:e].float() @ blk.T  # [queries, block]
            m, a = sims.max(dim=1)
            upd = m > top1_s[s:e]
            top1_s[s:e] = torch.where(upd, m, top1_s[s:e])
            top1_i[s:e] = torch.where(upd, a + b0, top1_i[s:e])
            # 정답 문장이 이 구간에 있으면 지운 뒤 "정답보다 높은 오답 수"와 "가장 비슷한 오답"을 센다
            o = own_t[s:e] - b0
            inb = (o >= 0) & (o < blk.shape[0])
            rows = torch.nonzero(inb).squeeze(1)
            sims[rows, o[rows]] = -9.0
            greater[s:e] += (sims > own_s[s:e].unsqueeze(1)).sum(dim=1)
            m, a = sims.max(dim=1)
            upd = m > wrong_s[s:e]
            wrong_s[s:e] = torch.where(upd, m, wrong_s[s:e])
            wrong_i[s:e] = torch.where(upd, a + b0, wrong_i[s:e])
        print(f"  score: cache rows {min(b0 + block, n_i):,}/{n_i:,}", flush=True)
    out = {"own": own_s, "wrong": wrong_s, "wrong_i": wrong_i, "top1": top1_s, "top1_i": top1_i,
           "rank": greater + 1, "rand": rand_s}
    return {k: v.cpu().numpy() for k, v in out.items()}


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
    p.add_argument("--chunk", type=int, default=1024, help="Queries scored at once.")
    p.add_argument("--block", type=int, default=262144, help="Cache rows scored at once (cast to float32).")
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
        idx_vec = embed(model, index["text"].tolist(), args.batch_size, args.device, "cache")
        q_vec = embed(model, pairs["dialect"].tolist(), args.batch_size, args.device, "queries")
        r = score_queries(q_vec, idx_vec, own, rand, args.chunk, args.block)

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
