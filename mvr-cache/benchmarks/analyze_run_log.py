"""Post-hoc metrics (log F) from one or more --run-log-dir directories, per condition x task.

  hit / error rate        hits / n, false hits / n
  ceiling                 share of prompts with at least one correct candidate
  recall cos/uni/w        share of prompts whose cosine / uniform / weighted #1 is correct
  case3                   ceiling - weighted recall (a correct candidate existed, #1 was wrong)
  selection success       weighted recall / ceiling
  random baseline         mean over prompts of (correct candidates / candidates)
  flag1 / flag2 rate      #1 changed by reranking / by weighting
  fixed / broke / net     top-1 changes wrong->right / right->wrong (rerank and weight separately)
  AUC, d                  separation of the selected neighbour's (s, c)
  delta_t / sigma_t       per query-segment position: best-match cosine, mean(c=1) - mean(c=0),
                          divided by the pooled std (from B6)
  HNSW recall             exact top-K vs HNSW top-K overlap (sampled, diag.jsonl)
  brute-force loss        brute-force MVR #1 correct rate - weighted #1 correct rate (sampled)

Example:
  python benchmarks/analyze_run_log.py runs/cond3_idf_r1 runs/cond2_rule_r1 --out runs/summary.csv

Reading the result (G): low ceiling -> retrieval; high ceiling but large case3 -> rerank /
weights (compare with the random baseline); #1 correct but few hits -> vCache (t_hat, n_obs,
tau); many false hits -> the (s, c) distribution. Always read per task: tasks with few
labels (amazon yes/no) have a high ceiling by chance. Under the no-split condition flag1
should always be 0.
"""
# [새 파일] 실험 로그(requests.jsonl, diag.jsonl)로 조건 × 과제별 사후 지표(F)를 계산한다.

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def _auc(s: np.ndarray, c: np.ndarray):
    pos, neg = s[c == 1], s[c == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # 순위 기반 AUC (동점은 0.5)
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _d(s: np.ndarray, c: np.ndarray):
    pos, neg = s[c == 1], s[c == 0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    sd = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    return float((pos.mean() - neg.mean()) / sd) if sd > 0 else None


def metrics(reqs: list, diags: list) -> dict:
    n = len(reqs)
    if n == 0:
        return {}
    ranked = [r for r in reqs if r["rank"]["n_candidates"] > 0]
    m = len(ranked) or 1
    ceiling = sum(r["rank"]["ceiling"] for r in ranked) / m
    rec = {k: sum(r["rank"]["top1"][k]["c"] for r in ranked) / m for k in ("cos", "uniform", "weighted")}
    change = {k: defaultdict(int) for k in ("rerank_change", "weight_change")}
    for r in ranked:
        for k in change:
            change[k][r["rank"][k]] += 1
    sel = [r["selected"] for r in reqs if r.get("selected") and r["selected"]["c"] is not None]
    s = np.array([x["s"] for x in sel], dtype=float)
    c = np.array([x["c"] for x in sel], dtype=int)

    # 조각 위치별 Δ_t / σ_t (쿼리 조각 t의 최대 유사도가 정답/오답 이웃에서 얼마나 다른지)
    pos_vals = defaultdict(lambda: ([], []))
    for x in sel:
        sm = x.get("segment_match")
        if not sm:
            continue
        rows = sm["query_row_max"][:-1]  # 마지막 행 = 문장 전체
        for t, v in enumerate(rows[:8]):
            pos_vals[t][x["c"]].append(v)
    delta_sigma = {}
    for t, (neg, pos) in sorted(pos_vals.items()):
        if len(pos) > 1 and len(neg) > 1:
            sd = np.sqrt(0.5 * (np.var(pos, ddof=1) + np.var(neg, ddof=1)))
            delta_sigma[f"seg{t}"] = round(float((np.mean(pos) - np.mean(neg)) / sd), 4) if sd > 0 else None

    hnsw = [d["hnsw_recall"] for d in diags if d.get("hnsw_recall") is not None]
    bf = [(d["bruteforce_best"]["c"], d["order"]) for d in diags if d.get("bruteforce_best")]
    w_by_order = {r["order"]: r["rank"]["top1"]["weighted"]["c"] for r in ranked}
    bf_pairs = [(b, w_by_order[o]) for b, o in bf if o in w_by_order]
    return {
        "n": n,
        "hit_rate": round(sum(r["hit"] for r in reqs) / n, 5),
        "error_rate": round(sum(r["false_hit"] for r in reqs) / n, 5),
        "ceiling": round(ceiling, 5),
        "recall_cos": round(rec["cos"], 5),
        "recall_uniform": round(rec["uniform"], 5),
        "recall_weighted": round(rec["weighted"], 5),
        "case3_rate": round(ceiling - rec["weighted"], 5),
        "selection_success": round(rec["weighted"] / ceiling, 5) if ceiling > 0 else None,
        "random_baseline": round(float(np.mean([r["rank"]["n_correct"] / r["rank"]["n_candidates"] for r in ranked])), 5) if ranked else None,
        "flag1_rate": round(sum(r["rank"]["flag_rerank_changed"] for r in ranked) / m, 5),
        "flag2_rate": round(sum(r["rank"]["flag_weight_changed"] for r in ranked) / m, 5),
        "rerank_fixed": change["rerank_change"]["fixed"],
        "rerank_broke": change["rerank_change"]["broke"],
        "rerank_net": change["rerank_change"]["fixed"] - change["rerank_change"]["broke"],
        "weight_fixed": change["weight_change"]["fixed"],
        "weight_broke": change["weight_change"]["broke"],
        "weight_net": change["weight_change"]["fixed"] - change["weight_change"]["broke"],
        "auc_selected": None if len(s) == 0 else _auc(s, c),
        "d_selected": None if len(s) == 0 else _d(s, c),
        "delta_over_sigma_by_segment": delta_sigma,
        "hnsw_recall": round(float(np.mean(hnsw)), 5) if hnsw else None,
        "bruteforce_recall_loss": round(float(np.mean([b - w for b, w in bf_pairs])), 5) if bf_pairs else None,
        "n_diag": len(diags),
    }


def load(run_dir: str):
    with open(os.path.join(run_dir, "run.json")) as f:
        header = json.load(f)
    reqs = [json.loads(l) for l in open(os.path.join(run_dir, "requests.jsonl"))]
    dpath = os.path.join(run_dir, "diag.jsonl")
    diags = [json.loads(l) for l in open(dpath)] if os.path.exists(dpath) else []
    return header, reqs, diags


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default=None, help="CSV path (a .json with the same stem is written too)")
    args = p.parse_args()

    rows = []
    for d in args.run_dirs:
        header, reqs, diags = load(d)
        cond = header.get("condition") or header.get("run_id") or os.path.basename(d.rstrip("/"))
        groups = {"ALL": (reqs, diags)}
        for task in sorted({r["task"] for r in reqs if r.get("task") is not None}):
            groups[task] = ([r for r in reqs if r["task"] == task], [x for x in diags if x.get("task") == task])
        for task, (rq, dg) in groups.items():
            rows.append({"run_dir": d, "condition": cond, "repeat": header.get("repeat"), "task": task, **metrics(rq, dg)})

    df = pd.DataFrame(rows)
    flat = df.drop(columns=["delta_over_sigma_by_segment"])
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(flat.to_string(index=False))
    if args.out:
        flat.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
