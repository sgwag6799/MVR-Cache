"""Experiment 3: fit the answer-agreement judge and fix p* from the training stream (no test data).

Input : decision logs (DECISION_LOG jsonl from scripts/judge/run_eval.py) of the baseline C run with
        --collect-consensus on the train+val stream. One log per collection seed.
Model : logistic regression on standardized [s, a_excl, candidate count] -> P(top-1 answer is right),
        rows with a top-1 candidate, a known label, a_excl defined and >= 5 candidates (as the team judge).
p*    : additive error budget (team policy learned_*_add): among requests the baseline did NOT reuse
        (scope all: cold + explore, scope cold: cold only), accept in order of out-of-fold P while
        accepted wrong answers / all requests of the stream <= budget b. p* = the lowest P accepted.
        With several logs, each log gives one p*; the saved p* is their median (team rule Jmed) and
        the saved model is fit on all logs together.
Output: JSON read by eval_sembenchmark_verified_splitter.py --consensus-model.

Usage: python scripts/judge/judge_fit.py --logs collect_s42.jsonl [collect_s43.jsonl ...] \
         --budget 0.0035 --scope all --out data/judge_cls/judge/shared_L8.json
"""
# [새 파일] 실험 3: 판정자 학습 · p* 고정 (test 미사용). 팀 판정자(learned_all_add35)와 같은 특징 · 같은 예산 규칙

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

FEATURES = ["similarity", "agreement", "candidate_count"]
MIN_CANDIDATES = 5


def load(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def usable(r: dict) -> bool:
    return (r.get("has_nn") and r.get("nn_correct") is not None and r.get("s") is not None
            and r.get("a_excl") is not None and int(r.get("candidate_count") or 0) >= MIN_CANDIDATES)


def xy(rows: list):
    x = np.array([[float(r["s"]), float(r["a_excl"]), float(r["candidate_count"])] for r in rows])
    y = np.array([int(bool(r["nn_correct"])) for r in rows])
    return x, y


def oof_prob(x, y, seed=0):
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        sc = StandardScaler().fit(x[tr])
        m = LogisticRegression(max_iter=1000).fit(sc.transform(x[tr]), y[tr])
        p[va] = m.predict_proba(sc.transform(x[va]))[:, 1]
    return p


def pick_pstar(records, rows, p, budget, scope):
    """Lowest P such that accepted wrong / all requests <= budget among baseline misses in scope."""
    reasons = {"cold"} if scope == "cold" else {"cold", "explore"}
    miss = [(pi, r) for pi, r in zip(p, rows) if not r.get("hit") and r.get("reason") in reasons]
    n = len(records)
    best = {"pstar": None, "accepted": 0, "wrong": 0}
    for t in sorted({pi for pi, _ in miss}, reverse=True):
        acc = [r for pi, r in miss if pi >= t]
        wrong = sum(1 for r in acc if not r["nn_correct"])
        if wrong / n > budget:
            break
        best = {"pstar": float(t), "accepted": len(acc), "wrong": wrong}
    best.update({"requests": n, "baseline_miss_candidates": len(miss),
                 "extra_error_rate": best["wrong"] / n if n else None})
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--budget", type=float, default=0.0035, help="extra wrong reuses / requests (add35 = 0.0035)")
    ap.add_argument("--scope", choices=["all", "cold"], default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_log, pooled = [], []
    for path in args.logs:
        recs = load(path)
        rows = [r for r in recs if usable(r)]
        x, y = xy(rows)
        p = oof_prob(x, y)
        sel = pick_pstar(recs, rows, p, args.budget, args.scope)
        sel.update({"log": os.path.basename(path), "rows": len(rows), "oof_auc": float(roc_auc_score(y, p)),
                    "correct_rate": float(y.mean())})
        per_log.append(sel)
        pooled += rows
        print(json.dumps(sel, ensure_ascii=False))

    pstars = [s["pstar"] for s in per_log if s["pstar"] is not None]
    if not pstars:
        raise SystemExit("no p* satisfies the budget in any log")
    x, y = xy(pooled)
    sc = StandardScaler().fit(x)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(x), y)
    model = {"features": FEATURES, "coefficients": [float(c) for c in m.coef_[0]], "intercept": float(m.intercept_[0]),
             "feature_means": [float(v) for v in sc.mean_], "feature_scales": [float(v) for v in sc.scale_],
             "min_probability": float(np.median(pstars)), "pstar_rule": "median over logs" if len(pstars) > 1 else "single log",
             "budget": args.budget, "scope": args.scope, "min_candidates": MIN_CANDIDATES, "per_log": per_log}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1, ensure_ascii=False)
    print(f"saved {args.out}: p* = {model['min_probability']:.6f} ({model['pstar_rule']})")


if __name__ == "__main__":
    main()
