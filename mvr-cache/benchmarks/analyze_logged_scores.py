"""Analyse the (s, c) actually seen by vCache in logged runs (results/obs).

For each weighting it reads scores_w<w>.jsonl (one line per request: s, c, action, t_hat,
gamma, n_obs at decision time) and obs_w<w>.json (every (s, c) the per-entry logistic fits
used) and reports:
  - per-class normality of s (mean/std/skew/kurtosis/Shapiro, std ratio), AUC
  - global logistic fit vs binned empirical P(c=1|s)
  - per-entry statistics: observations per entry, perfectly separable entries, t_hat/gamma
  - where hits happen (s at exploit) and their precision
and saves two figures (distribution, calibration).

Example:
  python benchmarks/analyze_logged_scores.py --dir results/obs --out results/obs/analysis
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

WEIGHTINGS = ["uniform", "idf", "length", "centroid", "mlp"]


def load(dir_, w):
    rows = [json.loads(l) for l in open(f"{dir_}/scores_w{w}.jsonl")]
    rows = [r for r in rows if "s" in r]  # first request has no neighbour
    obs = json.load(open(f"{dir_}/obs_w{w}.json"))["observations"]
    return rows, np.array(obs, dtype=float)


def class_stats(x):
    sub = np.random.default_rng(0).choice(x, size=min(len(x), 4999), replace=False)
    return {
        "n": int(len(x)),
        "mean": round(float(x.mean()), 4),
        "std": round(float(x.std()), 4),
        "skew": round(float(stats.skew(x)), 3),
        "kurtosis": round(float(stats.kurtosis(x)), 3),
        "shapiro_p": float(f"{stats.shapiro(sub).pvalue:.3g}"),
        "p05/p50/p95": [round(float(v), 4) for v in np.percentile(x, [5, 50, 95])],
    }


def analyse(rows, obs):
    s = np.array([r["s"] for r in rows])
    c = np.array([r["c"] for r in rows])
    act = np.array([r["action"] for r in rows])
    ex = act == "explore"
    out = {"requests": len(rows), "explore": int(ex.sum()), "exploit": int((~ex).sum())}

    # (s, c) fed to the fits = explore requests (cache-on-miss)
    se, ce = s[ex], c[ex]
    out["auc_explore"] = round(roc_auc_score(ce, se), 4)
    out["c=1"] = class_stats(se[ce == 1])
    out["c=0"] = class_stats(se[ce == 0])
    out["std_ratio(c1/c0)"] = round(out["c=1"]["std"] / out["c=0"]["std"], 3)
    out["d_prime"] = round(
        (out["c=1"]["mean"] - out["c=0"]["mean"])
        / np.sqrt(0.5 * (out["c=1"]["std"] ** 2 + out["c=0"]["std"] ** 2)),
        3,
    )
    out["frac_s>=0.999 (c=1, c=0)"] = [
        round(float((se[ce == 1] >= 0.999).mean()), 4),
        round(float((se[ce == 0] >= 0.999).mean()), 4),
    ]

    # global logistic vs binned empirical
    lr = LogisticRegression(C=1e6).fit(se.reshape(-1, 1), ce)
    gamma, t = float(lr.coef_[0, 0]), float(-lr.intercept_[0] / lr.coef_[0, 0])
    edges = np.quantile(se, np.linspace(0, 1, 21))
    idx = np.clip(np.searchsorted(edges, se, side="right") - 1, 0, 19)
    bins = []
    for b in range(20):
        m = idx == b
        if m.sum() == 0:
            continue
        sm = float(se[m].mean())
        bins.append((sm, float(ce[m].mean()), float(lr.predict_proba([[sm]])[0, 1]), int(m.sum())))
    dev = np.array([abs(e - p) for _, e, p, _ in bins])
    out["global_logistic"] = {
        "gamma": round(gamma, 2),
        "t": round(t, 4),
        "mean_abs_dev_pp": round(100 * dev.mean(), 2),
        "max_abs_dev_pp": round(100 * dev.max(), 2),
    }

    # per-entry fits (from the dumped observations)
    per = defaultdict(list)
    for e, si, ci in obs:
        per[int(e)].append((si, int(ci)))
    nobs = np.array([len(v) for v in per.values()])
    both = sep = 0
    for v in per.values():
        a = np.array(v)
        s1, s0 = a[a[:, 1] == 1, 0], a[a[:, 1] == 0, 0]
        if len(s1) and len(s0):
            both += 1
            sep += s0.max() < s1.min()
    out["entries"] = {
        "n_entries": len(per),
        "obs_per_entry p50/p90/max": [int(np.median(nobs)), int(np.percentile(nobs, 90)), int(nobs.max())],
        "entries_with_1_obs": int((nobs == 1).sum()),
        "entries_with_both_classes": both,
        "perfectly_separable_(of_both)": sep,
    }

    # thresholds at decision time (requests with a fitted entry)
    tg = [(r["t_hat"], r["gamma"]) for r in rows if r.get("t_hat") is not None]
    if tg:
        th, gm = np.array(tg).T
        out["t_hat at decisions p10/p50/p90"] = [round(float(v), 3) for v in np.percentile(th, [10, 50, 90])]
        out["gamma at decisions p10/p50/p90"] = [round(float(v), 1) for v in np.percentile(gm, [10, 50, 90])]
        out["frac_decisions_gamma>=100"] = round(float((gm >= 100).mean()), 3)

    # hits
    sh, ch = s[~ex], c[~ex]
    out["hits"] = {
        "n": int(len(sh)),
        "precision": round(float(ch.mean()), 4) if len(sh) else None,
        "s p05/p50/p95": [round(float(v), 4) for v in np.percentile(sh, [5, 50, 95])] if len(sh) else None,
        "frac_s>=0.999": round(float((sh >= 0.999).mean()), 4) if len(sh) else None,
    }
    out["_plot"] = {"se": se, "ce": ce, "bins": bins, "gamma": gamma, "t": t}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/obs")
    p.add_argument("--out", default="results/obs/analysis")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    res = {}
    for w in WEIGHTINGS:
        if os.path.exists(f"{args.dir}/obs_w{w}.json"):
            res[w] = analyse(*load(args.dir, w))
    ws = list(res)

    fig, axes = plt.subplots(1, len(ws), figsize=(4 * len(ws), 3.4), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), ws):
        d = res[w]["_plot"]
        b = np.linspace(0.6, 1.0, 81)
        for cl, col in ((1, "tab:green"), (0, "tab:red")):
            x = d["se"][d["ce"] == cl]
            ax.hist(x, bins=b, density=True, alpha=0.45, color=col, label=f"c={cl}")
            xx = np.linspace(0.6, 1.0, 300)
            ax.plot(xx, stats.norm.pdf(xx, x.mean(), x.std()), color=col, lw=1)
        ax.set_title(f"{w}  AUC {res[w]['auc_explore']}")
        ax.set_xlabel("s (explore requests)")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}/sc_distribution_real.png", dpi=130)

    fig, axes = plt.subplots(1, len(ws), figsize=(4 * len(ws), 3.4), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), ws):
        d = res[w]["_plot"]
        bx = np.array(d["bins"])
        ax.plot(bx[:, 0], bx[:, 1], "o", ms=4, label="empirical P(c=1|s)")
        xx = np.linspace(bx[:, 0].min(), 1.0, 200)
        ax.plot(xx, 1 / (1 + np.exp(-d["gamma"] * (xx - d["t"]))), label="global logistic")
        ax.set_title(f"{w}  dev {res[w]['global_logistic']['mean_abs_dev_pp']}pp")
        ax.set_xlabel("s (20 quantile bins)")
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/sc_calibration_real.png", dpi=130)

    for w in ws:
        res[w].pop("_plot")
    json.dump(res, open(f"{args.out}/summary.json", "w"), indent=1, default=int)
    print(json.dumps(res, indent=1, default=int))


if __name__ == "__main__":
    main()
