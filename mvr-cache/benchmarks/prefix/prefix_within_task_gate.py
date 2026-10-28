"""Experiment 1: the stage 1 gate measured within each sub-task (no GPU, saved scores only).

The original gate pools all test anchors into one auc_selected. The three sub-tasks have very
different score ranges and answer rates (ecommerce: 4 answers, top-1 correct ~95%; amazon: 2
answers; commonsense: answers depend on the question), so a pooled AUC also rewards a score that
merely ranks ecommerce above the other tasks. A prefix that shrinks those between-task gaps then
looks like a loss of discrimination even when nothing changes inside a task.

This script recomputes the gate from the per-pair scores saved by prefix_offline.py --save-scores
(scores_<mode>_L<L>.npz, same anchors at every level):

  within-task auc_selected   AUC of (s, c) of each anchor's #1 candidate, inside one sub-task
  macro                      mean of the three within-task values (each task counts once)  <- gate
  ecommerce                  the one task where L0 is clearly above chance (reported, not gated)
  pooled                     the original metric (should match prefix_gate.json)

Gate (same thresholds as the original): at some L >= 4, R - M2 >= 0.03 and
(M3 - M2) / (R - M2) >= 0.5, with R = M2 at L0. Paired bootstrap over anchors, stratified by
task, gives 95% intervals for the drop, M3 - M2, M4 - M2 and the recovery.

The macro gate was fixed after the per-task numbers of the pool x label runs had been looked
at (see PREFIX_EXPERIMENT.md 11, 12), so it is exploratory, not a replacement for section 8.

Example:
  python -m benchmarks.prefix.prefix_within_task_gate --out results_within_task \
    prefix:8b=data/prefix/offline_8b_scores prefix:70b=results_labels_tasks/prefix/offline_70b ...
"""
# [새 파일] 실험 1: 과제 안에서 잰 통과 판정 (저장된 쌍별 점수만 사용, GPU 불필요)

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

TASKS = ("ecommerce_dataset", "amazon_instant_video", "commonsense_qa")
CONDS = ("M1", "M2", "M3", "M4", "M5", "M6")


def auc(s, c):
    pos, neg = c == 1, c == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selected(scores, c):
    """(s, c) of each anchor's #1 candidate."""
    top = scores.argmax(1)
    i = np.arange(len(top))
    return scores[i, top], c[i, top]


def summarise(sel, tasks, idx):
    """sel: {cond: (s, c)} over all anchors -> {cond: {task|macro|pooled: auc}} on anchor rows idx."""
    out = {}
    for cond, (s, c) in sel.items():
        s, c, t = s[idx], c[idx], tasks[idx]
        per = {task: auc(s[t == task], c[t == task]) for task in TASKS}
        out[cond] = {**per, "macro": float(np.mean([per[x] for x in TASKS])), "pooled": auc(s, c)}
    return out


def gate_row(r0, rl, key):
    R, M2, M3, M4 = r0["M2"][key], rl["M2"][key], rl["M3"][key], rl["M4"][key]
    drop = R - M2
    return {"R": R, "M2": M2, "M3": M3, "M4": M4, "drop": drop,
            "recovery": (M3 - M2) / drop if drop > 0 else float("nan"),
            "m3_minus_m2": M3 - M2, "m4_minus_m2": M4 - M2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="NAME=DIR with scores_variant_L*.npz, e.g. prefix:8b=data/prefix/offline_8b_scores")
    p.add_argument("--out", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--gate-levels", type=int, nargs="+", default=[4, 8])
    p.add_argument("--gate-drop", type=float, default=0.03)
    p.add_argument("--gate-recovery", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, gates = [], {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        z0 = np.load(os.path.join(d, "scores_variant_L0.npz"), allow_pickle=True)
        tasks = z0["tasks"].astype(str)
        conds = [c for c in CONDS if c in z0.files]
        sel0 = {c: selected(z0[c], z0["c"]) for c in conds}
        everyone = np.arange(len(tasks))
        r0 = summarise(sel0, tasks, everyone)
        by_task = [np.where(tasks == t)[0] for t in TASKS]
        rng = np.random.default_rng(args.seed)
        boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in by_task]) for _ in range(args.bootstrap)]
        b0 = [summarise(sel0, tasks, b) for b in boots]
        gates[name] = {}
        for L in args.levels:
            zl = np.load(os.path.join(d, f"scores_variant_L{L}.npz"), allow_pickle=True)
            assert (zl["anchors"] == z0["anchors"]).all(), f"{name} L{L}: anchors differ from L0"
            sel = {c: selected(zl[c], zl["c"]) for c in conds}
            rl = summarise(sel, tasks, everyone)
            bl = [summarise(sel, tasks, b) for b in boots]
            for key in (*TASKS, "macro", "pooled"):
                g = gate_row(r0, rl, key)
                bs = [gate_row(a, b, key) for a, b in zip(b0, bl)]
                for stat in ("drop", "m3_minus_m2", "m4_minus_m2", "recovery"):
                    v = np.array([x[stat] for x in bs], dtype=float)
                    g[stat + "_ci95"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                for c in conds:
                    g[c] = rl[c][key]
                rows.append({"run": name, "level": L, "scope": key, **{k: v for k, v in g.items() if not k.endswith("ci95")}})
                gates[name].setdefault(f"L{L}", {})[key] = g
        for key in ("macro", "ecommerce_dataset", "pooled"):
            passed = [L for L in args.gate_levels
                      if gates[name][f"L{L}"][key]["drop"] >= args.gate_drop
                      and gates[name][f"L{L}"][key]["recovery"] >= args.gate_recovery]
            gates[name].setdefault("pass", {})[key] = {"levels": passed, "stage2_level": max(passed) if passed else None}
        g8 = gates[name][f"L{max(args.gate_levels)}"]
        print(f"[{name}] L{max(args.gate_levels)}  "
              + "  ".join(f"{k[:9]}: drop={g8[k]['drop']:+.3f} rec={g8[k]['recovery']:.2f}" for k in ("pooled", "macro", "ecommerce_dataset"))
              + f"  pass(macro)={gates[name]['pass']['macro']['levels']}")

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "within_task_gate.csv"), index=False)
    meta = {"args": vars(args), "gate_metric": "macro mean of within-task auc_selected", "results": gates}
    with open(os.path.join(args.out, "within_task_gate.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
