"""Pack evaluation results into a small shareable bundle.

Result JSONs are ~7.6 MB each because they store one record per sample. This keeps the
summary + args of every run and a downsampled hit-rate curve, which is everything needed
to check the numbers and redraw the Figure 4 style plots.

  python tools/make_share_bundle.py --out ~/Desktop/mvr-cache-results
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="mvr-cache/results")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=500, help="Curve downsampling interval.")
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, "curves"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "weight_stats"), exist_ok=True)

    rows, all_args = [], {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True)):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if "summary" not in data or "per_sample" not in data:
            continue
        name = os.path.splitext(os.path.relpath(path, args.results_dir))[0].replace("/", "__")
        s, a = data["summary"], data.get("args", {})
        n = s["n"]
        rows.append({
            "run": name,
            "dataset": os.path.basename(str(a.get("dataset", ""))),
            "n": n,
            "delta": s.get("delta"),
            "sleep": a.get("sleep"),
            "splitter_mode": a.get("splitter_mode", "-"),
            "segment_weighting": a.get("segment_weighting", "-"),
            "max_segments": a.get("splitter_max_segments", "-"),
            "candidate_selection": a.get("candidate_selection", "-"),
            "candidate_k": a.get("candidate_k", "-"),
            "evaluator": a.get("similarity_evaluator"),
            "hit_rate_pct": round(s["hit_rate"] * 100, 4),
            "tp": s["tp"], "fp": s["fp"], "tn": s["tn"], "fn": s["fn"],
            "error_rate_pct": round(s["fp"] / n * 100, 4),
            "minutes": round(s["total_time"] / 60, 1),
            "checkpoint": os.path.basename(str(a.get("splitter_checkpoint", "") or "")),
        })
        all_args[name] = a

        ps = data["per_sample"]
        with open(os.path.join(args.out, "curves", f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_index", "running_hit_rate_pct"])
            for i in range(args.every - 1, len(ps), args.every):
                w.writerow([ps[i]["sample_index"], round(ps[i]["running_hit_rate"] * 100, 4)])
            if ps:
                w.writerow([ps[-1]["sample_index"], round(ps[-1]["running_hit_rate"] * 100, 4)])

    rows.sort(key=lambda r: (r["dataset"], r["run"]))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(all_args, open(os.path.join(args.out, "args.json"), "w"), indent=1, ensure_ascii=False)

    for pt in glob.glob(os.path.join(args.results_dir, "**", "*.pt"), recursive=True):
        shutil.copy(pt, os.path.join(args.out, "weight_stats", os.path.basename(pt)))

    print(f"{len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
