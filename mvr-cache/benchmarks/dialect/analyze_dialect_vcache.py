"""Split eval_sembenchmark_verified.py runs on stream_cold / stream_zipf by request kind and variant.

kind (from prepare_dialect_dataset.py)
  first          the meaning has not been requested before: nothing correct to reuse yet
  exact_repeat   the same text was requested before
  cross_variant  the meaning was requested before but in another form (dialect <-> standard):
                 the case a semantic cache exists for

For each (kind, variant):
  hit_rate        vCache reused a cached answer
  error_rate      wrong reuses / rows (vCache's delta bounds this over the whole stream)
  nn_correct_rate the nearest cached entry had the right answer, (tp + fn) / rows: what a perfect
                  reuse decision could reach. The gap to hit_rate is vCache declining to reuse, e.g.
                  an entry needs 6 observations before it can ever be reused

Example:
  python benchmarks/dialect/analyze_dialect_vcache.py \
    --stream /content/drive/MyDrive/dialect/prepared/stream_zipf.parquet \
    --results /content/drive/MyDrive/dialect/results/vcache_zipf_bge-m3.json --out summary.csv
"""

import argparse
import json

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True)
    p.add_argument("--results", nargs="+", required=True, help="--output-json files of eval_sembenchmark_verified.py")
    p.add_argument("--out", default=None, help="Optional CSV path for the table")
    args = p.parse_args()

    stream = pd.read_parquet(args.stream)
    if "kind" not in stream.columns:
        stream["kind"] = "-"
    rows = []
    for path in args.results:
        with open(path) as f:
            res = json.load(f)
        per = pd.DataFrame(res["per_sample"])
        df = stream.head(len(per)).reset_index(drop=True).join(per)
        model = res.get("args", {}).get("embedding_model") or "BAAI/bge-base-en-v1.5 (default)"
        parts = [("ALL", "ALL", df)] + [(k, v, g) for (k, v), g in df.groupby(["kind", "variant"])]
        for kind, variant, g in parts:
            rows.append({
                "results": path.rsplit("/", 1)[-1],
                "embedding_model": model,
                "kind": kind,
                "variant": variant,
                "n": len(g),
                "hit_rate": round(float(g["is_hit"].mean()), 5),
                "error_rate": round(float(g["fp"].sum() / len(g)), 5),
                "nn_correct_rate": round(float((g["tp"] + g["fn"]).sum() / len(g)), 5),
            })
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
