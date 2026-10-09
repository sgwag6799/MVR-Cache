"""Split an eval_sembenchmark_verified.py run on stream.parquet into standard rows and dialect queries.

For each part:
  hit_rate        vCache reused a cached answer
  error_rate      wrong reuses / rows (vCache's delta is a bound on this)
  nn_correct_rate the nearest cached entry had the right answer (tp + fn) / rows, i.e. what a
                  perfect reuse decision could have reached; the gap to hit_rate is vCache declining
                  to reuse (e.g. every entry needs 6 observations before it can ever be reused)

Example:
  python benchmarks/dialect/analyze_dialect_vcache.py --stream /content/drive/MyDrive/dialect/prepared/stream.parquet \
    --results /content/drive/MyDrive/dialect/vcache_bge-m3.json
"""

import argparse
import json

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True)
    p.add_argument("--results", nargs="+", required=True, help="--output-json files of eval_sembenchmark_verified.py")
    args = p.parse_args()

    stream = pd.read_parquet(args.stream)
    rows = []
    for path in args.results:
        with open(path) as f:
            res = json.load(f)
        per = pd.DataFrame(res["per_sample"])
        df = stream.head(len(per)).reset_index(drop=True).join(per)
        model = res.get("args", {}).get("embedding_model") or "BAAI/bge-base-en-v1.5 (default)"
        for variant, g in df.groupby("variant"):
            rows.append({
                "results": path.rsplit("/", 1)[-1],
                "embedding_model": model,
                "variant": variant,
                "n": len(g),
                "hit_rate": round(float(g["is_hit"].mean()), 5),
                "error_rate": round(float(g["fp"].sum() / len(g)), 5),
                "nn_correct_rate": round(float((g["tp"] + g["fn"]).sum() / len(g)), 5),
            })
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
