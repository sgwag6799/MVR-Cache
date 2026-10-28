"""Experiment 7: per-request embedding latency of encoders (base vs LoRA-merged), p50 / p95.

Encodes the same prompts one at a time (batch 1, as a cache lookup does) after a warm-up, with
CUDA synchronisation around each call. A merged LoRA model has the base architecture, so the
expected difference is noise; this checks it.

Example:
  python -m benchmarks.prefix.prefix_encoder_latency --data ../data/lmarena/prefix/data/variant/L8/test.parquet \
    --out ../data/exp7/latency.json base=BAAI/bge-base-en-v1.5 lora=../data/exp7/lora/model
"""
# [새 파일] 실험 7: 인코더별 요청당 임베딩 지연시간 (p50 / p95)

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="NAME=PATH_OR_HF_ID")
    p.add_argument("--data", required=True, help="Parquet with a 'prompt' column")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer

    prompts = pd.read_parquet(args.data)["prompt"].astype(str).tolist()[: args.n + args.warmup]
    res = {"n": args.n, "data": args.data,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "models": {}}
    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        lm = AutoModel.from_pretrained(path).to(args.device).eval()
        times = []
        with torch.no_grad():
            for i, t in enumerate(prompts):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                e = tok(t, return_tensors="pt", truncation=True, max_length=512).to(args.device)
                with torch.autocast("cuda", dtype=torch.float16):
                    h = lm(**e).last_hidden_state
                h.mean(1).cpu()
                torch.cuda.synchronize()
                if i >= args.warmup:
                    times.append((time.perf_counter() - t0) * 1000)
        res["models"][name] = {"path": path, "p50_ms": float(np.percentile(times, 50)),
                               "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times)),
                               "parameters": sum(x.numel() for x in lm.parameters())}
        print(f"[{name}] p50={res['models'][name]['p50_ms']:.2f}ms p95={res['models'][name]['p95_ms']:.2f}ms")
        del lm
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
