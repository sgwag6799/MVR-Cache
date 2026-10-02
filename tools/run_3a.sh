#!/bin/bash
# Step 2 (reference) + Step 3a weightings on the full Classification dataset, paper conditions.
#
#   bash tools/run_3a.sh                      # uniform idf, 3 repeats each  (~12h)
#   CONDITIONS="uniform idf length centroid" REPEATS=3 bash tools/run_3a.sh   # 전체 (~24h)
#   MACHINE=mini REPEATS=1 bash tools/run_3a.sh                               # 빠른 확인
#
# Conditions run here must be compared against the `uniform` runs from THIS machine:
# the verified policy updates thresholds on a background thread, so a faster or slower
# machine can shift hit rate slightly. Cross-machine absolute numbers are not comparable.
set -euo pipefail

cd "$(dirname "$0")/../mvr-cache"
PY=../.venv/bin/python
MACHINE="${MACHINE:-mini}"
REPEATS="${REPEATS:-3}"
CONDITIONS="${CONDITIONS:-uniform idf}"
OUT="results/paper"
STATS="$OUT/classification_weight_stats_prefix3k.pt"

export HF_ENDPOINT=https://huggingface.co
export HF_CACHE_BASE="$HOME/hf_cache"
export HF_HUB_DISABLE_XET=1

mkdir -p "$OUT"

# IDF / centroid statistics. Built from the first 3000 prompts only, and they use no
# labels (document frequency + mean segment embedding), so no label leakage into the
# evaluation. Note in write-ups that these prompts are also part of the eval stream.
if [ ! -f "$STATS" ]; then
  echo "== 가중치 통계 생성 (IDF, 중심 벡터) — 약 5분"
  $PY benchmarks/build_segment_weight_stats.py \
    --dataset data/classification.parquet --response-col response_llama_3_8b \
    --start 0 --n 3000 --epochs 1 --out "$STATS" 2>&1 | tail -4
fi

for w in $CONDITIONS; do
  for r in $(seq 1 "$REPEATS"); do
    tag="classification_full_rule_w${w}_${MACHINE}_run${r}"
    if [ -f "$OUT/$tag.json" ]; then echo "skip $tag (이미 있음)"; continue; fi
    echo "== $tag  ($(date '+%H:%M'))"
    caffeinate -i $PY benchmarks/eval_sembenchmark_verified_splitter.py \
      --dataset data/classification.parquet --delta 0.01 --sleep 0.02 \
      --splitter-mode rule --splitter-max-segments 4 \
      --segment-weighting "$w" --segment-weight-stats "$STATS" \
      --candidate-selection multivector_top_k --candidate-k 10 \
      --splitter-device cpu --mix-fullcos --include-full-embedding \
      --similarity-evaluator string --llm-col response_llama_3_8b \
      --output-json "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
    $PY -c "import json;s=json.load(open('$OUT/$tag.json'))['summary'];print('  hit %.2f%%  tp %d fp %d  err %.2f%%  %.0f분'%(s['hit_rate']*100,s['tp'],s['fp'],s['fp']/s['n']*100,s['total_time']/60))"
  done
done

echo
echo "== 요약"
$PY - <<'EOF'
import glob, json, statistics as st, re, os
rows = {}
for f in sorted(glob.glob('results/paper/classification_full_rule_w*_run*.json')):
    m = re.search(r'_w([a-z]+)_([a-z0-9]+)_run', os.path.basename(f))
    if not m: continue
    s = json.load(open(f))['summary']
    rows.setdefault(f"{m.group(1)} ({m.group(2)})", []).append(
        (s['hit_rate'] * 100, s['fp'] / s['n'] * 100))
for k, v in rows.items():
    hr = [x[0] for x in v]; err = [x[1] for x in v]
    sd = f" ± {st.stdev(hr):.2f}" if len(hr) > 1 else ""
    print(f"{k:22s} hit {st.mean(hr):.2f}%{sd}  (n={len(hr)})  err {st.mean(err):.2f}%")
EOF
