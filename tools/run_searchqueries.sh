#!/bin/bash
# SearchQueries (paper dataset #2) — control + Step 2 under paper conditions.
#
#   bash tools/run_searchqueries.sh                        # nosplit + uniform, 3 repeats (~17h)
#   REPEATS=1 bash tools/run_searchqueries.sh              # 빠른 확인 (~6h)
#   CONDITIONS="nosplit uniform idf" bash tools/run_searchqueries.sh
#
# Why: on Classification the whole gain came from candidate selection (+3.43%p), not from
# splitting (-0.11%p). This checks whether that holds on the second paper dataset.
#
# vCache (Step 1) on this dataset is already done: 3.55% ± 0.02, error 0.42%.
#
# Conditions:
#   nosplit  = multivector top-10 rerank, no splitting   (--splitter-max-segments 0)
#   uniform  = rule punctuation split, equal weights     (Step 2)
#   idf      = rule split + IDF weights                  (Step 3a)
#
# NOTE: SearchQueries prompts are ~5 words and only 0.6% contain punctuation, so the rule
# splitter usually produces a single segment. `uniform` is therefore expected to land very
# close to `nosplit`; that is itself the result worth recording.
set -euo pipefail

cd "$(dirname "$0")/../mvr-cache"
PY=../.venv/bin/python
MACHINE="${MACHINE:-mini}"
REPEATS="${REPEATS:-3}"
CONDITIONS="${CONDITIONS:-nosplit uniform}"
OUT="results/paper"
STATS="$OUT/searchqueries_weight_stats_prefix3k.pt"

export HF_ENDPOINT=https://huggingface.co
export HF_CACHE_BASE="$HOME/hf_cache"
export HF_HUB_DISABLE_XET=1

mkdir -p "$OUT"
[ -f data/searchqueries.parquet ] || { echo "data/searchqueries.parquet 이 없어요 (맥북에서 복사)"; exit 1; }

if [[ " $CONDITIONS " == *" idf "* ]] && [ ! -f "$STATS" ]; then
  echo "== IDF 통계 생성 (앞 3,000개, 라벨 미사용)"
  $PY benchmarks/build_segment_weight_stats.py \
    --dataset data/searchqueries.parquet --label-col id_set \
    --start 0 --n 3000 --epochs 1 --out "$STATS" 2>&1 | tail -3
fi

for c in $CONDITIONS; do
  case "$c" in
    nosplit)  SEG=0; W=uniform ;;
    uniform)  SEG=4; W=uniform ;;
    idf)      SEG=4; W=idf ;;
    *) echo "알 수 없는 조건: $c"; exit 1 ;;
  esac
  for r in $(seq 1 "$REPEATS"); do
    tag="searchqueries_full_${c}_${MACHINE}_run${r}"
    [ -f "$OUT/$tag.json" ] && { echo "skip $tag (이미 있음)"; continue; }
    echo "== $tag  시작 $(date '+%m/%d %H:%M')  (1회 약 3시간)"
    caffeinate -i $PY benchmarks/eval_sembenchmark_verified_splitter.py \
      --dataset data/searchqueries.parquet --delta 0.01 --sleep 0.02 \
      --splitter-mode rule --splitter-max-segments $SEG \
      --segment-weighting $W ${STATS:+--segment-weight-stats "$STATS"} \
      --candidate-selection multivector_top_k --candidate-k 10 \
      --splitter-device cpu --mix-fullcos --include-full-embedding \
      --similarity-evaluator benchmark_id_set \
      --output-json "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
    $PY -c "import json;s=json.load(open('$OUT/$tag.json'))['summary'];print('  hit %.2f%%  tp %d fp %d  err %.2f%%  %.0f분'%(s['hit_rate']*100,s['tp'],s['fp'],s['fp']/s['n']*100,s['total_time']/60))"
  done
done

echo
echo "== 요약 (vCache 기준선 3.55% ± 0.02)"
$PY - <<'EOF'
import glob, json, os, re, statistics as st
rows = {}
for f in sorted(glob.glob('results/paper/searchqueries_full_*_run*.json')):
    m = re.search(r'searchqueries_full_([a-z]+)_([a-z0-9]+)_run', os.path.basename(f))
    if not m: continue
    s = json.load(open(f))['summary']
    rows.setdefault(f"{m.group(1)} ({m.group(2)})", []).append((s['hit_rate'] * 100, s['fp'] / s['n'] * 100))
for k, v in rows.items():
    hr = [x[0] for x in v]; err = [x[1] for x in v]
    sd = f" ± {st.stdev(hr):.2f}" if len(hr) > 1 else ""
    print(f"{k:22s} hit {st.mean(hr):.2f}%{sd}  (n={len(hr)})  err {st.mean(err):.2f}%")
EOF
