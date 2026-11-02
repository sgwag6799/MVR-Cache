#!/usr/bin/env bash
# 실험 3: 공통 문단이 붙은 Classification 41K 스트림에서 판정자(learned_all_add35) 확인.
# 1) 판정자가 없으면 train+val 스트림(3,726행)을 기준선 C로 흘려 특징을 모으고(test 미사용)
#    scripts/judge/judge_fit.py로 판정자와 p*를 고정한다 (단계 · 풀마다 따로).
# 2) 작업 목록의 test 실행을 N개씩 병렬로 돌린다.
# Usage (repo root): POOL=shared bash scripts/judge/run_judge_prefix_cls.sh scripts/judge/jobs_judge_core.txt [병렬, 기본 3]
#   jobs 한 줄 = "<이름> <단계 0|8> <시드> <조건>"
#     C  규칙 분할(구두점, 상한 32) + 균등, 후보 10개 (기준선, 실험 2의 C와 같은 설정)
#     CJ C + 판정자            E  C + 학습 가중치(M3)            EJ E + 판정자
#   환경변수: POOL (shared | task), SLEEP (0.02), BUDGET (0.0035 = add35), SCOPE (all | cold),
#            COLLECT_SEEDS (판정자 수집 시드, 기본 "42". 여러 개면 p* 중앙값), HEAD (E의 헤드), OUT
set -uo pipefail
JOBS=$(realpath "$1"); PAR=${2:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
POOL=${POOL:-shared}; SLEEP=${SLEEP:-0.02}; BUDGET=${BUDGET:-0.0035}; SCOPE=${SCOPE:-all}
COLLECT_SEEDS=${COLLECT_SEEDS:-42}
DATA=$REPO/data/judge_cls/$POOL
OUT=$(realpath -m "${OUT:-$REPO/data/judge_cls/runs/$POOL}")
JDIR=$REPO/data/judge_cls/judge
HEAD=${HEAD:-$REPO/mvr-cache/benchmarks/prefix/stage1_out/prefix_w_variant_L8.pt}
case $POOL in
  shared) PREFIX_POOL=benchmarks/prefix/prefix_pool.json ;;
  task)   PREFIX_POOL=benchmarks/prefix/prefix_pool_task_flat.json ;;
  *) echo "POOL must be shared or task"; exit 1 ;;
esac
cd "$REPO/mvr-cache"
export PYTHONPATH=$PWD HF_ENDPOINT=https://huggingface.co
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$OUT/collect" "$JDIR"
BASE="--llm-col response_llama_3_8b --similarity-evaluator string --delta 0.01 --sleep $SLEEP"
RULE="--splitter-mode prefix_rule --rule-max-cuts 32 --rule-cap-scope original --prefix-pool $PREFIX_POOL --candidate-selection top_k --candidate-k 10 --use-cached-candidate-segments --splitter-device cuda"
EVAL="python ../scripts/judge/run_eval.py benchmarks.eval_sembenchmark_verified_splitter"

judge_path() { echo "$JDIR/${POOL}_L$1_${SCOPE}_b${BUDGET}.json"; }

build_judge() {  # build_judge <단계>
  local L=$1 J logs="" cs log
  J=$(judge_path "$L")
  [ -f "$J" ] && { echo "[judge] reuse $J"; return 0; }
  for cs in $COLLECT_SEEDS; do
    log=$OUT/collect/L${L}_s${cs}.jsonl
    if [ ! -s "$log" ]; then
      echo "[collect] L$L seed $cs"
      DECISION_LOG=$log $EVAL --dataset "$DATA/L${L}_trainval.parquet" $BASE --seed "$cs" \
        --output-json "$OUT/collect/L${L}_s${cs}.json" $RULE --collect-consensus \
        > "$OUT/collect/L${L}_s${cs}.log" 2>&1 || { echo "collect failed (see log)"; return 1; }
    fi
    logs="$logs $log"
  done
  python ../scripts/judge/judge_fit.py --logs $logs --budget "$BUDGET" --scope "$SCOPE" --out "$J" \
    | tee "$OUT/collect/fit_L${L}.log" || return 1
}

run_one() {
  local name=$1 level=$2 seed=$3 cond=$4
  [ -z "$cond" ] && return 0
  [ -f "$OUT/$name.json" ] && { echo "[skip] $name"; return 0; }
  local common="--dataset $DATA/L${level}_test.parquet $BASE --filter-true-column fits_all_levels --seed $seed --output-json $OUT/$name.json --timing-output $OUT/$name.timing.txt"
  local judge="--consensus-model $(judge_path "$level") --consensus-scope $SCOPE"
  local t0=$(date +%s) rc
  case $cond in
    C)  DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE ;;
    CJ) DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE $judge ;;
    E)  DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE --segment-head $HEAD ;;
    EJ) DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE --segment-head $HEAD $judge ;;
    *) echo "unknown condition $cond"; return 1 ;;
  esac > "$OUT/$name.log" 2>&1
  rc=$?
  echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s"
}

# 판정자가 필요한 단계는 test 실행 전에 먼저 고정한다
for L in $(grep -v "^#" "$JOBS" | awk 'NF==4 && ($4=="CJ" || $4=="EJ") {print $2}' | sort -u); do
  build_judge "$L" || exit 1
done
export -f run_one judge_path
export REPO OUT DATA JDIR POOL SCOPE BUDGET HEAD BASE RULE EVAL
grep -v "^#" "$JOBS" | grep -v "^\s*$" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "ALL JOBS DONE"
