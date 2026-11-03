#!/usr/bin/env bash
# 실험 3: 공통 문단이 붙은 Classification 41K 스트림에서 판정자(learned_all_add35) 확인.
# 1) 판정자가 없으면 train+val 스트림(3,726행)을 기준선 C로 흘려 특징을 모으고(test 미사용)
#    scripts/judge/judge_fit.py로 판정자와 p*를 고정한다 (단계 · 풀마다 따로).
# 2) 작업 목록의 test 실행을 N개씩 병렬로 돌린다.
# Usage (repo root): POOL=shared bash scripts/judge/run_judge_prefix_cls.sh scripts/judge/jobs_judge_core.txt [병렬, 기본 3]
#   jobs 한 줄 = "<이름> <단계 0|8> <시드> <조건>"
#     C  규칙 분할(구두점, 상한 32) + 균등, 후보 10개 (기준선, 실험 2의 C와 같은 설정)
#     CJ C + 판정자            E  C + 학습 가중치(M3)            EJ E + 판정자
#     [판정자 실험 G] CJL C + 판정자 + 실행 중 한도 JC1L (calibrated, z 1, probe 0.1, label only, 한도 ONLINE_EPS)
#                     EJL E + 판정자 + JC1L          C0125 · C015 · C020  C 에서 δ 만 0.0125 · 0.015 · 0.02
#   환경변수: POOL (shared | task), SLEEP (0.02), BUDGET (0.0035 = add35), SCOPE (all | cold),
#            COLLECT_SEEDS (판정자 수집 시드, 기본 "42". 여러 개면 p* 중앙값), HEAD (E의 헤드), OUT
set -uo pipefail
JOBS=$(realpath "$1"); PAR=${2:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
POOL=${POOL:-shared}; SLEEP=${SLEEP:-0.02}; BUDGET=${BUDGET:-0.0035}; SCOPE=${SCOPE:-all}
COLLECT_SEEDS=${COLLECT_SEEDS:-42}
ONLINE_EPS=${ONLINE_EPS:-0.0035}
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
  local cap="--consensus-online-budget $ONLINE_EPS --consensus-online-mode calibrated --consensus-online-z 1 --consensus-online-probe 0.1 --consensus-online-prior 20 --consensus-online-bin-width 0.02 --consensus-online-label-only --consensus-seed $seed"
  local dcommon="$common"
  case $cond in C0125) dcommon="${common/--delta 0.01/--delta 0.0125}" ;; C015) dcommon="${common/--delta 0.01/--delta 0.015}" ;; C020) dcommon="${common/--delta 0.01/--delta 0.02}" ;; esac
  # 실행 기록: GPU 종류 · 커밋 해시 · 조건 (실행 조건 "실행 기록에 GPU 종류 · 커밋 해시를 남긴다")
  printf '{"name": "%s", "cond": "%s", "level": %s, "seed": %s, "gpu": "%s", "commit": "%s", "dirty": %s, "started": "%s"}\n' \
    "$name" "$cond" "$level" "$seed" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" \
    "$(git -C "$REPO" rev-parse HEAD)" "$(git -C "$REPO" status --porcelain -uno | grep -q . && echo true || echo false)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/$name.meta.json"
  local t0=$(date +%s) rc
  case $cond in
    C)  DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE ;;
    CJ) DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE $judge ;;
    E)  DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE --segment-head $HEAD ;;
    EJ) DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE --segment-head $HEAD $judge ;;
    CJL) ONLINE_BUDGET_STATE=$OUT/$name.budget.json DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE $judge $cap ;;
    EJL) ONLINE_BUDGET_STATE=$OUT/$name.budget.json DECISION_LOG=$OUT/$name.jsonl $EVAL $common $RULE --segment-head $HEAD $judge $cap ;;
    C0125|C015|C020) DECISION_LOG=$OUT/$name.jsonl $EVAL $dcommon $RULE ;;
    *) echo "unknown condition $cond"; return 1 ;;
  esac > "$OUT/$name.log" 2>&1
  rc=$?
  echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s"
}

# 판정자가 필요한 단계는 test 실행 전에 먼저 고정한다
for L in $(grep -v "^#" "$JOBS" | awk 'NF==4 && ($4 ~ /^(CJ|EJ|CJL|EJL)$/) {print $2}' | sort -u); do
  build_judge "$L" || exit 1
done
export -f run_one judge_path
export REPO OUT DATA JDIR POOL SCOPE BUDGET HEAD BASE RULE EVAL ONLINE_EPS
grep -v "^#" "$JOBS" | grep -v "^\s*$" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "ALL JOBS DONE"
