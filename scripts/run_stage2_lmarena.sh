#!/usr/bin/env bash
# 실험 8: 2단계 41K 스트림 (LmArena, 합친 정답). 작업 목록을 받아 N개씩 병렬로 돌린다.
# Usage: bash scripts/run_stage2_lmarena.sh <jobs.txt> <parallel>
#   jobs.txt 한 줄 = "<이름> <단계 0|8> <시드> <조건 A..G, I>"
#   I = 논문 MVR-Cache(F)에서 --mix-fullcos만 뺀 것 (SMaxSim만으로 판정, base 인코더)
#   환경변수 (기본값 = 실험 8에서 쓴 값):
#     SLEEP=0.02        요청마다 기다리는 시간(초). 주장 A 보강 실험은 0.1 (논문 README의 MVR-Cache 값)
#     CAP_SCOPE=original 상한 32를 셀 범위. original = 공통 문장 목록으로 공통 문단 끝을 찾아 원본 부분에서만 셈,
#                       all = 목록 없이 맨 앞부터 셈 (주장 A 보강 실험)
set -uo pipefail
JOBS=$(realpath "$1"); PAR=${2:-4}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${OUT:-$REPO/data/exp8/runs}")  # 상대 경로는 실행한 위치 기준 (아래 cd 전에 고정)
cd "$REPO/mvr-cache"
export PYTHONPATH=$PWD HF_ENDPOINT=https://huggingface.co
# 프로세스마다 CPU 스레드를 2개로 제한 (병렬 실행 시 12코어를 모든 프로세스가 나눠 쓰다 느려짐)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
LORA=$REPO/data/exp7/lora/model
# 조건 E의 M3 가중치 헤드. 기본은 L8 헤드 하나를 모든 단계에 씀 (실제 서비스는 공통 문단 수를 모르므로)
HEAD=${HEAD:-$REPO/mvr-cache/benchmarks/prefix/exp2_lmarena/offline_merged/prefix_w_variant_L8.pt}
RLCKPT=${RLCKPT:-$REPO/data/exp8/rl/best.ckpt}
SLEEP=${SLEEP:-0.02}
CAP_SCOPE=${CAP_SCOPE:-original}
mkdir -p "$OUT"

run_one() {
  local name=$1 level=$2 seed=$3 cond=$4
  [ -z "$cond" ] && return 0
  local D=$REPO/data/exp8/stream/L${level}_test.parquet
  local common="--dataset $D --similarity-evaluator benchmark_id_set --delta 0.01 --sleep $SLEEP --filter-true-column fits_all_levels --seed $seed --output-json $OUT/$name.json"
  local pool=""
  [ "$CAP_SCOPE" = "original" ] && pool="--prefix-pool benchmarks/prefix/prefix_pool.json"
  local rule="--splitter-mode prefix_rule --rule-max-cuts 32 --rule-cap-scope $CAP_SCOPE $pool --candidate-selection top_k --candidate-k 10 --use-cached-candidate-segments --splitter-device cuda --timing-output $OUT/$name.timing.txt"
  local paper="--splitter-mode rl --splitter-checkpoint $RLCKPT --candidate-selection multivector_top_k --candidate-k 10 --mix-fullcos --include-full-embedding --splitter-overlap-tokens 0 --splitter-device cuda --timing-output $OUT/$name.timing.txt"
  [ -f "$OUT/$name.json" ] && { echo "[skip] $name"; return 0; }
  local t0=$(date +%s) rc
  case $cond in
    A) python benchmarks/eval_sembenchmark_verified.py $common --device cuda ;;
    B) BGE_MODEL_PATH=$LORA python benchmarks/eval_sembenchmark_verified.py $common --device cuda ;;
    C) python benchmarks/eval_sembenchmark_verified_splitter.py $common $rule ;;
    D) BGE_MODEL_PATH=$LORA python benchmarks/eval_sembenchmark_verified_splitter.py $common $rule ;;
    E) python benchmarks/eval_sembenchmark_verified_splitter.py $common $rule --segment-head $HEAD ;;
    F) python benchmarks/eval_sembenchmark_verified_splitter.py $common $paper ;;
    G) BGE_MODEL_PATH=$LORA python benchmarks/eval_sembenchmark_verified_splitter.py $common $paper ;;
    I) python benchmarks/eval_sembenchmark_verified_splitter.py $common ${paper/--mix-fullcos /} ;;
  esac > "$OUT/$name.log" 2>&1
  rc=$?
  echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s"
}
export -f run_one
export REPO OUT LORA HEAD RLCKPT SLEEP CAP_SCOPE
grep -v "^#" "$JOBS" | grep -v "^\s*$" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "ALL JOBS DONE"
