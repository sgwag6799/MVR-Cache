#!/usr/bin/env bash
# 주장 A (Classification): 2단계 41K 스트림. 작업 목록을 받아 N개씩 병렬로 돌린다.
# scripts/run_stage2_lmarena.sh와 같은 조건이고, 데이터 · 정답 · 헤드 · RL 체크포인트만 Classification 것이다.
# Usage: bash scripts/run_stage2_cls.sh <jobs.txt> <parallel>
#   jobs.txt 한 줄 = "<이름> <단계 0|8> <시드> <조건>"
#     A vCache · C 규칙 분할 + 균등 · E 규칙 분할 + 학습 가중치(M3)
#     J 논문 MVR-Cache: RL 분할기 + multivector_top_k k 10 (--mix-fullcos · --include-full-embedding 없음)
#     F J에 --mix-fullcos --include-full-embedding 추가 (README 예시 설정, 기본 작업 목록에는 없음)
#   정답: 8B 응답 문자열 일치 (--llm-col response_llama_3_8b --similarity-evaluator string)
#   환경변수: SLEEP (기본 0.02), CAP_SCOPE (기본 original), HEAD, RLCKPT, OUT
set -uo pipefail
JOBS=$(realpath "$1"); PAR=${2:-4}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${OUT:-$REPO/data/cls_stage2/runs}")  # 상대 경로는 실행한 위치 기준 (아래 cd 전에 고정)
cd "$REPO/mvr-cache"
export PYTHONPATH=$PWD HF_ENDPOINT=https://huggingface.co
# 프로세스마다 CPU 스레드를 2개로 제한 (병렬 실행 시 모든 프로세스가 코어를 나눠 쓰다 느려짐)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# 조건 E의 M3 가중치 헤드: Classification L8 train(8B 라벨)으로 학습한 것. L8 헤드 하나를 모든 단계에 씀
HEAD=${HEAD:-$REPO/mvr-cache/benchmarks/prefix/stage1_out/prefix_w_variant_L8.pt}
RLCKPT=${RLCKPT:-$REPO/data/cls_stage2/rl/best.ckpt}
SLEEP=${SLEEP:-0.02}
CAP_SCOPE=${CAP_SCOPE:-original}
mkdir -p "$OUT"

run_one() {
  local name=$1 level=$2 seed=$3 cond=$4
  [ -z "$cond" ] && return 0
  local D=$REPO/data/cls_stage2/stream/L${level}_test.parquet
  local common="--dataset $D --llm-col response_llama_3_8b --similarity-evaluator string --delta 0.01 --sleep $SLEEP --filter-true-column fits_all_levels --seed $seed --output-json $OUT/$name.json"
  local pool=""
  [ "$CAP_SCOPE" = "original" ] && pool="--prefix-pool benchmarks/prefix/prefix_pool.json"
  local rule="--splitter-mode prefix_rule --rule-max-cuts 32 --rule-cap-scope $CAP_SCOPE $pool --candidate-selection top_k --candidate-k 10 --use-cached-candidate-segments --splitter-device cuda --timing-output $OUT/$name.timing.txt"
  local paper="--splitter-mode rl --splitter-checkpoint $RLCKPT --candidate-selection multivector_top_k --candidate-k 10 --mix-fullcos --include-full-embedding --splitter-overlap-tokens 0 --splitter-device cuda --timing-output $OUT/$name.timing.txt"
  local paper_seg=${paper/--mix-fullcos --include-full-embedding /}
  [ -f "$OUT/$name.json" ] && { echo "[skip] $name"; return 0; }
  local t0=$(date +%s) rc
  case $cond in
    A) python benchmarks/eval_sembenchmark_verified.py $common --device cuda ;;
    C) python benchmarks/eval_sembenchmark_verified_splitter.py $common $rule ;;
    E) python benchmarks/eval_sembenchmark_verified_splitter.py $common $rule --segment-head $HEAD ;;
    F) python benchmarks/eval_sembenchmark_verified_splitter.py $common $paper ;;
    J) python benchmarks/eval_sembenchmark_verified_splitter.py $common $paper_seg ;;
  esac > "$OUT/$name.log" 2>&1
  rc=$?
  echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s"
}
export -f run_one
export REPO OUT HEAD RLCKPT SLEEP CAP_SCOPE
grep -v "^#" "$JOBS" | grep -v "^\s*$" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "ALL JOBS DONE"
