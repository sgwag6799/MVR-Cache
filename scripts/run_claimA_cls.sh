#!/usr/bin/env bash
# 주장 A (Classification): main (sleep 0.02, cap original) → capall (sleep 0.02, cap all) 순서로 하나씩 돌린다.
# 배치마다 결과 폴더를 나누고, autosave가 집계표 · 실행 요약을 mvr-cache/benchmarks/prefix/claimA_cls/<배치>/ 로 push 한다.
# Usage (repo root, VSCode 터미널에서. 데이터 · RL 체크포인트 준비 후):
#   nohup bash scripts/run_claimA_cls.sh [병렬 수, 기본 5] [시작 배치: main|capall] > data/cls_stage2/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PAR=${1:-5}; START=${2:-main}
mkdir -p data/cls_stage2
run_batch() {  # run_batch <이름> <작업 목록> <SLEEP> <CAP_SCOPE>
  local name=$1 jobs=$2 sleep=$3 cap=$4
  echo "[$(date +%T)] start $name (sleep=$sleep cap=$cap)"
  : > "data/cls_stage2/$name.log"  # 이전 실행의 ALL JOBS DONE을 autosave가 보고 바로 끝나지 않게
  nohup bash scripts/autosave_stage2.sh "data/cls_stage2/$name" "mvr-cache/benchmarks/prefix/claimA_cls/$name" \
    "data/cls_stage2/$name.log" > "data/cls_stage2/autosave_$name.log" 2>&1 < /dev/null &
  SLEEP=$sleep CAP_SCOPE=$cap OUT="data/cls_stage2/$name" bash scripts/run_stage2_cls.sh "$jobs" "$PAR" > "data/cls_stage2/$name.log" 2>&1
  wait  # autosave가 마지막 저장을 끝낼 때까지
  echo "[$(date +%T)] done $name"
}
started=0
for b in main capall; do
  [ "$b" = "$START" ] && started=1
  [ $started -eq 1 ] || continue
  case $b in
    main)   run_batch main   scripts/stage2_jobs_cls_main.txt   0.02 original ;;
    capall) run_batch capall scripts/stage2_jobs_cls_capall.txt 0.02 all ;;
  esac
done
echo "ALL BATCHES DONE"
