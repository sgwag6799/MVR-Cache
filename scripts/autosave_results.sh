#!/usr/bin/env bash
# 실행 중인 결과(data/ 아래, git 제외)를 저장소 폴더로 복사해 주기적으로 커밋·push 한다.
# run_prefix_labels_tasks.sh 로그에 ALL DONE 이 찍히면 마지막으로 한 번 더 저장하고 끝난다.
# Usage: bash scripts/autosave_results.sh [interval_sec]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST=$REPO/mvr-cache/benchmarks/prefix/results_labels_tasks
LOG=$REPO/data/run_labels_tasks.log
INTERVAL=${1:-600}
cd "$REPO"
save() {
  mkdir -p "$DEST"
  for pool in prefix prefix_task; do
    for d in "data/$pool"/offline_* "data/$pool"/contamination_*; do
      [ -d "$d" ] && mkdir -p "$DEST/$pool" && rsync -a "$d" "$DEST/$pool/"
    done
    for f in "data/$pool"/offline_*.log "data/$pool"/contamination_*.log "data/$pool/data/data_meta.json"; do
      [ -f "$f" ] && mkdir -p "$DEST/$pool" && cp "$f" "$DEST/$pool/"
    done
  done
  [ -f "$LOG" ] && cp "$LOG" "$DEST/"
  git add -f "$DEST"
  if ! git diff --cached --quiet; then
    git -c user.name=sgwag6799 -c user.email=sgwag6799@gmail.com commit -qm "Prefix stress test: autosave pool x label results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin prefix-stress-colab && echo "[$(date +%T)] pushed $(git log --oneline -1)"
  fi
}
while true; do
  save
  grep -q "ALL DONE" "$LOG" 2>/dev/null && { save; echo "finished"; exit 0; }
  sleep "$INTERVAL"
done
