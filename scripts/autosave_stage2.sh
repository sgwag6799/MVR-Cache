#!/usr/bin/env bash
# 주장 A 스트림 실행 결과를 주기적으로 저장소에 집계 · 커밋 · push 한다 (VM이 끊겨도 결과가 남게).
# 실행별 원본 JSON(per_sample 포함, 수 MB)은 올리지 않고, 집계표 · 실행 요약(summary) · timing만 올린다.
# 배치 로그에 ALL JOBS DONE 이 찍히면 마지막으로 한 번 더 저장하고 끝난다.
# Usage (repo root, VSCode 터미널에서: push 인증이 필요):
#   nohup bash scripts/autosave_stage2.sh <runs 폴더> <저장소 안 결과 폴더> <배치 로그> [간격 초, 기본 900] \
#     > data/autosave_stage2.log 2>&1 < /dev/null &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUNS=$(realpath -m "$1"); DEST=$REPO/$2; LOG=$3; INTERVAL=${4:-900}
cd "$REPO"
save() {
  mkdir -p "$DEST/run_summaries"
  (cd mvr-cache && python -m benchmarks.prefix.prefix_stage2_report --runs "$RUNS" --out "$DEST" > /dev/null 2>&1) || true
  python - "$RUNS" "$DEST/run_summaries" <<'EOF'
import glob, json, os, sys
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    try:
        json.dump(json.load(open(f))["summary"], open(os.path.join(sys.argv[2], os.path.basename(f)), "w"),
                  indent=1, default=float)
    except Exception:
        pass  # 아직 쓰는 중인 파일
EOF
  cp "$RUNS"/*.timing.txt "$DEST/run_summaries/" 2>/dev/null || true
  [ -f "$LOG" ] && cp "$LOG" "$DEST/batch.log"
  git add -f "$DEST"
  if ! git diff --cached --quiet; then
    git -c user.name=sgwag6799 -c user.email=sgwag6799@gmail.com commit -qm "Claim A: autosave stage 2 results ($(basename "$DEST"))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
      && git push -q origin "$(git branch --show-current)" && echo "[$(date +%T)] pushed $(git log --oneline -1)"
  fi
}
while true; do
  save
  grep -q "ALL JOBS DONE" "$LOG" 2>/dev/null && { save; echo "finished"; exit 0; }
  sleep "$INTERVAL"
done
