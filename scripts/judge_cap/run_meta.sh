#!/usr/bin/env bash
# 실행 기록에 GPU 종류 · 커밋 해시를 남긴다. 사용: bash scripts/judge_cap/run_meta.sh <실행이름> <출력폴더>
set -euo pipefail
name=$1; out=${2:-.}
mkdir -p "$out"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "none")
commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
dirty=$(git status --porcelain 2>/dev/null | grep -qv '^??' && echo true || echo false)
printf '{"run": "%s", "gpu": "%s", "commit": "%s", "dirty": %s, "started": "%s"}\n' \
  "$name" "$gpu" "$commit" "$dirty" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out/$name.meta.json"
cat "$out/$name.meta.json"
