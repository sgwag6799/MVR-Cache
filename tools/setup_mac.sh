#!/bin/bash
# Set up this repo on another Mac (Apple Silicon, no CUDA).
#
# Copy the whole MVR-Cache folder over (AirDrop is fine) WITHOUT .venv, then run:
#   bash tools/setup_mac.sh
#
# Installs Python 3.11 + a virtualenv with the exact package versions from
# requirements-lock.txt, builds the local multi-vector hnswlib, and verifies imports.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
echo "repo: $ROOT"

if ! command -v brew >/dev/null; then
  echo "Homebrew가 필요해요: https://brew.sh 에서 설치한 뒤 다시 실행하세요."
  exit 1
fi

if ! command -v /opt/homebrew/bin/python3.11 >/dev/null; then
  echo "== Python 3.11 설치"
  brew install python@3.11
fi

echo "== 가상환경 생성"
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip setuptools wheel

echo "== 패키지 설치 (몇 분 걸려요)"
.venv/bin/pip install -q -r requirements-lock.txt

echo "== 멀티벡터 hnswlib 빌드 (기계마다 새로 빌드해야 함)"
.venv/bin/pip install -q -e mvr-cache/vcache/vcache_core/cache/embedding_store/hnswlib

echo "== vcache 패키지 경로 등록"
echo "$ROOT/mvr-cache" > .venv/lib/python3.11/site-packages/mvr_cache.pth

echo "== 확인"
.venv/bin/python - <<'EOF'
import torch, hnswlib, scipy, vcache
from vcache.vcache_core.splitter.RuleSplitter import RulePunctuationSplitter
assert hasattr(hnswlib.Index, "knn_query_with_parent"), "멀티벡터 hnswlib 빌드 실패"
print("OK  torch", torch.__version__, "| scipy", scipy.__version__)
EOF

echo
echo "데이터 파일 확인:"
ls -la mvr-cache/data/*.parquet 2>/dev/null || echo "  (없음) 맥북에서 mvr-cache/data/*.parquet 를 복사해 오세요."
echo
echo "다음: bash tools/run_3a.sh"
