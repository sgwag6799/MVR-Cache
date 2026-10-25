#!/usr/bin/env bash
# 공통 문단 탐색 묶음 실행: (공통 문장 풀: 공용 / 하위 과제별) × (라벨: 8B / 70B)
#   각 조합마다 1단계 오프라인 진단(설계 기본값 + M5·M6, 쌍별 점수 저장)과 문맥 오염 2×2를 돌린다.
#   /content/drive/MyDrive 가 마운트되어 있으면 단계가 끝날 때마다 결과를 $DRIVE_DIR 로 복사한다.
# Usage: bash scripts/run_prefix_labels_tasks.sh   (colab_setup.sh 이후, 저장소 루트에서)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$REPO/data
DRIVE_DIR=${DRIVE_DIR:-/content/drive/MyDrive/mvr_eval}
export PYTHONPATH=$REPO/mvr-cache
cd "$REPO/mvr-cache"

sync_drive() {
  if mountpoint -q /content/drive 2>/dev/null || [ -d /content/drive/MyDrive ]; then
    mkdir -p "$DRIVE_DIR"
    rsync -a --exclude 'data/' "$DATA/prefix" "$DATA/prefix_task" "$DRIVE_DIR/" 2>/dev/null || true
    echo "[sync] copied results to $DRIVE_DIR"
  fi
}

# 0. 하위 과제별 공통 문장 풀로 데이터 만들기 + 점검
if [ ! -f "$DATA/prefix_task/data/data_meta.json" ]; then
  python -m benchmarks.prefix.make_prefix_data --pool benchmarks/prefix/prefix_pool_task.json \
    --train "$DATA/cls_seed42/train.parquet" --val "$DATA/cls_seed42/val.parquet" \
    --test "$DATA/cls_seed42_nodedup/test41k.parquet" --out "$DATA/prefix_task/data"
  python -m benchmarks.prefix.prefix_sanity --data "$DATA/prefix_task/data"
fi

run() {  # run <pool dir> <label tag> <resp col>
  local base=$DATA/$1 tag=$2 col=$3
  if [ ! -f "$base/offline_$tag/prefix_gate.json" ]; then
    python -m benchmarks.prefix.prefix_offline --data "$base/data" --out "$base/offline_$tag" \
      --resp-col "$col" --extra-conditions M5 M6 --save-scores 2>&1 | grep -v Warning | tee "$base/offline_$tag.log"
    sync_drive
  fi
  if [ ! -f "$base/contamination_$tag/prefix_contamination.json" ]; then
    python -m benchmarks.prefix.prefix_contamination --data "$base/data" --out "$base/contamination_$tag" \
      --resp-col "$col" 2>&1 | grep -v Warning | tee "$base/contamination_$tag.log"
    sync_drive
  fi
}

run prefix_task 8b response_llama_3_8b
run prefix_task 70b response_llama_3_70b
run prefix 70b response_llama_3_70b
# 공용 풀 · 8B 오프라인은 이미 있음 (stage1_out). 문맥 오염만 과제별 지표를 넣어 다시 돈다.
mkdir -p "$DATA/prefix/offline_8b" && cp -n "$DATA/prefix/offline/"* "$DATA/prefix/offline_8b/" 2>/dev/null || true
run prefix 8b response_llama_3_8b
echo "ALL DONE"
