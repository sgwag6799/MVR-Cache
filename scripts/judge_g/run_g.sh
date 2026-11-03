#!/usr/bin/env bash
# 판정자 실험 G: 공통문단 L8 + 실행 중 한도 (CJ-L · EJ-L · C δ 곡선). SSH 로 Colab L4 에서 돌린다.
# 처음부터 끝까지 한 번에: 환경 → 데이터(없으면 만듦) → 판정자(없으면 수집 · 학습) → 실행 → 집계.
# 끝난 실행(runs/<이름>.json)은 건너뛰므로, 끊기면 같은 명령을 다시 실행하면 이어서 돈다.
#
# 사용 (repo 루트, tmux 안에서):  bash scripts/judge_g/run_g.sh [작업 목록, 기본 jobs_g_s1.txt] [병렬, 기본 3]
# 환경변수: ALLOW_GPU=1 (L4 가 아니어도 돌림), SKIP_SETUP=1 (환경 설치 건너뜀)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
JOBS=${1:-$REPO/scripts/judge_g/jobs_g_s1.txt}
PAR=${2:-3}
cd "$REPO"
mkdir -p data/judge_cls/runs/shared
LOG=data/judge_g_run.log
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u +%FT%TZ) 판정자 실험 G 시작 · 작업 $(basename "$JOBS") · 병렬 $PAR"

# 0. GPU · 커밋 (기존 기록과 같은 GPU 여야 짝지어 비교할 수 있다)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
echo "GPU: ${GPU:-없음} · 커밋 $(git rev-parse --short HEAD) · 브랜치 $(git rev-parse --abbrev-ref HEAD)"
if [[ "$GPU" != *L4* && "${ALLOW_GPU:-0}" != 1 ]]; then
  echo "🚨 GPU 가 L4 가 아니다 ($GPU). 기존 기록 · 판정자 실험 E 와 같은 L4 로 돌린다 (그래도 돌리려면 ALLOW_GPU=1)"; exit 1
fi

# 1. 환경 (Colab 기본 Python 에 필요한 것만)
if [[ "${SKIP_SETUP:-0}" != 1 ]]; then
  set +u; source colab_setup.sh; set -u
fi
export PYTHONPATH="$REPO/mvr-cache" HF_ENDPOINT=https://huggingface.co

# 2. 데이터 (data/ 는 git 에 올리지 않는다). 있으면 건너뜀
if [[ ! -s data/judge_cls/shared/L8_test.parquet ]]; then
  echo "[data] Classification 원본 → 분할 → 공통 문단 L0 · L8 (공용 풀) → 판정자 스트림"
  [[ -s data/raw/train.parquet ]] || python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkClassification','train.parquet',repo_type='dataset',local_dir='data/raw')"
  [[ -s data/cls_seed42/train.parquet ]] || python scripts/make_cls_splits.py
  (cd mvr-cache && python -m benchmarks.prefix.make_prefix_data --train ../data/cls_seed42/train.parquet --val ../data/cls_seed42/val.parquet \
     --test ../data/cls_seed42_nodedup/test41k.parquet --out ../data/prefix/data --levels 0 8 --modes variant)
  python scripts/judge/make_judge_cls_data.py
fi
ls -la data/judge_cls/shared/

# 3~4. 판정자(없으면 수집 시드 42~45 로 만들고 p* 중앙값) → 실행
POOL=shared COLLECT_SEEDS="42 43 44 45" bash scripts/judge/run_judge_prefix_cls.sh "$JOBS" "$PAR"

# 5. 집계 · 판정 기준
python scripts/judge_g/report_g.py --runs data/judge_cls/runs/shared --judge data/judge_cls/judge/shared_L8_all_b0.0035.json \
  --out data/judge_g_report
echo "===== $(date -u +%FT%TZ) 끝. 결과: data/judge_g_report/ (report_g.txt · results_g.csv)"
