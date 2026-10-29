#!/usr/bin/env bash
# 실험 8: 논문 MVR-Cache의 RL 분할기를 논문 레시피 그대로 LmArena train(원본 L0)으로 학습한다.
# 논문 체크포인트(SearchQueries 10K)가 공개되지 않아, 이 분할기를 MVR-Cache 기준선으로 쓴다.
# val reward가 가장 높은 체크포인트(Lightning이 epoch=*.ckpt로 저장)를 best.ckpt로 연결한다.
# Usage (repo root, scripts/make_stage2_lmarena_data.py 이후):  bash scripts/train_rl_splitter_lmarena.sh
#   BGE_MODEL_PATH를 주면 그 인코더로 학습 (기본: bge-base-en-v1.5)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${D:-$REPO/data/exp8/rl}
cd "$REPO/rl-training-algorithm"
export PYTHONPATH=$REPO/mvr-cache:$PWD HF_ENDPOINT=https://huggingface.co
python RL4COTrainer.py --gpu_id 0 \
  --train_parquet "$D/train.parquet" --val_parquet "$D/val.parquet" --test_parquet "$D/val.parquet" \
  --parquet_text_column prompt --label_mode id_set --train_sampling_mode anchor_nn \
  --nn_warmup_epochs 5 --nn_candidate_topk 10 --nn_rebuild_every_n_epochs 1 --train_data_size 10000 \
  --batch_size 8 --accumulate_grad_batches 2 --lr 1e-4 --max_epochs 200 --check_val_every_n_epoch 5 \
  --checkpoint_dir "$D/ckpt" --policy_mode separate --punctuation_only --bce_auto_balance \
  --precompute_token_embeddings > "$D/train.log" 2>&1
B=$(ls "$D"/ckpt/epoch=*.ckpt | head -1)
ln -f "$B" "$D/best.ckpt"
echo "best=$B"
