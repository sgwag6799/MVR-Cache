#!/usr/bin/env bash
# 주장 A (Classification): 논문 MVR-Cache의 RL 분할기를 Classification train(원본 L0)으로 학습한다.
# 레시피는 LmArena(scripts/train_rl_splitter_lmarena.sh)와 같고, 정답 판정만 8B 응답 문자열 일치로 바꾼다
# (--label_mode string --response_column response_llama_3_8b). 논문 체크포인트가 공개되지 않아 이 분할기를 기준선으로 쓴다.
# Usage (repo root, scripts/make_stage2_cls_data.py 이후):  bash scripts/train_rl_splitter_cls.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${D:-$REPO/data/cls_stage2/rl}
cd "$REPO/rl-training-algorithm"
export PYTHONPATH=$REPO/mvr-cache:$PWD HF_ENDPOINT=https://huggingface.co
python RL4COTrainer.py --gpu_id 0 \
  --train_parquet "$D/train.parquet" --val_parquet "$D/val.parquet" --test_parquet "$D/val.parquet" \
  --parquet_text_column prompt --label_mode string --response_column response_llama_3_8b \
  --train_sampling_mode anchor_nn \
  --nn_warmup_epochs 5 --nn_candidate_topk 10 --nn_rebuild_every_n_epochs 1 --train_data_size 10000 \
  --batch_size 8 --accumulate_grad_batches 2 --lr 1e-4 --max_epochs 200 --check_val_every_n_epoch 5 \
  --checkpoint_dir "$D/ckpt" --policy_mode separate --punctuation_only --bce_auto_balance \
  --precompute_token_embeddings > "$D/train.log" 2>&1
B=$(ls "$D"/ckpt/epoch=*.ckpt | head -1)
ln -f "$B" "$D/best.ckpt"
echo "best=$B"
