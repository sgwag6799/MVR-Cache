"""Score-only comparison of the vCache and MVR-cache similarity functions (no cache policy).

For each prompt in a held-out pool, the cache would score the top-k cached neighbours
found by single-vector retrieval. This script builds exactly those candidate pairs,
labels them by ID_Set equality, and compares how well each similarity separates
same-meaning from different-meaning pairs:

  - vcache:   cosine of the BGE mean-pooled embedding (what the vCache baseline uses)
  - mvr:      0.5 * MaxSim(segments + full row) + 0.5 * full cosine
              (= --include-full-embedding --mix-fullcos in eval_sembenchmark_verified_splitter.py)
  - maxsim:   MaxSim(segments + full row) alone

The mvr/maxsim scores are computed for three segmentations, all restricted to the
punctuation split points the RL policy may choose from:

  - learned:  the trained RL splitter
  - rule:     split at every punctuation mark (excluding the final one), capped at
              --max-segments boundaries
  - random:   per prompt, the same number of boundaries as the learned splitter,
              placed at random punctuation positions (isolates *where* from *how many*)

Reported: ROC-AUC over candidate pairs, top-1 reranking accuracy, and how many
segments each segmentation produces.

Example:
  python benchmarks/eval_splitter_pair_separation.py \
    --dataset data/lmarena_rl/eval_stream.parquet --n 2000 --k 10 \
    --splitter-checkpoint ../rl-training-algorithm/checkpoints/lmarena_heldout_ckpt/epoch=25-step=4160.ckpt
"""
# [새 파일] 캐시 정책 없이 "유사도 점수"만 비교하는 실험 스크립트.
# 각 프롬프트마다 문장 벡터로 비슷한 후보 k개를 뽑아 (프롬프트, 후보) 쌍을 만들고,
# 쌍이 같은 의미인지(정답 1 / 오답 0) 점수가 얼마나 잘 구분하는지 AUC로 잰다.
# 분할 방식 3가지(학습된 RL / 규칙 / 무작위)를 같은 조건에서 비교해서
# "RL이 고른 자르는 위치가 정말 의미가 있는지"를 확인하는 용도.

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.vcache_core.splitter.RuleSplitter import PUNCT_CHARS, punctuation_positions
from vcache.vcache_policy.strategies.verified_splitter import VerifiedSplitterDecisionPolicy


# 자를 위치(pointers)가 주어지면 조각 벡터 + 전체 문장 벡터를 이어 붙인 텐서를 만든다.
# (원본 MaxSimSplitter의 조각 생성 함수를 그대로 사용 → 분할 위치만 다르고 계산 방식은 동일)
def tensor_from_pointers(enc: dict, pointers: list) -> torch.Tensor:
    sent, full = MaxSimSplitter._segment_embeds_from_pointers(enc["token_emb"], enc["length"], pointers)
    return torch.cat([sent, full], dim=0).float().cpu()


def main() -> None:
    # ---- 명령줄 옵션 ----
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)  # 평가할 데이터 파일
    p.add_argument("--label-col", default="ID_Set")  # 같은 의미 묶음 번호 열 (LmArena, SearchQueries)
    p.add_argument(
        "--response-col",
        default=None,
        help="If set, a pair is correct when the responses match under the same static string "
        "comparison as --similarity-evaluator string (use for datasets without ID_Set).",
    )
    p.add_argument("--n", type=int, default=2000, help="Prompts taken from the start of the dataset.")
    p.add_argument("--k", type=int, default=10, help="Candidates per prompt (single-vector top-k).")
    p.add_argument("--splitter-checkpoint", required=True)  # 학습된 RL 분할기 체크포인트
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-segments", type=int, default=4)  # 최대 자르는 횟수
    p.add_argument("--seed", type=int, default=0)  # 무작위 분할용 난수 시드
    p.add_argument("--output-json", default=None)  # 결과 저장 경로 (없으면 화면 출력만)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- 데이터 읽기, 정답 판정 함수 정하기 ----
    # 데이터셋 앞에서부터 n개만 사용
    df = pd.read_parquet(args.dataset).iloc[: args.n].reset_index(drop=True)
    prompts = df["prompt"].astype(str).tolist()
    if args.response_col:
        # 응답 열을 주면: 두 응답 문자열이 같으면 정답 쌍 (Classification처럼 ID_Set이 없는 데이터용)
        responses = df[args.response_col].astype(str).tolist()

        def same(i: int, j: int) -> int:
            return int(answers_have_same_meaning_static(responses[i], responses[j]))
    else:
        # 아니면: 같은 ID_Set 묶음이면 정답 쌍
        labels = df[args.label_col].to_numpy()

        def same(i: int, j: int) -> int:
            return int(labels[i] == labels[j])

    # ---- 모델 불러오기 ----
    embedder = EmbeddingModel(device=args.device)  # BGE 임베딩 모델
    # 학습된 RL 분할기. 전체 문장 벡터를 마지막 행으로 포함
    splitter = MaxSimSplitter(
        checkpoint_path=args.splitter_checkpoint,
        device=args.device,
        embedding_model=embedder,
        max_segments=args.max_segments,
        include_full_embedding=True,
    )

    # 구두점 토큰 번호 목록 (규칙/무작위 분할에서 자를 수 있는 후보)
    tok = splitter.generator.tokenizer
    punct_ids = {i for i in tok.convert_tokens_to_ids(PUNCT_CHARS) if i != tok.unk_token_id}

    # ---- 모든 프롬프트를 임베딩하고 세 가지 방식으로 분할 ----
    variants = ("learned", "rule", "random")  # 학습된 RL / 규칙 / 무작위
    pooled_knn, pooled_no_cls = [], []  # 후보 검색용 문장 벡터, 전체 문장 코사인용 벡터
    tensors = {v: [] for v in variants}  # 방식별 조각 텐서 목록
    n_segments = {v: [] for v in variants}  # 방식별 조각 수 (통계용)
    for text in tqdm(prompts, desc="encode+split"):
        enc = splitter.encode_text(text)  # 문장을 한 번 임베딩 (토큰 벡터 + 문장 벡터)
        pooled_knn.append(enc["pooled_knn"].float().cpu())
        pooled_no_cls.append(enc["pooled_no_cls"].float().cpu())

        # 1) RL 분할기가 고른 위치로 분할
        learned = splitter.split_text_return_maxsim_tensor_from_encoded(enc).float().cpu()
        # 2) 구두점 위치 전체
        positions = punctuation_positions(enc["input_ids"], enc["length"], punct_ids)
        # 3) 무작위: RL과 "같은 개수"만큼 자르되 위치는 구두점 중에서 무작위로 고른다
        #    → 개수가 아니라 "어디를 자르는지"의 효과만 비교하기 위함
        #    (learned 행 수 - 2 = 자른 횟수. 조각 수 = 행 수 - 1(전체 행), 자른 횟수 = 조각 수 - 1)
        n_boundaries = min(max(int(learned.shape[0]) - 2, 0), len(positions))
        random_ptrs = sorted(rng.choice(positions, size=n_boundaries, replace=False).tolist()) if n_boundaries else []
        by_variant = {
            "learned": learned,
            # 규칙: 구두점 앞에서부터 최대 max_segments개 위치에서 자름
            "rule": tensor_from_pointers(enc, positions[: args.max_segments]),
            "random": tensor_from_pointers(enc, random_ptrs),
        }
        for v in variants:
            tensors[v].append(by_variant[v])
            n_segments[v].append(int(by_variant[v].shape[0]) - 1)  # last row is the full embedding

    # ---- 후보 뽑기: 문장 벡터 코사인으로 각 프롬프트의 top-k 이웃 ----
    knn = F.normalize(torch.stack(pooled_knn), dim=-1)  # 길이 1로 정규화 → 내적 = 코사인
    sims = knn @ knn.T  # [N, N] 모든 쌍의 코사인
    sims.fill_diagonal_(-2.0)  # 자기 자신은 후보에서 제외
    k = min(args.k, len(prompts) - 1)
    topk = torch.topk(sims, k=k, dim=1).indices  # 프롬프트마다 가장 비슷한 k개의 번호

    # 실제 vCache 정책이 쓰는 함수를 그대로 가져와 점수 계산 (평가와 같은 공식 보장)
    maxsim_fn = VerifiedSplitterDecisionPolicy._maxsim_from_tensors
    cos01_fn = VerifiedSplitterDecisionPolicy._cos01

    # ---- 모든 (프롬프트, 후보) 쌍의 점수 계산 ----
    # 점수 종류 7개: vcache(문장 코사인) + 분할 방식 3개 × (mvr = 섞은 점수, maxsim = 순수 MaxSim)
    names = ["vcache"] + [f"{kind}_{v}" for v in variants for kind in ("mvr", "maxsim")]
    y = []  # 쌍별 정답(1)/오답(0)
    scores = {name: [] for name in names}  # 점수 종류별 쌍 점수
    top1 = {name: 0 for name in names}  # 점수 1등 후보가 정답이었던 횟수
    n_with_pos = 0  # 후보 중 정답이 하나라도 있는 프롬프트 수
    for i in tqdm(range(len(prompts)), desc="score pairs"):
        cand = topk[i].tolist()  # i번 프롬프트의 후보 k개
        row_y = [same(i, j) for j in cand]  # 후보별 정답 여부
        fullcos = [cos01_fn(pooled_no_cls[i], pooled_no_cls[j]) for j in cand]  # 전체 문장 코사인 (0~1)
        rows = {"vcache": [float((sims[i, j] + 1.0) * 0.5) for j in cand]}  # vCache 기준선 점수 (0~1로 변환)
        for v in variants:
            ms = [maxsim_fn(tensors[v][i], tensors[v][j]) for j in cand]  # MaxSim 점수
            rows[f"maxsim_{v}"] = ms
            rows[f"mvr_{v}"] = [0.5 * (m + c) for m, c in zip(ms, fullcos)]  # MaxSim과 문장 코사인 반반
        y += row_y
        for name in names:
            scores[name] += rows[name]
        if any(row_y):
            # 정답 후보가 있을 때만 "1등으로 고른 후보가 정답인가"를 센다 (재순위 정확도)
            n_with_pos += 1
            for name in names:
                top1[name] += row_y[int(np.argmax(rows[name]))]

    # ---- 결과 정리 ----
    y_arr = np.array(y)
    result = {
        "n_prompts": len(prompts),
        "k": k,
        "n_pairs": int(len(y_arr)),
        "positive_pair_rate": round(float(y_arr.mean()), 4),
        # AUC: 점수가 정답 쌍과 오답 쌍을 얼마나 잘 구분하는지 (0.5 = 무작위, 1.0 = 완벽)
        "auc": {name: round(float(roc_auc_score(y_arr, scores[name])), 4) for name in names},
        # top-1 정확도: 점수 1등 후보가 정답인 비율
        "top1_accuracy": {name: round(v / max(n_with_pos, 1), 4) for name, v in top1.items()},
        "n_prompts_with_positive_candidate": n_with_pos,
        "segments_per_prompt": {
            v: {str(n): c for n, c in sorted(Counter(n_segments[v]).items())} for v in variants
        },
        "mean_segments": {v: round(float(np.mean(n_segments[v])), 3) for v in variants},
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
