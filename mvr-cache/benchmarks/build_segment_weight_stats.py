"""Build the statistics used by RulePunctuationSplitter segment weighting (Step 3a/3b).

Reads a training split that lies outside the evaluation stream (Classification:
train/train3k.parquet; evaluation uses test/test41k.parquet) and writes one file with:

  - idf:       [vocab] tensor, log((N + 1) / (df + 1)) + 1 over the training prompts
  - centroid:  [H] mean of L2-normalised rule-split segment embeddings
  - mlp_state: SegmentWeightMLP trained with BCE so that the weighted MaxSim score (the
               eval score with --candidate-selection multivector_top_k --include-full-embedding;
               --mix-fullcos has no effect there) separates correct from incorrect pairs.
               Pass --mix-fullcos here to reproduce the older, fullcos-mixed training score.

Candidate pairs mirror the cache: each prompt's top-k neighbours by single-vector cosine.
Anchors are split 80/20 so the MLP is scored on pairs it was not trained on.

Example:
  python benchmarks/build_segment_weight_stats.py \
    --dataset train/train3k.parquet --response-col response_llama_3_8b \
    --out results/classification_train3k_weight_stats.pt
"""
# [새 파일] 조각 가중치에 필요한 통계를 만드는 스크립트 (Step 3a, 3b, 5에서 사용하는 .pt 파일 생성)
#   - idf      : 토큰별 IDF 표 (라벨 사용 안 함)
#   - centroid : 조각 벡터들의 평균 방향 (라벨 사용 안 함)
#   - mlp      : 조각 벡터 → 가중치를 내는 작은 신경망 (응답 일치 라벨로 BCE 학습)
#
# 입력은 평가 구간과 겹치지 않는 학습 파일(train/train3k.parquet)을 통째로 쓴다 (기본 --start 0, --n 전체).
# 학습 점수는 실제 평가(multivector_top_k)와 같은 순수 가중 MaxSim. 옛 50% 혼합 방식은 --mix-fullcos.

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.RuleSplitter import (
    MIN_WEIGHT,
    RulePunctuationSplitter,
    SegmentWeightMLP,
)


# 여러 쌍 (ai[k], bi[k])의 점수를 한 번에(배치로) 계산.
# 평가의 _maxsim_from_tensors와 같은 가중 MaxSim을 텐서 연산으로 구현한 것.
#
#   s = ( (row + col) / 2 + 1 ) / 2
#   row = Σ_i w_i · max_j cos(q_i, c_j) / Σ_i w_i      (질문 조각 i 기준)
#   col = Σ_j w_j · max_i cos(q_i, c_j) / Σ_j w_j      (후보 조각 j 기준)
#   (q, c의 마지막 행은 전체 문장 벡터, 그 가중치 = 조각 가중치 평균)
#
# [수정] 기본값은 전체 문장 코사인을 섞지 않는다(mix_fullcos=False).
#        실제 평가(multivector_top_k)가 순수 가중 MaxSim이기 때문에 학습도 같은 공식으로 맞춘다.
#        이전 결과(Step 3b·5)를 재현하려면 --mix-fullcos 를 주면 된다.
def pair_scores(rows, mask, full_nocls, ai, bi, seg_w, mix_fullcos=False):
    """Weighted MaxSim for pairs (ai, bi), matching the eval scoring (multivector_top_k).

    rows: [N, S, H] segment rows (last valid row is the full embedding), mask: [N, S] bool,
    seg_w: [N, S] weights for every row (full-row weight already set to the segment mean).
    mix_fullcos: also average with the full-embedding cosine (the old training score).
    """
    # 각 쌍의 질문 조각들(q)과 후보 조각들(c)을 길이 1로 정규화 → 내적 = 코사인
    q, c = F.normalize(rows[ai], dim=-1), F.normalize(rows[bi], dim=-1)
    qm, cm = mask[ai], mask[bi]  # 실제 조각인 행만 True (패딩 행은 False)
    cos = torch.bmm(q, c.transpose(1, 2))  # [쌍 수, 질문 조각 수, 후보 조각 수] 모든 조각 쌍의 코사인
    # 패딩 행은 최댓값에 뽑히지 않도록 -2로 채움
    cos = cos.masked_fill(~cm.unsqueeze(1), -2.0).masked_fill(~qm.unsqueeze(2), -2.0)
    # row_max: 질문 조각마다 후보에서 가장 비슷한 조각과의 코사인, col_max: 그 반대 방향
    row_max, col_max = cos.max(dim=2).values, cos.max(dim=1).values
    wq, wc = seg_w[ai] * qm, seg_w[bi] * cm  # 패딩 행의 가중치는 0
    # 가중 평균 (분모에 1e-8을 더해 0으로 나누기 방지)
    row = (row_max.clamp_min(-1) * wq).sum(1) / (wq.sum(1) + 1e-8)
    col = (col_max.clamp_min(-1) * wc).sum(1) / (wc.sum(1) + 1e-8)
    # 두 방향 평균 → -1~1 범위를 0~1로 변환. 실제 평가 점수와 같은 순수 가중 MaxSim
    maxsim01 = ((0.5 * (row + col) + 1.0) * 0.5).clamp(0, 1)
    if not mix_fullcos:
        return maxsim01
    # (옛 방식, --mix-fullcos일 때만) 전체 문장 코사인을 0~1로 변환해 MaxSim과 반반 섞는다
    fullcos01 = (
        (F.cosine_similarity(full_nocls[ai], full_nocls[bi], dim=-1) + 1.0) * 0.5
    ).clamp(0, 1)
    return 0.5 * (maxsim01 + fullcos01)


# 모든 프롬프트의 모든 행에 대해 MLP 가중치를 계산 (평가 때 RuleSplitter가 하는 것과 같은 규칙)
def rows_weights(mlp, rows, mask, n_seg):
    """MLP weights for segment rows; the full row gets the mean segment weight."""
    w = mlp(rows)  # [N, S] 행마다 MLP 출력
    idx = torch.arange(rows.shape[1]).unsqueeze(0)  # 행 번호 0, 1, 2, ...
    seg_mask = idx < n_seg.unsqueeze(1)  # 조각 행인지 (전체 문장 행 앞쪽)
    seg_mean = (w * seg_mask).sum(1) / seg_mask.sum(1).clamp_min(1)  # 조각 가중치 평균
    full_pos = idx == n_seg.unsqueeze(1)  # 전체 문장 행 위치 (조각 바로 다음 행)
    w = torch.where(full_pos, seg_mean.unsqueeze(1), w)  # 전체 문장 행 가중치 = 조각 평균
    return torch.where(mask, w, torch.zeros_like(w))  # 패딩 행은 0


def main() -> None:
    # ---- 명령줄 옵션 ----
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)  # 데이터 파일
    # [수정] 기본값을 "파일 전체"로 변경 (예전 기본 10000~12999행은 45k 원본 기준이라 train3k에서는 0행이 됨)
    p.add_argument("--start", type=int, default=0, help="First row to use (default: 0).")
    p.add_argument("--n", type=int, default=None, help="Rows to use from --start (default: all).")
    p.add_argument("--label-col", default="ID_Set")  # 같은 의미 묶음 열 (response-col이 없을 때 정답 기준)
    p.add_argument("--response-col", default=None)  # 응답 열. 주면 "응답이 같으면 정답 쌍"
    p.add_argument("--k", type=int, default=10)  # 프롬프트마다 이웃 후보 수 (학습 쌍 만들 때)
    p.add_argument("--mlp-hidden", type=int, default=128)  # MLP 중간층 크기
    p.add_argument("--epochs", type=int, default=40)  # MLP 학습 반복 수. 0이면 MLP 없이 idf/centroid만 저장
    p.add_argument("--lr", type=float, default=1e-3)  # 학습률
    # [추가] 학습 점수에 전체 문장 코사인 50%를 섞을지. 기본은 안 섞음(평가와 동일한 순수 가중 MaxSim)
    p.add_argument("--mix-fullcos", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)  # 저장할 .pt 파일 경로
    args = p.parse_args()
    torch.manual_seed(args.seed)  # MLP 초기값·배치 순서 고정
    rng = np.random.default_rng(args.seed)  # 학습/검증 분리용 난수

    # ---- 데이터 읽기 ----

    df = (
        pd.read_parquet(args.dataset)
        .iloc[args.start : (None if args.n is None else args.start + args.n)]
        .reset_index(drop=True)
    )
    prompts = df["prompt"].astype(str).tolist()
    if args.response_col:
        # 정답 기준: 두 응답 문자열이 같으면 1 (평가의 --similarity-evaluator string과 같은 비교)
        responses = df[args.response_col].astype(str).tolist()

        def same(i, j):
            return int(answers_have_same_meaning_static(responses[i], responses[j]))
    else:
        # 정답 기준: 같은 ID_Set이면 1
        labels = df[args.label_col].to_numpy()

        def same(i, j):
            return int(labels[i] == labels[j])

    # ---- 모든 프롬프트 임베딩 + 규칙 분할 ----
    embedder = EmbeddingModel(device=args.device)
    # 평가와 같은 규칙 분할기(punctuation_rules.py)로 조각을 만든다
    splitter = RulePunctuationSplitter(
        device=args.device,
        embedding_model=embedder,
        include_full_embedding=True,
    )

    vocab = len(splitter.embedding_model.tokenizer)  # 어휘 크기
    doc_freq = torch.zeros(vocab)  # 토큰별 "등장한 문장 수" (문서 빈도)
    tensors, knn, nocls = [], [], []
    seg_tokens = []  # [로그 E] 프롬프트별 조각 토큰 수 (가중치 분포를 길이별로 보기 위함)
    for text in tqdm(prompts, desc="encode+split"):
        enc = splitter.encode_text(text)  # 토큰 번호, 토큰 벡터, 문장 벡터
        # 이 문장에 나온 토큰 종류마다 1씩 증가 (한 문장에 여러 번 나와도 1번만 셈)
        doc_freq[torch.unique(enc["input_ids"][: enc["length"]].cpu())] += 1
        tensors.append(splitter.split_text_return_maxsim_tensor_from_encoded(enc).cpu())  # 조각 + 전체 행
        seg_tokens.append(splitter.describe_split(enc)["segment_tokens"])
        knn.append(enc["pooled_knn"].float().cpu())  # 이웃 검색용 문장 벡터
        nocls.append(enc["pooled_no_cls"].float().cpu())  # 전체 문장 코사인용 벡터

    n = len(prompts)
    # IDF = log((문장 수+1)/(등장 문장 수+1)) + 1 → 드문 토큰일수록 큰 값
    idf = torch.log((n + 1) / (doc_freq + 1)) + 1.0
    # 모든 조각 벡터(마지막 전체 문장 행 제외)를 정규화해서 평균 → 코퍼스 중심 방향
    seg_rows = torch.cat([F.normalize(t[:-1], dim=-1) for t in tensors])
    centroid = seg_rows.mean(0)

    if args.epochs == 0:
        # idf/centroid only. The MLP path builds an N x N similarity matrix, which is far
        # too large when the reference corpus is a whole dataset (45k prompts -> 8 GB).
        torch.save(
            {
                "idf": idf,
                "centroid": centroid,
                "meta": {
                    "dataset": args.dataset,
                    "start": args.start,
                    "n": n,
                    "max_segments": "all punctuation",
                    "mlp": False,
                    "min_weight": MIN_WEIGHT,
                },
            },
            args.out,
        )
        print(f"saved (idf + centroid, no MLP) -> {args.out}")
        return

    # Pad rows: [N, S, H]; last valid row of each prompt is its full embedding.
    # ---- MLP 학습 준비: 프롬프트마다 행 수가 달라서 가장 긴 것에 맞춰 0으로 채운 3차원 텐서로 만든다 ----
    s_max = max(t.shape[0] for t in tensors)
    H = tensors[0].shape[1]
    rows = torch.zeros(n, s_max, H)
    mask = torch.zeros(n, s_max, dtype=torch.bool)
    n_seg = torch.tensor([t.shape[0] - 1 for t in tensors])
    for i, t in enumerate(tensors):
        rows[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    full_nocls = torch.stack(nocls)

    # ---- 학습 쌍 만들기: 캐시처럼 각 프롬프트의 문장 벡터 top-k 이웃과 짝 ----
    knn_n = F.normalize(torch.stack(knn), dim=-1)
    sims = knn_n @ knn_n.T  # N×N 코사인 (n이 크면 메모리를 많이 씀 → 45k는 --epochs 0으로)
    sims.fill_diagonal_(-2.0)  # 자기 자신 제외
    topk = torch.topk(sims, k=min(args.k, n - 1), dim=1).indices
    # 기준 프롬프트(anchor) 단위로 80% 학습 / 20% 검증 분리 (같은 기준의 쌍이 양쪽에 섞이지 않게)
    anchors = rng.permutation(n)
    n_train = int(0.8 * n)
    split = {"train": anchors[:n_train], "val": anchors[n_train:]}
    pairs = {}
    for name, idx in split.items():
        ai = torch.tensor([a for a in idx for _ in range(topk.shape[1])])
        bi = topk[torch.tensor(idx)].reshape(-1)
        y = torch.tensor(
            [same(int(a), int(b)) for a, b in zip(ai, bi)], dtype=torch.float32
        )
        pairs[name] = (ai, bi, y)
        print(f"{name}: {len(y)} pairs, positive rate {y.mean():.3f}")

    # [로그 E] 학습 쌍 통계: 쌍 수, 양성 비율, 과제별 쌍 수·양성 비율
    task_col = df["dataset_name"].tolist() if "dataset_name" in df.columns else None
    train_log = {"args": vars(args), "pairs": {}, "epochs": []}
    for name, (a_, _, y_) in pairs.items():
        info = {"n": int(len(y_)), "positive_rate": round(float(y_.mean()), 4)}
        if task_col is not None:
            by = {}
            for t in sorted(set(task_col)):
                m = torch.tensor([task_col[int(i)] == t for i in a_])
                if m.any():
                    by[t] = {"n": int(m.sum()), "positive_rate": round(float(y_[m].mean()), 4)}
            info["by_task"] = by
        train_log["pairs"][name] = info

    # ---- MLP 학습 ----
    mlp = SegmentWeightMLP(dim=H, hidden=args.mlp_hidden)
    # 점수(0~1)를 로짓으로 바꾸는 보정 파라미터 a, b도 함께 학습 (vCache의 로지스틱과 비슷한 역할)
    calib = torch.nn.Parameter(torch.tensor([10.0, -5.0]))  # logit = a * score + b
    opt = torch.optim.Adam(list(mlp.parameters()) + [calib], lr=args.lr)
    ai, bi, y = pairs["train"]
    # 정답 쌍이 많으면(불균형) 오답을 잘 못 배우므로, 정답 쌍 손실에 (오답 수/정답 수) 배를 곱해 균형을 맞춘다
    pos_weight = (1 - y.mean()) / y.mean().clamp_min(1e-6)
    va, vb, vy = pairs["val"]
    best = {"epoch": -1, "val_auc": -1.0, "state": None}
    for epoch in range(args.epochs):
        perm = torch.randperm(len(y))  # 매 에폭 쌍 순서 섞기
        total, grad_norms = 0.0, []
        for start in range(0, len(y), 512):  # 512쌍씩 배치
            b = perm[start : start + 512]
            # 전체 프롬프트의 가중치를 매번 다시 계산 (정확하지만 느림)
            w = rows_weights(mlp, rows, mask, n_seg)
            # 그 가중치로 점수 계산 → MLP는 정답을 직접 맞히는 게 아니라, 점수가 정답 쌍을 가르도록 간접 학습
            s = pair_scores(rows, mask, full_nocls, ai[b], bi[b], w, args.mix_fullcos)
            loss = F.binary_cross_entropy_with_logits(
                calib[0] * s + calib[1], y[b], pos_weight=pos_weight
            )
            opt.zero_grad()  # 이전 기울기 초기화
            loss.backward()  # 기울기 계산
            # [로그 E] 기울기 크기 (업데이트 전)
            grad_norms.append(float(torch.norm(torch.stack([
                p.grad.norm() for p in list(mlp.parameters()) + [calib] if p.grad is not None
            ]))))
            opt.step()  # MLP와 a, b 업데이트
            total += loss.item() * len(b)

        # [로그 E] 에폭마다 val BCE·AUC 계산, val AUC가 가장 높은 에폭의 MLP를 최종으로 선택
        with torch.no_grad():
            w = rows_weights(mlp, rows, mask, n_seg)
            vs = pair_scores(rows, mask, full_nocls, va, vb, w, args.mix_fullcos)
            val_bce = float(F.binary_cross_entropy_with_logits(calib[0] * vs + calib[1], vy, pos_weight=pos_weight))
            val_auc = float(roc_auc_score(vy.numpy(), vs.numpy()))
        train_log["epochs"].append({
            "epoch": epoch,
            "train_bce": round(total / len(y), 5),
            "val_bce": round(val_bce, 5),
            "val_auc": round(val_auc, 5),
            "lr": opt.param_groups[0]["lr"],
            "grad_norm_mean": round(float(np.mean(grad_norms)), 5),
            "grad_norm_max": round(float(np.max(grad_norms)), 5),
            "calib": [round(float(v), 4) for v in calib.detach()],
        })
        if val_auc > best["val_auc"]:
            best = {"epoch": epoch, "val_auc": val_auc,
                    "state": {k: v.detach().clone() for k, v in mlp.state_dict().items()}}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: train BCE {total / len(y):.4f}  val BCE {val_bce:.4f}  val AUC {val_auc:.4f}")

    if best["state"] is not None:
        mlp.load_state_dict(best["state"])
    train_log["selected_epoch"] = best["epoch"]
    train_log["selected_val_auc"] = round(best["val_auc"], 5)
    print(f"selected epoch {best['epoch']} (val AUC {best['val_auc']:.4f})")

    # ---- 학습 결과 확인: 균등 가중치 vs MLP 가중치의 AUC 비교 (학습/검증 쌍 각각) ----
    uniform_w = torch.where(mask, torch.ones(n, s_max), torch.zeros(n, s_max))
    with torch.no_grad():
        mlp_w = rows_weights(mlp, rows, mask, n_seg)
    for name, (a, b, yy) in pairs.items():
        auc_u = roc_auc_score(
            yy.numpy(), pair_scores(rows, mask, full_nocls, a, b, uniform_w, args.mix_fullcos).numpy()
        )
        auc_m = roc_auc_score(
            yy.numpy(), pair_scores(rows, mask, full_nocls, a, b, mlp_w, args.mix_fullcos).numpy()
        )
        print(f"{name} AUC  uniform {auc_u:.4f}  mlp {auc_m:.4f}")
        train_log.setdefault("final_auc", {})[name] = {"uniform": round(float(auc_u), 5), "mlp": round(float(auc_m), 5)}

    # [로그 E] 학습된 가중치 분포: 조각 위치별(처음/중간/끝), 길이별 평균, 프롬프트별 엔트로피
    by_pos, by_len, entropies = {"first": [], "middle": [], "last": [], "only": []}, {}, []
    for i in range(n):
        k_seg = int(n_seg[i])
        wi = mlp_w[i, :k_seg].numpy()
        for j, wv in enumerate(wi):
            pos = "only" if k_seg == 1 else "first" if j == 0 else "last" if j == k_seg - 1 else "middle"
            by_pos[pos].append(float(wv))
            ln = seg_tokens[i][j] if j < len(seg_tokens[i]) else 0
            bucket = "1" if ln <= 1 else "2-3" if ln <= 3 else "4-7" if ln <= 7 else "8-15" if ln <= 15 else "16+"
            by_len.setdefault(bucket, []).append(float(wv))
        if k_seg > 1:
            pr = wi / wi.sum()
            entropies.append(float(-(pr * np.log(pr + 1e-12)).sum() / np.log(k_seg)))  # 1 = 균등, 0 = 한 조각에 몰림
    train_log["weight_distribution"] = {
        "by_position_mean": {k: round(float(np.mean(v)), 5) for k, v in by_pos.items() if v},
        "by_length_mean": {k: round(float(np.mean(v)), 5) for k, v in sorted(by_len.items())},
        "normalized_entropy_mean": round(float(np.mean(entropies)), 5) if entropies else None,
    }
    log_path = args.out + ".train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=1, default=str)
    print(f"training log -> {log_path}")

    # ---- 저장: idf, centroid, MLP 파라미터, 설정 정보(meta) ----
    torch.save(
        {
            "idf": idf,
            "centroid": centroid,
            "mlp_state": mlp.state_dict(),
            "mlp_hidden": args.mlp_hidden,
            "meta": {
                "dataset": args.dataset,
                "start": args.start,
                "n": n,
                "k": args.k,
                "max_segments": "all punctuation",
                "seed": args.seed,
                "mix_fullcos": bool(args.mix_fullcos),
                "selected_epoch": train_log.get("selected_epoch"),
                "min_weight": MIN_WEIGHT,
            },
        },
        args.out,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
