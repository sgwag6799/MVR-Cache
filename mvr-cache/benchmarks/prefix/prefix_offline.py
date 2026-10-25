"""Stage 1: offline pair diagnostics for the shared-prefix stress test (GPU).

For each level L (and the fixed-prefix reference) it encodes train / val / test with
bge-base-en-v1.5, cuts every prompt with the rules in prefix_split.py, and scores the
same candidate pairs four ways:

  M1  single vector: cosine of the mean of all tokens incl. [CLS] (what vCache uses), (x+1)/2
  M2  MVR, uniform weights: symmetric SMaxSim over segments, (x+1)/2, no full-sentence row
  M3  MVR, learned weights: Linear(768 -> 1), softmax over the segments of a prompt, zero init,
      trained per level on that level's train split (BCE, positive-class weighting,
      Adam 1e-3, batch 2048, 20 epochs, best val AUC). It never sees which segment is prefix
  M4  MVR, prefix segments weight 0 (upper bound for any weighting)

Exploratory, label-free weightings (only with --extra-conditions; same formulas as the
team RuleSplitter on gpu-production, statistics from that level's train split):
  M5  idf:      mean IDF of the segment's tokens, IDF = log((N+1)/(df+1)) + 1
  M6  centroid: 1 - cos(segment, mean of L2-normalised train segment vectors), min 1e-3

Pairs: test anchors (6,000, seed 0, same rows at every level) x 10 nearest neighbours by
the M1 vector over the test split, searched again at every level. Only rows that fit in
512 tokens at the largest level are used (same rows everywhere, so no level truncates).

Metrics per level x condition x task (ALL + each sub-task):
  ceiling@10      share of anchors with at least one correct candidate (same for every condition)
  top1_acc        the condition's #1 candidate is correct
  auc_selected    AUC of (s, c) of each anchor's #1 candidate     <- gate metric
  auc_pairs       AUC over all anchor x candidate pairs
  tpr_at_fpr1     TPR at 1% FPR over all pairs (team proxy)

Outputs in --out: prefix_offline.csv, prefix_w_<mode>_L<L>.pt, prefix_gate.json, run meta.

Example (Colab, L4 GPU):
  python -m benchmarks.prefix.prefix_offline \
    --data /content/drive/MyDrive/mvr_eval/prefix/data \
    --out  /content/drive/MyDrive/mvr_eval/prefix/offline
"""
# [새 파일] 1단계: 오프라인 쌍 진단 (단계 × 조건 M1~M4 × 과제)

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.prefix._shared import ps

CONDITIONS = ("M1", "M2", "M3", "M4")
EXTRA_CONDITIONS = ("M5", "M6")  # 탐색용: M5 idf, M6 centroid (라벨 미사용)
MIN_WEIGHT = 1e-3  # gpu-production RuleSplitter와 같은 하한


# ----------------------------------------------------------------------------- 인코딩


@torch.no_grad()
def encode_split(df, tok, lm, device, max_cuts, cap_scope, batch_size=64, fp16=True):
    """Segment vectors (list of [S,H] fp16 CPU), prefix flags (list of bool tensors),
    single vectors [N,H] (fp32, L2-normalised), per-segment token ids (list of lists of tensors)."""
    texts = df["prompt"].astype(str).tolist()
    n_chars = df["prefix_chars"].astype(int).tolist()
    order = np.argsort([len(t) for t in texts])  # 길이순으로 묶어 패딩 낭비를 줄임
    segs, flags, knn = [None] * len(texts), [None] * len(texts), [None] * len(texts)
    seg_ids = [None] * len(texts)  # M5(idf)용 조각별 토큰 번호
    for b in range(0, len(texts), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=ps.MAX_LEN, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.startswith("cuda")):
            hs = lm(**enc).last_hidden_state
        hs = hs.float()
        for j, i in enumerate(idx):
            length = int(enc["attention_mask"][j].sum())
            ids = enc["input_ids"][j][:length].tolist()
            tokens = tok.convert_ids_to_tokens(ids)
            off = [tuple(o) for o in offsets[j][:length]]
            pend = ps.prefix_token_end(off, n_chars[i])
            cuts = ps.cut_points(tokens, length, max_cuts=max_cuts, cap_scope=cap_scope, prefix_end=pend)
            spans = ps.segment_spans(length, cuts)
            h = hs[j]
            segs[i] = torch.stack([h[s:e].mean(0) for s, e in spans]).half().cpu()
            flags[i] = torch.tensor(ps.prefix_segment_flags(spans, pend))
            knn[i] = h[:length].mean(0).cpu()  # [CLS]·[SEP] 포함 평균 = EmbeddingModel.get_embedding
            seg_ids[i] = [torch.tensor(ids[s:e]) for s, e in spans]
    return segs, flags, F.normalize(torch.stack(knn), dim=-1), seg_ids


def idf_table(seg_ids, vocab):
    """IDF over prompts (each prompt counts a token once): log((N+1)/(df+1)) + 1."""
    doc_freq = torch.zeros(vocab)
    for row in seg_ids:
        doc_freq[torch.unique(torch.cat(row))] += 1
    return torch.log((len(seg_ids) + 1) / (doc_freq + 1)) + 1.0


def fixed_weights(cond, segs, seg_ids, idf, centroid):
    """Label-free per-segment weights (unnormalised) for M5 / M6, one tensor per prompt."""
    if cond == "M5":
        return [torch.stack([idf[t].mean() for t in row]).clamp_min(MIN_WEIGHT) for row in seg_ids]
    return [(1.0 - F.cosine_similarity(s.float(), centroid.unsqueeze(0), dim=-1)).clamp_min(MIN_WEIGHT)
            for s in segs]


def neighbours(knn, query_idx, k, device, chunk=2048):
    """Top-k by cosine over all rows of `knn`, excluding the query itself. [len(query_idx), k]."""
    bank = knn.to(device)
    out = []
    for b in range(0, len(query_idx), chunk):
        q = torch.as_tensor(query_idx[b:b + chunk])
        sim = bank[q.to(device)] @ bank.T
        sim[torch.arange(len(q)), q.to(device)] = -2.0  # 자기 자신 제외
        out.append(sim.topk(k, dim=1).indices.cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- 점수


# 한 단계 안에서 같은 bank를 수백 번 패딩하지 않도록 bank 전체를 한 번만 GPU에 패딩해 둔다.
# 단계가 바뀔 때 clear_pad_cache()로 비운다 (id 재사용 방지).
_PAD_CACHE: dict = {}


def clear_pad_cache():
    _PAD_CACHE.clear()


def _padded_bank(seg_list, flag_list, device):
    key = (id(seg_list), device)
    if key not in _PAD_CACHE:
        n, s_max, h = len(seg_list), max(r.shape[0] for r in seg_list), seg_list[0].shape[1]
        x = torch.zeros(n, s_max, h, dtype=seg_list[0].dtype)
        m = torch.zeros(n, s_max, dtype=torch.bool)
        pf = torch.zeros(n, s_max, dtype=torch.bool)
        lens = torch.zeros(n, dtype=torch.long)
        for i, r in enumerate(seg_list):
            x[i, : r.shape[0]] = r
            m[i, : r.shape[0]] = True
            pf[i, : r.shape[0]] = flag_list[i]
            lens[i] = r.shape[0]
        _PAD_CACHE[key] = (x.to(device), m.to(device), pf.to(device), lens.to(device))
    return _PAD_CACHE[key]


def pad(seg_list, flag_list, idx, device):
    x, m, pf, lens = _padded_bank(seg_list, flag_list, device)
    j = torch.as_tensor(np.asarray(idx), device=device)
    s = int(lens[j].max())  # 배치 안 최대 조각 수까지만 (예전 배치별 패딩과 같은 모양)
    return x[j, :s].float(), m[j, :s], pf[j, :s]


class SegmentHead(nn.Module):
    """M3: Linear(H -> 1), zero init, softmax over the segments of each prompt."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.lin = nn.Linear(dim, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, m):
        logits = self.lin(x).squeeze(-1).masked_fill(~m, float("-inf"))
        return torch.softmax(logits, dim=-1)


def pad_weights(w_list, idx, s_max, device):
    key = (id(w_list), device)
    if key not in _PAD_CACHE:
        w = torch.zeros(len(w_list), max(r.shape[0] for r in w_list))
        for i, r in enumerate(w_list):
            w[i, : r.shape[0]] = r
        _PAD_CACHE[key] = w.to(device)
    j = torch.as_tensor(np.asarray(idx), device=device)
    return _PAD_CACHE[key][j, :s_max]


def weights(cond, x, m, pf, head, fw=None):
    if cond == "M3":
        return head(x, m)
    if cond in EXTRA_CONDITIONS:
        return fw / fw.sum(1, keepdim=True)  # 패딩 칸은 0이므로 조각끼리 합 1
    keep = m & ~pf if cond == "M4" else m
    keep = torch.where(keep.any(1, keepdim=True), keep, m)  # 전부 공통 문단이면 균등으로 (원본이 있어 실제로는 없음)
    w = keep.float()
    return w / w.sum(1, keepdim=True)


def smaxsim01(qx, qm, qw, cx, cm, cw):
    """Symmetric weighted SMaxSim -> [0, 1] (same as VerifiedSplitterDecisionPolicy._maxsim_from_tensors)."""
    qn, cn = F.normalize(qx, dim=-1), F.normalize(cx, dim=-1)
    cos = torch.bmm(qn, cn.transpose(1, 2))
    row = cos.masked_fill(~cm.unsqueeze(1), -1e4).max(2).values.masked_fill(~qm, 0.0)
    col = cos.masked_fill(~qm.unsqueeze(2), -1e4).max(1).values.masked_fill(~cm, 0.0)
    raw = 0.5 * (qw * row).sum(1) + 0.5 * (cw * col).sum(1)
    return ((raw + 1.0) * 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def score_pairs(cond, bank_q, bank_c, qi, ci, device, head=None, knn_q=None, knn_c=None, bs=4096,
                fw_q=None, fw_c=None):
    if cond == "M1":
        s = (knn_q[qi] * knn_c[ci]).sum(-1)
        return ((s + 1) * 0.5).numpy()
    out = []
    for b in range(0, len(qi), bs):
        qx, qm, qpf = pad(*bank_q, qi[b:b + bs], device)
        cx, cm, cpf = pad(*bank_c, ci[b:b + bs], device)
        qfw = pad_weights(fw_q, qi[b:b + bs], qx.shape[1], device) if fw_q is not None else None
        cfw = pad_weights(fw_c, ci[b:b + bs], cx.shape[1], device) if fw_c is not None else None
        out.append(smaxsim01(qx, qm, weights(cond, qx, qm, qpf, head, qfw),
                             cx, cm, weights(cond, cx, cm, cpf, head, cfw)).float().cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- 지표


def auc(s, c):
    pos, neg = s[c == 1], s[c == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(s, c, fpr=0.01):
    pos, neg = s[c == 1], s[c == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean())


def metrics(scores, c):
    """scores, c: [A, k] arrays for a set of anchors."""
    top = scores.argmax(1)
    sel_s, sel_c = scores[np.arange(len(top)), top], c[np.arange(len(top)), top]
    return {
        "n_anchors": int(len(scores)),
        "ceiling10": float(c.any(1).mean()),
        "top1_acc": float(sel_c.mean()),
        "auc_selected": auc(sel_s, sel_c),
        "auc_pairs": auc(scores.ravel(), c.ravel()),
        "tpr_at_fpr1": tpr_at_fpr(scores.ravel(), c.ravel()),
    }


# ----------------------------------------------------------------------------- M3 학습


def train_head(bank_tr, pairs_tr, bank_va, pairs_va, device, epochs, lr, batch, seed, dim,
               k=10, select_metric="auc_pairs", patience=0):
    """select_metric: val metric for picking the epoch ('auc_pairs' = all val pairs, the
    original rule; 'auc_selected' = each val row's #1 candidate, same as the gate metric).
    patience: stop after this many epochs without a new best (0 = run all epochs)."""
    torch.manual_seed(seed)
    head = SegmentHead(dim).to(device)
    calib = nn.Parameter(torch.tensor([10.0, -5.0], device=device))  # 점수 → 로짓 (a·s + b)
    opt = torch.optim.Adam(list(head.parameters()) + [calib], lr=lr)
    qi, ci, y = pairs_tr
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    pos_weight = (1 - y_t.mean()) / y_t.mean().clamp_min(1e-6)  # 클래스 비율 보정
    best = {"epoch": -1, "val_auc": -1.0, "state": None}
    log = []
    for ep in range(epochs):
        head.train()
        perm = np.random.default_rng(seed + ep).permutation(len(y))
        tot = 0.0
        for b in range(0, len(perm), batch):
            j = perm[b:b + batch]
            qx, qm, _ = pad(*bank_tr, qi[j], device)
            cx, cm, _ = pad(*bank_tr, ci[j], device)
            s = smaxsim01(qx, qm, head(qx, qm), cx, cm, head(cx, cm))
            loss = F.binary_cross_entropy_with_logits(calib[0] * s + calib[1], y_t[j], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        head.eval()
        with torch.no_grad():
            vs = score_pairs("M3", bank_va, bank_va, pairs_va[0], pairs_va[1], device, head=head)
        vm = metrics(vs.reshape(-1, k), pairs_va[2].reshape(-1, k).astype(int))  # within_pairs는 행마다 k개씩
        va = vm[select_metric]
        log.append({"epoch": ep, "train_bce": tot / len(y), "val_auc": va,
                    "val_auc_pairs": vm["auc_pairs"], "val_auc_selected": vm["auc_selected"]})
        if va > best["val_auc"]:
            best = {"epoch": ep, "val_auc": va, "state": {n: v.detach().clone() for n, v in head.state_dict().items()}}
        elif patience and ep - best["epoch"] >= patience:
            break  # 조기 종료
    head.load_state_dict(best["state"])
    head.eval()
    return head, {"selected_epoch": best["epoch"], "selected_val_auc": best["val_auc"],
                  "select_metric": select_metric, "epochs_run": len(log), "epochs": log}


def within_pairs(knn, rows, labels, k, device):
    """Each row x its k nearest rows (same split), label = same normalised answer."""
    nb = neighbours(knn, rows, k, device)
    qi = np.repeat(rows, k)
    ci = nb.reshape(-1)
    y = (labels[qi] == labels[ci]).astype(np.float32)
    return qi, ci, y


# ----------------------------------------------------------------------------- 메인


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Output folder of make_prefix_data.py")
    p.add_argument("--out", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4, 8])
    p.add_argument("--fixed-levels", type=int, nargs="+", default=[4])
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--anchors", type=int, default=6000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pair-seed", type=int, default=0)
    p.add_argument("--max-cuts", type=int, default=32)
    p.add_argument("--cap-scope", default="original", choices=["original", "all"])
    p.add_argument("--row-filter", default="fits_all_levels", choices=["fits_all_levels", "none"])
    p.add_argument("--resp-col", default="response_llama_3_8b")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", type=int, default=None, help="Use only the first N rows per split (smoke test).")
    p.add_argument("--gate-drop", type=float, default=0.03)
    p.add_argument("--gate-recovery", type=float, default=0.5)
    p.add_argument("--extra-conditions", nargs="*", default=[], choices=EXTRA_CONDITIONS,
                   help="Exploratory label-free weightings to add (M5 idf, M6 centroid).")
    p.add_argument("--select-metric", default="auc_pairs", choices=["auc_pairs", "auc_selected"],
                   help="M3 epoch selection on val: all pairs (original rule) or each row's #1 candidate.")
    p.add_argument("--patience", type=int, default=0,
                   help="M3 early stopping: epochs without a new best val score (0 = off).")
    p.add_argument("--save-scores", action="store_true",
                   help="Save per-pair scores and labels (scores_<mode>_L<L>.npz) for bootstrap CIs.")
    args = p.parse_args()
    conditions = CONDITIONS + tuple(args.extra_conditions)

    from transformers import AutoModel, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModel.from_pretrained(args.model).to(args.device).eval()
    fp16 = not args.no_fp16

    def load(mode, level, split):
        df = pd.read_parquet(os.path.join(args.data, mode, f"L{level}", f"{split}.parquet"))
        if args.row_filter == "fits_all_levels":
            df = df[df["fits_all_levels"]].reset_index(drop=True)
        if args.quick:
            df = df.head(args.quick)
        return df

    runs = [("variant", L) for L in args.levels] + [("fixed", L) for L in args.fixed_levels]
    rows_out, gate_auc, train_logs = [], {}, {}
    for mode, level in runs:
        clear_pad_cache()
        t0 = time.time()
        dfs = {s: load(mode, level, s) for s in ("train", "val", "test")}
        banks, knns, labels, ids = {}, {}, {}, {}
        for s, df in dfs.items():
            segs, flags, knn, seg_ids = encode_split(df, tok, lm, args.device, args.max_cuts, args.cap_scope,
                                                     fp16=fp16)
            banks[s], knns[s], ids[s] = (segs, flags), knn, seg_ids
            labels[s] = df[args.resp_col].map(ps.normalize_answer).to_numpy()
        dim = knns["test"].shape[1]

        # M5·M6 통계는 이 단계의 train에서만 계산 (라벨 미사용)
        fixed_w = {}
        if args.extra_conditions:
            idf = idf_table(ids["train"], len(tok))
            centroid = torch.cat([F.normalize(s.float(), dim=-1) for s in banks["train"][0]]).mean(0)
            for cond in args.extra_conditions:
                fixed_w[cond] = fixed_weights(cond, banks["test"][0], ids["test"], idf, centroid)

        # 테스트 쌍: 앵커는 모든 단계에서 같은 행 (행 순서·필터가 같으므로 같은 위치)
        n_test = len(dfs["test"])
        anchors = np.sort(np.random.default_rng(args.pair_seed).choice(n_test, min(args.anchors, n_test), replace=False))
        nb = neighbours(knns["test"], anchors, args.k, args.device)
        qi, ci = np.repeat(anchors, args.k), nb.reshape(-1)
        c = (labels["test"][qi] == labels["test"][ci]).astype(int).reshape(-1, args.k)

        # M3: 이 단계의 train으로 학습, val로 에폭 선택
        tr_pairs = within_pairs(knns["train"], np.arange(len(dfs["train"])), labels["train"], args.k, args.device)
        va_pairs = within_pairs(knns["val"], np.arange(len(dfs["val"])), labels["val"], args.k, args.device)
        head, tlog = train_head(banks["train"], tr_pairs, banks["val"], va_pairs, args.device,
                                args.epochs, args.lr, args.batch, args.seed, dim,
                                k=args.k, select_metric=args.select_metric, patience=args.patience)
        train_logs[f"{mode}/L{level}"] = {k: v for k, v in tlog.items() if k != "epochs"} | {"epochs": tlog["epochs"]}
        torch.save({"state": head.state_dict(), "dim": dim, "mode": mode, "level": level,
                    "max_cuts": args.max_cuts, "cap_scope": args.cap_scope, "model": args.model,
                    "selected_epoch": tlog["selected_epoch"], "selected_val_auc": tlog["selected_val_auc"]},
                   os.path.join(args.out, f"prefix_w_{mode}_L{level}.pt"))

        tasks = dfs["test"]["dataset_name"].to_numpy()[anchors]
        saved = {}
        for cond in conditions:
            fw = fixed_w.get(cond)
            sc = score_pairs(cond, banks["test"], banks["test"], qi, ci, args.device, head=head,
                             knn_q=knns["test"], knn_c=knns["test"], fw_q=fw, fw_c=fw).reshape(-1, args.k)
            saved[cond] = sc
            for task in ["ALL"] + sorted(set(tasks)):
                m = np.ones(len(tasks), bool) if task == "ALL" else tasks == task
                rows_out.append({"mode": mode, "level": level, "condition": cond, "task": task,
                                 **metrics(sc[m], c[m])})
                if task == "ALL":
                    gate_auc[(mode, level, cond)] = rows_out[-1]["auc_selected"]
        if args.save_scores:
            np.savez_compressed(os.path.join(args.out, f"scores_{mode}_L{level}.npz"), c=c, tasks=tasks.astype(str),
                                anchors=anchors, **saved)
        print(f"[{mode} L{level}] {time.time() - t0:.0f}s  " + "  ".join(
            f"{cd} auc_sel={gate_auc[(mode, level, cd)]:.4f}" for cd in conditions))
        pd.DataFrame(rows_out).to_csv(os.path.join(args.out, "prefix_offline.csv"), index=False)

    # 통과 기준 (1단계 → 2단계): L ≥ 4 에서 M2 AUC가 L0 대비 0.03 이상 하락 + M3 회복률 ≥ 0.5
    gate = {"metric": "auc_selected", "drop_min": args.gate_drop, "recovery_min": args.gate_recovery, "levels": {}}
    r = gate_auc.get(("variant", 0, "M2"))
    for L in args.levels:
        if L < 4 or r is None:
            continue
        m2, m3, m4 = (gate_auc.get(("variant", L, x)) for x in ("M2", "M3", "M4"))
        drop = r - m2
        rec = (m3 - m2) / drop if drop > 0 else float("nan")
        gate["levels"][f"L{L}"] = {"R": r, "M2": m2, "M3": m3, "M4": m4, "drop": drop, "recovery": rec,
                                   "pass": bool(drop >= args.gate_drop and rec >= args.gate_recovery)}
    passed = [k for k, v in gate["levels"].items() if v["pass"]]
    gate["stage2_level"] = max(passed, key=lambda k: int(k[1:])) if passed else None
    with open(os.path.join(args.out, "prefix_gate.json"), "w") as f:
        json.dump(gate, f, indent=1)
    meta = {"args": vars(args), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                            text=True).stdout.strip(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "m3_training": train_logs}
    with open(os.path.join(args.out, "prefix_offline_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
