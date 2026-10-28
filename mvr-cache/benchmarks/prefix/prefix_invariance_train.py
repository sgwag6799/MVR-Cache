"""Experiment 7: fine-tune the encoder so the original segments do not change when shared
sentences are added (label-free, LoRA on bge-base).

Teacher = the frozen base encoder on the original prompt alone (the L0 segment vectors).
Student = base + LoRA (query/value of every layer) on the prompt with 0-8 training sentences
from prefix_pool_train.json in front (or, for --suffix-share of rows, after it). Both are cut with
the rules of prefix_split.py, so the original part has the same segments in both.

  invariance loss  mean (1 - cos) between student original segments (with shared sentences) and
                   the teacher segments of the same prompt alone
  keep loss        the same on rows that get no shared sentences (student L0 = teacher L0)
  total            invariance + keep (1:1)

--control trains with no shared sentences at all (keep loss only, same steps), to separate the
effect of "more training" from the invariance objective.

Fixed before the run (PREFIX_EXPERIMENT.md 19): train split only; the epoch is picked on the val
split by the same label-free losses (fixed augmentation per row); test and the evaluation pool
prefix_pool.json are never used here. The best epoch's LoRA is merged into the base weights and
saved as a plain transformers model for --model in prefix_offline.py / prefix_contamination.py.

Example:
  python -m benchmarks.prefix.prefix_invariance_train --train ../data/lmarena/train.parquet \
    --val ../data/lmarena/val.parquet --out ../data/exp7/lora
"""
# [새 파일] 실험 7: 공통 문장이 붙어도 원본 조각 벡터가 바뀌지 않도록 인코더를 LoRA로 학습 (라벨 미사용)

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from benchmarks.prefix._shared import ps

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- 분할 · 조각


def original_segments(tok, texts, starts, n_chars, suffix, max_cuts):
    """Tokenise and return, per text, (input batch, list of (s, e) spans of the original segments)."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=ps.MAX_LEN,
              return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping").tolist()
    spans_all = []
    for j in range(len(texts)):
        length = int(enc["attention_mask"][j].sum())
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][j][:length].tolist())
        off = [tuple(o) for o in offsets[j][:length]]
        if suffix[j] and n_chars[j] > 0:
            sstart = ps.suffix_token_start(off, starts[j])
            spans = ps.segment_spans(length, ps.suffix_cut_points(tokens, length, sstart, max_cuts=max_cuts))
            flags = ps.suffix_segment_flags(spans, sstart)
        else:
            pend = ps.prefix_token_end(off, n_chars[j])
            spans = ps.segment_spans(length, ps.cut_points(tokens, length, max_cuts=max_cuts, prefix_end=pend))
            flags = ps.prefix_segment_flags(spans, pend)
        spans_all.append([sp for sp, f in zip(spans, flags) if not f])
    return enc, spans_all


def pool_segments(hidden, spans_all):
    return [torch.stack([hidden[j, s:e].mean(0) for s, e in spans]) for j, spans in enumerate(spans_all)]


# ----------------------------------------------------------------------------- 증강


def augment(prompt, pool, level, suffix, rng):
    """Text with `level` training sentences (one random variant per slot, slots in random order)."""
    if level == 0:
        return prompt, 0, 0, False
    slots = rng.sample(range(len(pool)), level)
    shared = " ".join(rng.choice(pool[i]) for i in slots)
    if suffix:
        text, start = ps.join_suffix(shared, prompt)
        return text, start, len(shared), True
    text, n = ps.join_prompt(shared, prompt)
    return text, 0, n, False


def check_pool(train_pool, eval_pool_path, tok, lm, device):
    """No sentence shared with the evaluation pool; report the closest pair by sentence cosine."""
    eval_pool = ps.load_pool(eval_pool_path)
    a = [s for slot in train_pool for s in slot]
    b = [s for slot in eval_pool for s in slot]
    assert not set(a) & set(b), "training pool repeats an evaluation sentence"
    bad = [s for s in a if any(ch in s[:-1] for ch in ".,!?;:") or not s.endswith(".")]
    assert not bad, f"sentences must have only a final period: {bad[:3]}"

    @torch.no_grad()
    def emb(xs):
        e = tok(xs, return_tensors="pt", padding=True, truncation=True).to(device)
        h = lm(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    sim = emb(a) @ emb(b).T
    i, j = divmod(int(sim.argmax()), sim.shape[1])
    within = emb(b) @ emb(b).T
    within.fill_diagonal_(-1)
    return {"train_sentences": len(a), "eval_sentences": len(b), "max_cos_train_vs_eval": float(sim.max()),
            "closest_pair": [a[i], b[j]], "mean_max_cos_train_vs_eval": float(sim.max(1).values.mean()),
            "mean_max_cos_eval_vs_eval": float(within.max(1).values.mean())}


# ----------------------------------------------------------------------------- 메인


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Split parquet with a 'prompt' column (originals)")
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--pool", default=os.path.join(HERE, "prefix_pool_train.json"))
    p.add_argument("--eval-pool", default=os.path.join(HERE, "prefix_pool.json"), help="Only for the overlap check")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-level", type=int, default=8)
    p.add_argument("--keep-share", type=float, default=0.25, help="Rows with no shared sentences (keep loss)")
    p.add_argument("--suffix-share", type=float, default=0.2, help="Rows with the sentences after the prompt")
    p.add_argument("--control", action="store_true", help="Never add shared sentences (keep loss only)")
    p.add_argument("--max-cuts", type=int, default=32)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import peft.import_utils
    import peft.tuners.lora.torchao

    # Colab에 깔린 torchao(0.10)가 peft가 요구하는 버전보다 낮아 LoRA 생성 중 오류가 난다. torchao는 쓰지 않으므로 끈다
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModel.from_pretrained(args.model).to(dev).eval()
    pool = ps.load_pool(args.pool)
    pool_check = check_pool(pool, args.eval_pool, tok, teacher, dev)
    print("[pool]", json.dumps(pool_check))

    # 가장 긴 증강(max_level 문장)을 붙여도 512토큰 안에 드는 행만 (평가의 fits_all_levels와 같은 생각)
    longest = " ".join(max(slot, key=len) for slot in pool[:args.max_level])

    def load(path):
        prompts = pd.read_parquet(path)["prompt"].astype(str).tolist()
        n = [len(tok(longest + " " + t, truncation=False)["input_ids"]) for t in prompts]
        return [t for t, k in zip(prompts, n) if k <= ps.MAX_LEN]

    train, val = load(args.train), load(args.val)
    print(f"[data] train {len(train)}  val {len(val)}")

    @torch.no_grad()
    def teacher_segments(prompts, bs=64):
        out = []
        for b in range(0, len(prompts), bs):
            chunk = prompts[b:b + bs]
            enc, spans = original_segments(tok, chunk, [0] * len(chunk), [0] * len(chunk), [False] * len(chunk),
                                           args.max_cuts)
            h = teacher(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state.float()
            out += [s.cpu() for s in pool_segments(h, spans)]
        return out

    t_train, t_val = teacher_segments(train), teacher_segments(val)

    # 원래 가중치의 dropout은 끈다 (선생은 dropout 없이 계산하므로). 정규화는 LoRA dropout만
    student = get_peft_model(AutoModel.from_pretrained(args.model, hidden_dropout_prob=0.0,
                                                       attention_probs_dropout_prob=0.0),
                             LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                        target_modules=["query", "value"])).to(dev)
    n_train = sum(x.numel() for x in student.parameters() if x.requires_grad)
    print(f"[lora] trainable parameters {n_train}")

    def plan(rng, n):
        """(level, suffix) per row."""
        out = []
        for _ in range(n):
            if args.control or rng.random() < args.keep_share:
                out.append((0, False))
            else:
                out.append((rng.randint(1, args.max_level), rng.random() < args.suffix_share))
        return out

    def batch_loss(prompts, targets, rows_plan, rng):
        texts, starts, nch, suf = [], [], [], []
        for t, (lvl, sx) in zip(prompts, rows_plan):
            text, st, n, s = augment(t, pool, lvl, sx, rng)
            texts.append(text), starts.append(st), nch.append(n), suf.append(s)
        enc, spans = original_segments(tok, texts, starts, nch, suf, args.max_cuts)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
            h = student(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state
        segs = pool_segments(h.float(), spans)
        inv, keep = [], []
        for sv, tv, (lvl, _) in zip(segs, targets, rows_plan):
            assert sv.shape[0] == tv.shape[0], "original segments differ from the teacher's"
            d = 1.0 - F.cosine_similarity(sv, tv.to(dev), dim=-1)
            (keep if lvl == 0 else inv).append(d)
        inv_l = torch.cat(inv).mean() if inv else torch.zeros((), device=dev)
        keep_l = torch.cat(keep).mean() if keep else torch.zeros((), device=dev)
        return inv_l, keep_l

    # val: 행마다 고정된 증강 (단계 0·2·4·6·8 순환, 앞쪽) → 에폭끼리 같은 문제로 비교
    val_rng = random.Random(1234)
    val_plan = [(0 if args.control else [0, 2, 4, 6, 8][i % 5], False) for i in range(len(val))]
    val_texts_seed = [val_rng.random() for _ in val]

    @torch.no_grad()
    def evaluate():
        student.eval()
        tot_i = tot_k = 0.0
        ni = nk = 0
        for b in range(0, len(val), 64):
            rows = list(range(b, min(b + 64, len(val))))
            for r in rows:
                i_l, k_l = batch_loss([val[r]], [t_val[r]], [val_plan[r]], random.Random(val_texts_seed[r]))
                if val_plan[r][0] == 0:
                    tot_k += float(k_l); nk += 1
                else:
                    tot_i += float(i_l); ni += 1
        student.train()
        return (tot_i / ni if ni else 0.0), (tot_k / nk if nk else 0.0)

    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW([x for x in student.parameters() if x.requires_grad], lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.startswith("cuda"))

    v_inv0, v_keep0 = evaluate()
    log = [{"epoch": -1, "val_inv": v_inv0, "val_keep": v_keep0}]
    print(f"[epoch -1] val inv={v_inv0:.5f} keep={v_keep0:.5f}")
    best = {"epoch": -1, "score": v_inv0 + v_keep0, "state": None}
    rng = random.Random(args.seed)
    student.train()
    for ep in range(args.epochs):
        t0 = time.time()
        order = np.random.default_rng(args.seed + ep).permutation(len(train))
        tr_i = tr_k = 0.0
        nb = 0
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            rows_plan = plan(rng, len(idx))
            i_l, k_l = batch_loss([train[i] for i in idx], [t_train[i] for i in idx], rows_plan, rng)
            loss = i_l + k_l
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([x for x in student.parameters() if x.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_i += float(i_l); tr_k += float(k_l); nb += 1
        v_inv, v_keep = evaluate()
        score = v_inv + v_keep
        log.append({"epoch": ep, "train_inv": tr_i / nb, "train_keep": tr_k / nb, "val_inv": v_inv, "val_keep": v_keep,
                    "seconds": time.time() - t0})
        print(f"[epoch {ep}] {time.time() - t0:.0f}s train inv={tr_i / nb:.5f} keep={tr_k / nb:.5f} | "
              f"val inv={v_inv:.5f} keep={v_keep:.5f}")
        if score < best["score"]:
            best = {"epoch": ep, "score": score,
                    "state": {k: v.detach().clone() for k, v in student.state_dict().items() if "lora_" in k}}
        elif ep - best["epoch"] >= args.patience:
            break

    if best["state"] is not None:
        student.load_state_dict(best["state"], strict=False)
    merged = student.merge_and_unload()
    merged.save_pretrained(os.path.join(args.out, "model"))
    tok.save_pretrained(os.path.join(args.out, "model"))
    meta = {"args": vars(args), "pool_check": pool_check, "train_rows": len(train), "val_rows": len(val),
            "trainable_parameters": n_train, "selected_epoch": best["epoch"], "log": log,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"-> {args.out}/model (selected epoch {best['epoch']})")


if __name__ == "__main__":
    main()
