"""Build the dialect paraphrase dataset from AI Hub 경상방언 label files.

Each utterance has a Gyeongsang-dialect transcription (dialect_form) and its standard Korean
rendering (standard_form). Where the two differ they are a natural same-meaning pair, which is
what a semantic cache should treat as the same request.

Inputs can be folders or the AI Hub .zip files themselves (JSON members are read straight from the
zip, so 100k+ small files never have to be extracted onto Google Drive). AI Hub ships a Training and
a Validation part; pass them as --train and --val so the retrieval eval can pick a reuse threshold on
Training queries and measure it on Validation queries.

Outputs under --out-dir (keep it out of git: the data is licensed and this repository is public):
  index.parquet        every unique cleaned standard sentence from all splits = the cache searched
                       by the retrieval eval
  pairs.parquet        one row per utterance whose cleaned dialect and standard forms differ (the
                       queries), with `split` = train / val
  stream_cold.parquet  vCache stream where every meaning appears once: all standard sentences, then
                       the dialect queries. No cache entry can reach vCache's 6 observations
  stream_zipf.parquet  vCache stream with repeated requests: query pairs drawn with Zipf popularity,
                       each request in the dialect or the standard form, plus one-off standard
                       sentences as a long tail. `kind` = first / exact_repeat / cross_variant
  stream_repeat.parquet  vCache stream of real repetition only: every utterance whose standard
                       sentence was spoken at least --repeat-min times, as actually spoken (the
                       dialect transcription), plus one-off sentences as a long tail
  manifest.json        counts, cleaning rules, how often meanings really repeat in the data

id_set = group of the standard sentence, so benchmark_id_set scoring works on both streams.

Example:
  python benchmarks/dialect/prepare_dialect_dataset.py \
    --train "/content/drive/MyDrive/dialect/zips/(비식별화완료)경상도_학습데이터_1.zip" \
    --val   "/content/drive/MyDrive/dialect/zips/(비식별화완료)경상도_학습데이터_2.zip" \
    --out-dir /content/drive/MyDrive/dialect/prepared
"""

import argparse
import collections
import difflib
import glob
import json
import os
import random
import re
import zipfile

import numpy as np
import pandas as pd

CLEANING_RULES = [
    "{...} tags removed (e.g. {laughing})",
    "((x)) uncertain transcription -> x, empty (()) removed",
    "'~' (vowel lengthening) removed",
    "whitespace collapsed",
]


def clean(text: str) -> str:
    t = re.sub(r"\{[^}]*\}", " ", text or "")
    t = re.sub(r"\(\(([^)]*)\)\)", r"\1", t)
    t = t.replace("~", "")
    return re.sub(r"\s+", " ", t).strip()


def key(text: str) -> str:
    # 같은 문장인지 판단할 때는 띄어쓰기·문장부호 차이를 무시한다 ("그래서" == "그래서.")
    return re.sub(r"[^\w]", "", text)


def is_label_json(name: str) -> bool:
    # macOS가 압축할 때 넣는 __MACOSX/ 폴더와 ._파일명(리소스 포크)은 이름만 .json이고 JSON이 아니다
    base = os.path.basename(name)
    return name.lower().endswith(".json") and "__MACOSX" not in name and not base.startswith("._")


def iter_label_files(paths: list):
    """Yield parsed label JSON dicts from folders (recursive) and/or .zip files."""
    for path in paths:
        if os.path.isdir(path):
            for f in sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)):
                if not is_label_json(f):
                    continue
                with open(f, encoding="utf-8-sig") as fh:
                    yield json.load(fh)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for name in sorted(n for n in zf.namelist() if is_label_json(n)):
                    with zf.open(name) as fh:
                        yield json.loads(fh.read().decode("utf-8-sig"))
        else:
            raise SystemExit(f"not a folder or zip file: {path}")


def load_utterances(paths: list, split: str) -> list:
    utts = []
    for d in iter_label_files(paths):
        if "utterance" not in d:
            continue
        topic = (d.get("metadata") or {}).get("topic")
        for u in d["utterance"]:
            ej = u.get("eojeolList") or []
            utts.append({
                "split": split,
                "utt_id": u.get("id"),
                "file_id": d.get("id"),
                # 파일 ID 접두어: DKSR / DKCI (Training에는 둘 다, Validation에는 DKSR만 있음)
                "source": str(d.get("id") or "")[:4],
                "topic": topic,
                "dialect": clean(u.get("dialect_form")),
                "standard": clean(u.get("standard_form")),
                "n_eojeol": len(ej),
                "n_dialect_eojeol": sum(bool(e.get("isDialect")) for e in ej),
            })
    return utts


def kinds(df: pd.DataFrame) -> list:
    """first / exact_repeat (same text of a meaning seen before) / cross_variant (meaning seen, other text)."""
    seen: dict = {}
    out = []
    for g, text in zip(df["id_set"], df["prompt"]):
        texts = seen.setdefault(g, set())
        out.append("first" if not texts else "exact_repeat" if text in texts else "cross_variant")
        texts.add(text)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=False, help="Training label folders or .zip files")
    p.add_argument("--val", nargs="*", default=[], help="Validation label folders or .zip files")
    p.add_argument("--label-dir", default=None, help="(old) single folder, treated as --train")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-eojeol", type=int, default=2, help="Drop query pairs whose standard form is shorter.")
    p.add_argument("--zipf-requests", type=int, default=50000, help="Length of stream_zipf (0 = skip).")
    p.add_argument("--zipf-a", type=float, default=1.1, help="Zipf exponent of request popularity.")
    p.add_argument("--zipf-dialect-prob", type=float, default=0.5, help="Share of repeated-meaning requests sent in dialect form.")
    p.add_argument("--zipf-singleton-frac", type=float, default=0.3,
                   help="Share of requests that are one-off standard sentences (stream_zipf and stream_repeat).")
    p.add_argument("--repeat-min", type=int, default=6,
                   help="stream_repeat keeps meanings whose standard sentence was really spoken at least this often.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train_paths = list(args.train or []) + ([args.label_dir] if args.label_dir else [])
    if not train_paths:
        raise SystemExit("give --train (or --label-dir)")

    utts = load_utterances(train_paths, "train") + load_utterances(args.val, "val")
    if not utts:
        raise SystemExit("no utterances found")

    # ---- index: 모든 split의 표준어 문장 (중복 문장은 한 항목으로) ----
    groups: dict = {}
    for u in utts:
        k = key(u["standard"])
        if not k:
            continue
        g = groups.setdefault(k, {"text": u["standard"], "n": 0, "splits": set()})
        g["n"] += 1
        g["splits"].add(u["split"])
    keys = list(groups)
    gid = {k: i for i, k in enumerate(keys)}
    index = pd.DataFrame([
        {"group_id": gid[k], "text": groups[k]["text"], "n_utterances": groups[k]["n"],
         "splits": ",".join(sorted(groups[k]["splits"]))}
        for k in keys
    ])

    # ---- pairs: 방언형과 표준어형이 실제로 다른 발화 (질의) ----
    rows, dropped = [], collections.Counter()
    for u in utts:
        if not u["dialect"] or not u["standard"]:
            dropped["empty"] += 1
            continue
        if key(u["dialect"]) == key(u["standard"]):
            dropped["same_after_cleaning"] += 1
            continue
        if len(u["standard"].split()) < args.min_eojeol:
            dropped["short"] += 1
            continue
        k, dk = key(u["standard"]), key(u["dialect"])
        rows.append({
            **{c: u[c] for c in ("split", "source", "utt_id", "file_id", "topic", "dialect", "standard", "n_eojeol", "n_dialect_eojeol")},
            "group_id": gid[k],
            "dialect_ratio": u["n_dialect_eojeol"] / u["n_eojeol"] if u["n_eojeol"] else None,
            # 표면 유사도: 방언형과 표준어형이 글자 수준에서 얼마나 비슷한지 (1 = 같음)
            "surface_sim": difflib.SequenceMatcher(None, u["dialect"], u["standard"]).ratio(),
            # 방언 문장이 다른 그룹의 표준어 문장과 글자까지 같은 경우 (정답이 모호한 질의)
            "dialect_matches_other_standard": dk in gid and gid[dk] != gid[k],
        })
    pairs = pd.DataFrame(rows)

    rng = random.Random(args.seed)
    # ---- stream_cold: 모든 의미가 한 번씩 (표준어 전부 → 방언 질의) ----
    std_rows = [{"prompt": t, "id_set": g, "variant": "standard", "id": f"std{g}"}
                for g, t in zip(index["group_id"], index["text"])]
    dia_rows = [{"prompt": r["dialect"], "id_set": r["group_id"], "variant": "dialect", "id": r["utt_id"]}
                for r in rows]
    rng.shuffle(std_rows)
    rng.shuffle(dia_rows)
    cold = pd.DataFrame(std_rows + dia_rows)
    cold["dataset_name"] = "dialect_gyeongsang"
    cold["kind"] = kinds(cold)

    # ---- stream_zipf: 인기 있는 의미가 반복되는 요청 (방언형/표준어형 섞어서) + 일회성 표준어 문장 ----
    zipf = None
    if args.zipf_requests > 0 and len(pairs):
        nrng = np.random.default_rng(args.seed)
        pair_order = nrng.permutation(len(pairs))  # 인기 순위는 무작위로 배정 (pair_order[r] = r+1등인 질의)
        rank_of = np.empty(len(pairs), dtype=int)
        rank_of[pair_order] = np.arange(1, len(pairs) + 1)
        w = 1.0 / np.arange(1, len(pairs) + 1) ** args.zipf_a
        w /= w.sum()
        query_groups = set(pairs["group_id"])
        singles = index[~index["group_id"].isin(query_groups)]
        n_single = int(round(args.zipf_requests * args.zipf_singleton_frac)) if len(singles) else 0
        n_rep = args.zipf_requests - n_single
        picks = pair_order[nrng.choice(len(pairs), size=n_rep, p=w)]
        use_dialect = nrng.random(n_rep) < args.zipf_dialect_prob
        zrows = [{"prompt": pairs.at[i, "dialect"] if d else pairs.at[i, "standard"],
                  "id_set": int(pairs.at[i, "group_id"]), "variant": "dialect" if d else "standard",
                  "id": pairs.at[i, "utt_id"], "popularity_rank": int(rank_of[i])}
                 for i, d in zip(picks, use_dialect)]
        one_off = singles.sample(n=min(n_single, len(singles)), random_state=args.seed)
        zrows += [{"prompt": t, "id_set": int(g), "variant": "standard", "id": f"std{g}", "popularity_rank": None}
                  for g, t in zip(one_off["group_id"], one_off["text"])]
        zipf = pd.DataFrame(zrows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        zipf["dataset_name"] = "dialect_gyeongsang"
        zipf["kind"] = kinds(zipf)

    # ---- 실제 데이터에서 같은 표준어 문장이 몇 번 반복되는가 (vCache는 항목당 관측 6개가 필요) ----
    counts = collections.Counter(key(u["standard"]) for u in utts if key(u["standard"]) and len(u["standard"].split()) >= args.min_eojeol)
    repeat_stats = {f">={n}": sum(c >= n for c in counts.values()) for n in (2, 6, 20)}

    # ---- stream_repeat: 실제로 여러 번 말해진 의미(표준어 문장 --repeat-min회 이상)만, 실제 발화(방언 전사)로 ----
    rep_keys = {k for k, c in counts.items() if c >= args.repeat_min}
    repeat = None
    if rep_keys:
        rrows = []
        for u in utts:
            k = key(u["standard"])
            if k in rep_keys and u["dialect"]:
                rrows.append({"prompt": u["dialect"], "id_set": gid[k],
                              "variant": "dialect" if key(u["dialect"]) != k else "standard", "id": u["utt_id"]})
        one_singles = index[index["n_utterances"] == 1]
        n_single = int(round(len(rrows) * args.zipf_singleton_frac / (1 - args.zipf_singleton_frac))) if len(one_singles) else 0
        one_off = one_singles.sample(n=min(n_single, len(one_singles)), random_state=args.seed)
        rrows += [{"prompt": t, "id_set": int(g), "variant": "standard", "id": f"std{g}"}
                  for g, t in zip(one_off["group_id"], one_off["text"])]
        repeat = pd.DataFrame(rrows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        repeat["dataset_name"] = "dialect_gyeongsang"
        repeat["kind"] = kinds(repeat)
    top_repeated = [{"text": groups[k]["text"], "count": c} for k, c in counts.most_common(15)]

    os.makedirs(args.out_dir, exist_ok=True)
    index.to_parquet(os.path.join(args.out_dir, "index.parquet"), index=False)
    pairs.to_parquet(os.path.join(args.out_dir, "pairs.parquet"), index=False)
    cold.to_parquet(os.path.join(args.out_dir, "stream_cold.parquet"), index=False)
    if zipf is not None:
        zipf.to_parquet(os.path.join(args.out_dir, "stream_zipf.parquet"), index=False)
    if repeat is not None:
        repeat.to_parquet(os.path.join(args.out_dir, "stream_repeat.parquet"), index=False)
    manifest = {
        "source": "AI Hub 한국어 방언 발화 데이터(경상도) label JSON",
        "train_inputs": train_paths,
        "val_inputs": args.val,
        "n_utterances": {s: sum(u["split"] == s for u in utts) for s in ("train", "val")},
        "n_index_sentences": len(index),
        "n_query_pairs": pairs["split"].value_counts().to_dict() if len(pairs) else {},
        "n_query_pairs_by_source": {f"{s}/{src}": int(n) for (s, src), n in pairs.groupby(["split", "source"]).size().items()} if len(pairs) else {},
        "dropped_pairs": dict(dropped),
        "n_dialect_matches_other_standard": int(pairs["dialect_matches_other_standard"].sum()) if len(pairs) else 0,
        "standard_sentences_repeated": repeat_stats,
        "most_repeated_standard_sentences": top_repeated,
        "stream_repeat": None if repeat is None else {
            "rows": len(repeat), "kind": repeat["kind"].value_counts().to_dict(),
            "variant": repeat["variant"].value_counts().to_dict(),
            "repeated_meanings": len(rep_keys), "repeat_min": args.repeat_min,
            "singleton_frac": args.zipf_singleton_frac,
        },
        "stream_cold": {"rows": len(cold), "kind": cold["kind"].value_counts().to_dict()},
        "stream_zipf": None if zipf is None else {
            "rows": len(zipf), "kind": zipf["kind"].value_counts().to_dict(),
            "distinct_meanings": int(zipf["id_set"].nunique()), "zipf_a": args.zipf_a,
            "dialect_prob": args.zipf_dialect_prob, "singleton_frac": args.zipf_singleton_frac,
        },
        "min_eojeol": args.min_eojeol,
        "seed": args.seed,
        "cleaning": CLEANING_RULES,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "cleaning"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
