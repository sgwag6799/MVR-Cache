"""Build the dialect paraphrase dataset from AI Hub 경상방언 label JSON files.

Each utterance has a Gyeongsang-dialect transcription (dialect_form) and its standard Korean
rendering (standard_form). Where the two differ they are a natural same-meaning pair, which is
what a semantic cache should treat as the same request.

Outputs under --out-dir (keep it out of git: the data is licensed and public pushes are not allowed):
  index.parquet   every unique cleaned standard form = the "cache" searched in the retrieval eval
  pairs.parquet   one row per utterance whose cleaned dialect and standard forms differ (the queries)
  stream.parquet  vCache stream: index entries (standard) first, then the dialect queries;
                  id_set = group of the standard text, so benchmark_id_set scoring works
  manifest.json   counts and the cleaning rules used

Example:
  python benchmarks/dialect/prepare_dialect_dataset.py \
    --label-dir /content/drive/MyDrive/dialect/라벨링데이터 --out-dir /content/drive/MyDrive/dialect/prepared
"""

import argparse
import difflib
import glob
import json
import os
import random
import re

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label-dir", required=True, help="Folder with the label JSON files (searched recursively).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-eojeol", type=int, default=2, help="Drop query pairs whose standard form is shorter.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.label_dir, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"no JSON files under {args.label_dir}")

    utts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        meta = d.get("metadata", {})
        for u in d.get("utterance", []):
            ej = u.get("eojeolList") or []
            utts.append({
                "utt_id": u["id"],
                "file_id": d["id"],
                "topic": meta.get("topic"),
                "dialect": clean(u.get("dialect_form")),
                "standard": clean(u.get("standard_form")),
                "n_eojeol": len(ej),
                "n_dialect_eojeol": sum(bool(e.get("isDialect")) for e in ej),
            })

    # ---- index: 모든 발화의 표준어 문장 (중복 문장은 한 항목으로) ----
    groups: dict = {}
    for u in utts:
        if not u["standard"]:
            continue
        k = key(u["standard"])
        if not k:
            continue
        g = groups.setdefault(k, {"text": u["standard"], "utt_ids": [], "file_ids": set()})
        g["utt_ids"].append(u["utt_id"])
        g["file_ids"].add(u["file_id"])
    keys = list(groups)
    gid = {k: i for i, k in enumerate(keys)}
    index = pd.DataFrame([
        {"group_id": gid[k], "text": groups[k]["text"], "n_utterances": len(groups[k]["utt_ids"]),
         "utt_ids": groups[k]["utt_ids"], "n_files": len(groups[k]["file_ids"])}
        for k in keys
    ])

    # ---- pairs: 방언형과 표준어형이 실제로 다른 발화 (질의) ----
    rows, dropped = [], {"empty": 0, "same_after_cleaning": 0, "short": 0}
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
        k = key(u["standard"])
        dk = key(u["dialect"])
        rows.append({
            **{c: u[c] for c in ("utt_id", "file_id", "topic", "dialect", "standard", "n_eojeol", "n_dialect_eojeol")},
            "group_id": gid[k],
            "dialect_ratio": u["n_dialect_eojeol"] / u["n_eojeol"] if u["n_eojeol"] else None,
            # 표면 유사도: 방언형과 표준어형이 글자 수준에서 얼마나 비슷한지 (1 = 같음)
            "surface_sim": difflib.SequenceMatcher(None, u["dialect"], u["standard"]).ratio(),
            # 방언 문장이 다른 그룹의 표준어 문장과 글자까지 같은 경우 (정답이 모호한 질의)
            "dialect_matches_other_standard": dk in gid and gid[dk] != gid[k],
        })
    pairs = pd.DataFrame(rows)

    # ---- stream: vCache 실행용 (표준어 전부 → 방언 질의), id_set = 표준어 그룹 ----
    rng = random.Random(args.seed)
    std_rows = [{"prompt": t, "id_set": g, "variant": "standard", "id": f"std{g}"}
                for g, t in zip(index["group_id"], index["text"])]
    dia_rows = [{"prompt": r["dialect"], "id_set": r["group_id"], "variant": "dialect", "id": r["utt_id"]}
                for r in rows]
    rng.shuffle(std_rows)
    rng.shuffle(dia_rows)
    stream = pd.DataFrame(std_rows + dia_rows)
    stream["dataset_name"] = "dialect_gyeongsang"

    os.makedirs(args.out_dir, exist_ok=True)
    index.to_parquet(os.path.join(args.out_dir, "index.parquet"), index=False)
    pairs.to_parquet(os.path.join(args.out_dir, "pairs.parquet"), index=False)
    stream.to_parquet(os.path.join(args.out_dir, "stream.parquet"), index=False)
    manifest = {
        "source": "AI Hub 경상방언 AI 학습데이터 (솔트룩스, 2020) label JSON",
        "label_dir": args.label_dir,
        "n_files": len(files),
        "n_utterances": len(utts),
        "n_index_groups": len(index),
        "n_query_pairs": len(pairs),
        "dropped_pairs": dropped,
        "n_dialect_matches_other_standard": int(pairs["dialect_matches_other_standard"].sum()) if len(pairs) else 0,
        "n_stream_rows": len(stream),
        "min_eojeol": args.min_eojeol,
        "seed": args.seed,
        "cleaning": CLEANING_RULES,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "cleaning"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
