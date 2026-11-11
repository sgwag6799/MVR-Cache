"""Claim A on Classification: build the stage 2 stream files and the RL splitter train/val files.

Input: the Classification shared-paragraph data (BRANCH_NOTES.md의 PREFIX_EXPERIMENT 부분 2절), made by
scripts/make_cls_splits.py and benchmarks.prefix.make_prefix_data into data/prefix/data.

Output (under <repo>/data/cls_stage2, git-ignored):
  stream/L0_test.parquet, stream/L8_test.parquet  all test rows of the level (41,000; the eval scripts filter
                                                  fits_all_levels themselves -> 38,304). Answer = response_llama_3_8b
  rl/train.parquet, rl/val.parquet                L0 rows with fits_all_levels, columns id/prompt/response_llama_3_8b/
                                                  dataset_name (train 2,791 / val 935)

Usage (repo root):  python scripts/make_stage2_cls_data.py
"""
# [새 파일] 주장 A (Classification): 스트림 · RL 학습용 파일 준비

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "prefix", "data", "variant")
OUT = os.path.join(REPO, "data", "cls_stage2")
RESP = "response_llama_3_8b"


def main() -> None:
    os.makedirs(os.path.join(OUT, "rl"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "stream"), exist_ok=True)

    # RL 분할기 학습 · 선택용 (원본 L0, 512토큰 안에 드는 행만, 정답 = 8B 응답 문자열)
    for s in ["train", "val"]:
        d = pd.read_parquet(os.path.join(SRC, "L0", f"{s}.parquet"))
        d = d[d.fits_all_levels].reset_index(drop=True)
        d[["id", "prompt", RESP, "dataset_name"]].to_parquet(os.path.join(OUT, "rl", f"{s}.parquet"), index=False)
        print(s, len(d), d[RESP].nunique())

    # 2단계 스트림 (test 원래 순서 그대로)
    for L in [0, 8]:
        d = pd.read_parquet(os.path.join(SRC, f"L{L}", "test.parquet"))
        d.to_parquet(os.path.join(OUT, "stream", f"L{L}_test.parquet"), index=False)
        print(L, len(d), int(d.fits_all_levels.sum()))


if __name__ == "__main__":
    main()
