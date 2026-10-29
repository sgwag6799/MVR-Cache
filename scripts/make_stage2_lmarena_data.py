"""Experiment 8 / claim A: build the stage 2 stream files and the RL splitter train/val files.

Input: the LmArena prefix data (PREFIX_EXPERIMENT.md 2.1 · 8절), with the merged answer
column `id_set_merged` (scripts/lmarena_merge_duplicate_sets.py).

Output (under <repo>/data/exp8, git-ignored):
  stream/L0_test.parquet, stream/L8_test.parquet  all 40,964 test rows of the level, `id_set` = id_set_merged
                                                  (the eval scripts filter fits_all_levels themselves)
  rl/train.parquet, rl/val.parquet                L0 rows with fits_all_levels, columns id/prompt/id_set/dataset_name
                                                  (train 3,007 / val 1,000)

Usage (repo root):  python scripts/make_stage2_lmarena_data.py
"""
# [새 파일] 실험 8 데이터 준비. 원래 VM에서 인라인 명령으로 실행한 것을 같은 내용으로 스크립트화

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "lmarena", "prefix", "data", "variant")
OUT = os.path.join(REPO, "data", "exp8")


def main() -> None:
    os.makedirs(os.path.join(OUT, "rl"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "stream"), exist_ok=True)

    # RL 분할기 학습 · 선택용 (원본 L0, 512토큰 안에 드는 행만)
    for s in ["train", "val"]:
        d = pd.read_parquet(os.path.join(SRC, "L0", f"{s}.parquet"))
        d = d[d.fits_all_levels].reset_index(drop=True)
        d["id_set"] = d["id_set_merged"].astype(int)
        d = d.drop(columns=["ID_Set"])
        d[["id", "prompt", "id_set", "dataset_name"]].to_parquet(os.path.join(OUT, "rl", f"{s}.parquet"), index=False)
        print(s, len(d), d.id_set.nunique())

    # 2단계 스트림 (test 원래 순서 그대로)
    for L in [0, 8]:
        d = pd.read_parquet(os.path.join(SRC, f"L{L}", "test.parquet"))
        d["id_set"] = d["id_set_merged"].astype(int)
        d = d.drop(columns=["ID_Set"])
        d.to_parquet(os.path.join(OUT, "stream", f"L{L}_test.parquet"), index=False)
        print(L, len(d), int(d.fits_all_levels.sum()))


if __name__ == "__main__":
    main()
