"""Build group-disjoint RL training splits and an evaluation stream from LmArena.

Every ID_Set group is assigned to exactly one of train / val / test / eval, so the
trained splitter never sees paraphrases of prompts in the evaluation stream.
The evaluation stream keeps the original dataset order.

anchor_nn sampling only trains on (prompt, nearest neighbour) pairs, so a negative
pair only appears when a prompt's nearest neighbour belongs to another group.
Training groups are therefore capped at `--max-per-group` prompts, and a
`--singleton-frac` share of them keeps a single prompt (no paraphrase available),
which mirrors a cache stream where most first-seen prompts have no match.

Example:
  python scripts/prepare_lmarena_splits.py \
    --input ../mvr-cache/data/lmarena.parquet --out-dir ../mvr-cache/data/lmarena_rl
"""
# [파일 전체 요약]
# LmArena 데이터셋을 RL 분할기 학습용(train / val / test)과 캐시 평가용(eval_stream)으로 나누는 스크립트.
# - LmArena에는 "같은 의미의 질문 묶음"을 나타내는 ID_Set 번호가 있다. 같은 번호 = 같은 뜻(바꿔 말한 문장들).
# - 같은 묶음이 학습과 평가에 동시에 들어가면, 모델이 평가 문제를 미리 본 셈(데이터 누수)이 된다.
#   그래서 "행" 단위가 아니라 "묶음(그룹)" 단위로 나눠서, 한 묶음은 딱 한 곳에만 들어가게 한다.
# - 학습 쌍은 "각 문장 + 가장 비슷한 문장"으로 만드는데, 같은 묶음 문장이 여러 개 있으면
#   가장 비슷한 문장이 늘 같은 묶음이라 정답 쌍(1)만 생긴다. (처음에 정답 비율 1.0이 나온 원인)
#   그래서 묶음당 문장 수를 최대 2개로 제한하고, 절반은 1개만 남겨서 오답 쌍(0)도 생기게 한다.

import argparse  # 명령줄 옵션(--input 같은 것)을 읽기 위한 표준 라이브러리
import os  # 폴더 만들기, 파일 경로 합치기 등 운영체제 관련 기능

import numpy as np  # 숫자 계산 라이브러리. 여기서는 난수(랜덤) 생성에 사용
import pandas as pd  # 표(데이터프레임) 데이터를 다루는 라이브러리. parquet 파일 읽기/쓰기


# 주어진 그룹 목록에서 문장을 뽑아 학습/검증/테스트용 표 하나를 만드는 함수.
# 반환값: (뽑힌 문장들의 표, 문장을 1개만 남긴 그룹 수)
def sample_split(df: pd.DataFrame, groups: list, label_col: str, target: int,
                 max_per_group: int, singleton_frac: float, rng: np.random.Generator,
                 seed: int) -> tuple[pd.DataFrame, int]:
    # rows: 그룹별로 뽑은 표 조각을 모아둘 리스트
    # total: 지금까지 뽑은 문장 수
    # n_single: 문장을 1개만 남긴 그룹 수 (통계 출력용)
    rows, total, n_single = [], 0, 0
    for g in groups:  # 이 세트에 배정된 그룹들을 하나씩 돌면서
        if total >= target:  # 목표 개수(예: 학습 2000개)를 채웠으면
            break  # 더 뽑지 않고 반복을 멈춘다
        members = df[df[label_col] == g]  # 그룹 번호가 g인 문장(행)들만 골라낸다
        # 몇 개를 뽑을지 결정:
        #   singleton_frac(기본 50%) 확률로 1개만 뽑고,
        #   아니면 그룹 크기와 max_per_group(기본 2) 중 작은 수만큼 뽑는다.
        k = 1 if rng.random() < singleton_frac else min(len(members), max_per_group)
        n_single += int(k == 1)  # 1개만 뽑았으면 카운트 +1 (True→1, False→0)
        rows.append(members.sample(n=k, random_state=seed))  # 그룹에서 k개를 무작위로 뽑아 저장
        total += k  # 뽑은 개수를 누적
    # 모은 조각들을 하나의 표로 합치고(concat), 순서를 한 번 섞는다(frac=1.0 = 전체를 섞기).
    # 같은 그룹 문장끼리 붙어 있지 않게 하려는 것.
    return pd.concat(rows).sample(frac=1.0, random_state=seed), n_single


def main() -> None:
    # ---- 1. 명령줄 옵션 정의 ----
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)  # 원본 데이터 파일 경로 (필수)
    p.add_argument("--out-dir", required=True)  # 결과 파일을 저장할 폴더 (필수)
    p.add_argument("--label-col", default="ID_Set")  # 그룹 번호가 들어 있는 열 이름
    p.add_argument("--train-group-frac", type=float, default=0.25)  # 전체 그룹 중 학습용 비율 (25%)
    p.add_argument("--val-group-frac", type=float, default=0.05)  # 검증용 그룹 비율 (5%)
    p.add_argument("--test-group-frac", type=float, default=0.05)  # 테스트용 그룹 비율 (5%)
    p.add_argument("--train", type=int, default=2000)  # 학습 세트에 담을 문장 수
    p.add_argument("--val", type=int, default=300)  # 검증 세트 문장 수
    p.add_argument("--test", type=int, default=300)  # 테스트 세트 문장 수
    p.add_argument("--max-per-group", type=int, default=2)  # 한 그룹에서 뽑을 최대 문장 수
    p.add_argument("--singleton-frac", type=float, default=0.5)  # 그룹에서 1개만 뽑을 확률
    p.add_argument("--seed", type=int, default=0)  # 난수 시드. 같으면 매번 같은 결과가 나온다
    args = p.parse_args()  # 실제로 입력된 옵션 값을 읽어 args에 담는다

    # ---- 2. 데이터 읽고 그룹 섞기 ----
    rng = np.random.default_rng(args.seed)  # 시드를 고정한 난수 생성기
    df = pd.read_parquet(args.input)  # 원본 데이터를 표로 읽는다
    groups = df[args.label_col].unique().tolist()  # 서로 다른 그룹 번호 목록 (중복 제거)
    rng.shuffle(groups)  # 그룹 순서를 무작위로 섞는다 → 어떤 그룹이 어디로 갈지 랜덤 배정

    # ---- 3. 그룹을 train / val / test / eval 로 나누기 ----
    n = len(groups)  # 전체 그룹 수
    n_train = int(n * args.train_group_frac)  # 학습용 그룹 수 (전체의 25%)
    n_val = int(n * args.val_group_frac)  # 검증용 그룹 수 (5%)
    n_test = int(n * args.test_group_frac)  # 테스트용 그룹 수 (5%)
    # 섞인 그룹 목록을 앞에서부터 잘라서 배정한다:
    #   [0, n_train) → 학습, 그다음 n_val개 → 검증, 그다음 n_test개 → 테스트
    split_groups = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val : n_train + n_val + n_test],
    }
    # 남은 그룹(나머지 약 65%)은 전부 캐시 평가용. set으로 만들어 포함 여부 확인을 빠르게 한다.
    eval_groups = set(groups[n_train + n_val + n_test :])

    # ---- 4. 학습/검증/테스트 파일 저장 ----
    os.makedirs(args.out_dir, exist_ok=True)  # 결과 폴더 생성 (이미 있으면 그냥 넘어감)
    for name, target in [("train", args.train), ("val", args.val), ("test", args.test)]:
        # 해당 세트의 그룹들에서 목표 개수만큼 문장을 뽑는다 (위 sample_split 함수)
        part, n_single = sample_split(
            df, split_groups[name], args.label_col, target,
            args.max_per_group, args.singleton_frac, rng, args.seed,
        )
        path = os.path.join(args.out_dir, f"{name}.parquet")  # 예: out-dir/train.parquet
        # reset_index: 행 번호를 0부터 다시 매김. index=False: 행 번호는 파일에 저장 안 함
        part.reset_index(drop=True).to_parquet(path, index=False)
        # 몇 문장, 몇 그룹, 1개짜리 그룹이 몇 개인지 화면에 출력
        print(f"{name}: {len(part)} prompts, {part[args.label_col].nunique()} groups "
              f"({n_single} singletons) -> {path}")

    # ---- 5. 평가 스트림 파일 저장 ----
    # 평가용 그룹에 속한 문장을 "전부" 가져온다. 섞지 않으므로 원래 데이터셋 순서가 유지된다.
    # (캐시 실험은 요청이 들어오는 순서가 결과에 영향을 주기 때문에 원래 순서를 지킨다)
    stream = df[df[args.label_col].isin(eval_groups)].reset_index(drop=True)
    path = os.path.join(args.out_dir, "eval_stream.parquet")
    stream.to_parquet(path, index=False)
    print(f"eval_stream: {len(stream)} prompts, {len(eval_groups)} groups (original order) -> {path}")


# 이 파일을 직접 실행했을 때만 main()을 호출한다.
# (다른 파일에서 import할 때는 자동 실행되지 않게 하는 파이썬 관용구)
if __name__ == "__main__":
    main()
