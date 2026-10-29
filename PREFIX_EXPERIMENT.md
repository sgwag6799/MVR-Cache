# 공통 문단 데이터셋 (LmArena)

LmArena 프롬프트 앞에 **답과 무관한 공통 문장**을 붙여서, "긴 공통 템플릿 + 짧은 결정 부분" 상황을 만든 데이터셋이다. 실험 설계 · 결과는 [`CLAIM_A.md`](CLAIM_A.md)에 있다.

## 1. 원본

| 항목 | 내용 |
|---|---|
| 데이터 | `vCache/SemBenchmarkLmArena` (Hugging Face) 63,796행 |
| 정답 | 같은 뜻의 질문 묶음이면 정답. 원본 묶음 번호(`ID_Set`)에서, 글자까지 같은 질문이 서로 다른 묶음에 들어간 경우(186개)를 하나로 합친 `id_set_merged`를 쓴다 |
| 분할 | 묶음 단위로 나눔 (같은 묶음은 한 분할에만): train 3,007 / val 1,000 / test 40,964행 |

## 2. 공통 문단을 붙이는 방식

### 공통 문장 풀 (`mvr-cache/benchmarks/prefix/prefix_pool.json`)

칸 8개 × 변형 5개 = 40문장. 모두 영어 한 문장, 15~20토큰, 구두점은 끝 마침표 하나다 (그래서 문장 하나가 조각 하나가 된다).

| 칸 | 변형 0번 문장 (예) |
|---|---|
| role | You are a helpful assistant that answers questions for users of an online service. |
| tone | Please keep your answer short and polite and avoid using any unnecessary words in the reply. |
| format | Return only the final answer without adding any additional text or special formatting to it. |
| safety | Do not share harmful content and politely refuse any requests that could cause real damage. |
| privacy | Do not ask for or reveal personal information such as names or home addresses. |
| language | Reply in English even if the user writes the request in a different language. |
| accuracy | Base the answer only on the given input and do not guess any missing facts. |
| session | This conversation is part of an automated evaluation run that is performed on the service. |

### 붙이는 규칙

1. 프롬프트마다 칸별로 변형 5개 중 하나를 고른다. 고르는 난수는 (시드 42, `dataset_name:id`)로 정해서, 같은 행은 항상 같은 문장을 받는다.
2. 단계 **L**이면 앞에서부터 L칸의 문장을 쓴다. 따라서 L2의 문장은 L8의 앞 두 문장과 같다.
3. `공통 문장들(공백으로 연결) + " " + 원본 프롬프트`로 붙인다. 원본 프롬프트와 정답은 바꾸지 않는다.
4. 가장 긴 단계(L8)에서도 512토큰 안에 드는 행만 모든 단계에서 똑같이 쓴다 (`fits_all_levels`).

### 단계

| 단계 | 내용 | 평균 길이 (test) |
|---|---|---|
| **L0** | 원본 그대로 | 47토큰 |
| L2 · L4 | 공통 문장 2 · 4개 | – |
| **L8** | 공통 문장 8개 | 176토큰 (공통 문단 약 129토큰) |
| 고정 L4 | 모든 프롬프트에 칸마다 0번 문장 (참고용) | – |

### 예시 (L8, 칸마다 0번 문장이 뽑힌 경우)

```
You are a helpful assistant that answers questions for users of an online service. Please keep your
answer short and polite and avoid using any unnecessary words in the reply. Return only the final answer
without adding any additional text or special formatting to it. Do not share harmful content and
politely refuse any requests that could cause real damage. Do not ask for or reveal personal information
such as names or home addresses. Reply in English even if the user writes the request in a different
language. Base the answer only on the given input and do not guess any missing facts. This conversation
is part of an automated evaluation run that is performed on the service. How many sisters does each of
Alice's brothers have?
```

## 3. 만드는 방법

GPU는 필요 없다 (토크나이저만 사용, 수 분).

```bash
git clone -b claim-a-rule-weighting https://github.com/sgwag6799/MVR-Cache.git && cd MVR-Cache
source colab_setup.sh
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkLmArena','train.parquet',repo_type='dataset',local_dir='data/raw_lmarena')"
python scripts/make_lmarena_splits.py --raw data/raw_lmarena/train.parquet --out data/lmarena
(cd mvr-cache && python -m benchmarks.prefix.make_prefix_data --train ../data/lmarena/train.parquet \
  --val ../data/lmarena/val.parquet --test ../data/lmarena/test.parquet --out ../data/lmarena/prefix/data)
python scripts/lmarena_merge_duplicate_sets.py --raw data/raw_lmarena/train.parquet --data data/lmarena/prefix/data
python scripts/make_stage2_lmarena_data.py    # 캐시 스트림용 L0 · L8 test 파일 (data/exp8/stream/)
```

분할이 같은지는 `make_lmarena_splits.py`가 출력하는 지문으로 확인한다: train `ec6e778ee76f` / val `411c8e1b6412` / test `3e5cc1fe8fa8`.

## 4. 결과 파일

| 파일 | 내용 |
|---|---|
| `data/lmarena/prefix/data/variant/L{0,2,4,8}/{train,val,test}.parquet` | 단계별 데이터 |
| `data/lmarena/prefix/data/fixed/L4/{train,val,test}.parquet` | 고정 방식 (참고용) |
| `data/exp8/stream/L{0,8}_test.parquet` | 캐시 스트림 입력 (`id_set` = `id_set_merged`) |

원본 열은 그대로이고, 다음 열이 추가된다: `orig_prompt`(원본 프롬프트), `prefix_text`(붙인 공통 문단), `prefix_choice`(칸별로 고른 변형 번호), `level`, `n_tokens`, `truncated`, `fits_all_levels`, `id_set_merged`.
