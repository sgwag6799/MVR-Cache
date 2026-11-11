# `claim-a-rule-weighting` 브랜치 노트

main 대비 이 브랜치에서 추가한 문서를 이 파일 하나로 합쳤다. 위쪽이 이 브랜치에서 쓴 문서, 아래쪽이 부모 브랜치(`prefix-exp7-invariant-encoder`)에서 물려받은 문서다.
각 부분의 제목 앞 이름이 원래 파일 이름이고, 절 번호는 원래 문서 그대로다 (코드 주석의 "X 부분 N절"이 이 번호를 가리킨다).

| 부분 | 원래 파일 |
|---|---|
| CLAIM_A | `CLAIM_A.md` |
| exp8_stage2 README | `mvr-cache/benchmarks/prefix/exp8_stage2/README.md` |
| PREFIX_EXPERIMENT | `PREFIX_EXPERIMENT.md` |
| PREFIX_PLAN | `PREFIX_PLAN.md` |

---

<!-- 원래 파일: CLAIM_A.md -->

# CLAIM_A 부분: 주장 A: 공통 문단 환경에서 규칙 분할 + 학습 가중치가 vCache · MVR-Cache보다 낫다

> 프롬프트 앞에 답과 무관한 공통 문단이 붙는 환경에서, **규칙 분할(구두점 위치, 상한 32) + 학습 가중치(M3)** 를 쓰는 MVR이 단일 벡터 vCache와 논문 MVR-Cache(RL 분할기)보다 **오류율 한도(δ = 0.01)를 지키면서 더 높은 히트율**을 낸다.

- 인코더: 모든 비교는 base 인코더 `BAAI/bge-base-en-v1.5` 기준
- 데이터셋 (공통 문단 붙이는 방식): PREFIX_EXPERIMENT 부분, 설계 원문 PREFIX_PLAN 부분
- 결과 파일: [`mvr-cache/benchmarks/prefix/exp8_stage2/`](mvr-cache/benchmarks/prefix/exp8_stage2/)

## 근거: 실험 8 (LmArena 41K 스트림, 공통 문장 8개 L8)

앞 2,000요청 제외 38,964요청, δ = 0.01, 시드 1 · 2 · 3 평균. 괄호 안은 오류율.

| 방식 | 조건 | L0 (원본) | L8 | L8 − L0 |
|---|---|---|---|---|
| vCache | A | 21.74% (0.07%) | 17.54% (0.18%) | −4.20%p |
| 논문 MVR-Cache (재학습 RL 분할기) | F | 20.24% (0.08%) | 18.54% (0.23%) | −1.70%p |
| 규칙 분할 + 균등 (M2) | C | 20.67% (0.09%) | 18.59% (0.18%) | −2.08%p |
| **규칙 분할 + 학습 가중치 (M3)** | **E** | 미실행 | **20.19%** (0.18%) | – |

L8에서 E − A = **+2.65%p**, E − F = **+1.65%p**, E − C = **+1.60%p**, 세 비교 모두 시드 3개에서 같은 방향 ([`claim_a_L8_comparison.csv`](mvr-cache/benchmarks/prefix/exp8_stage2/claim_a_L8_comparison.csv)).

## 약점과 정한 것

| 약점 | 정한 것 · 보강 |
|---|---|
| 1단계 사전 기준 미달, E vs A · F는 실험 8의 보조 분석, 데이터 · 정답 정의는 결과 본 뒤 선택 | 아래 판정 기준을 실행 전에 커밋으로 고정하고 다시 확인 (A1 · A2) |
| E의 L0 없음 | A1에서 L0 실행 |
| 데이터셋 1개 | A2에서 Classification 추가 |
| 논문 체크포인트 미공개 → RL 분할기 재학습 | 재학습 분할기를 MVR-Cache 기준선으로 쓰고 학습 곡선 · 조각 수 분포 보고 |
| 상한 32 적용에 공통 문장 목록 사용 (`cap_scope original`) | `CAP_SCOPE=all`로 목록 없이 실행 |
| M3 헤드가 test와 같은 공통 문장으로 학습 | A3: 학습용 풀(`prefix_pool_train.json`)로 헤드 학습 후 평가 풀로 평가 |
| `--sleep 0.02` (논문 MVR-Cache 0.1) | `SLEEP=0.1`, 모든 조건 같게. δ는 0.01 그대로 |

## 보강 실험과 판정 기준

논문 MVR-Cache 옵션: `--include-full-embedding`은 조각 벡터에 문장 전체 벡터 1개를 조각처럼 한 줄 더 붙이고, `--mix-fullcos`는 최종 점수를 0.5 × SMaxSim + 0.5 × 문장 전체 코사인으로 섞는다. 실험 8의 F는 README 예시대로 둘 다 썼다. **조건 J = F에서 두 옵션을 모두 뺀 것**(조각 벡터의 SMaxSim만, RL 분할기 · `multivector_top_k` k 10은 같음)을 추가한다. L8에서는 문장 전체 벡터의 약 3/4이 공통 문단이라, 두 옵션이 F에 불리하게 작용했는지 본다.

| ID | 작업 목록 | 설정 | 내용 |
|---|---|---|---|
| A0 + 채우기 | `stage2_jobs_claimA_fill.txt` | 실험 8과 같음 (sleep 0.02) | 재현 확인(A · C · E · F, L8 시드 1) + **E의 L0** + **J** |
| **A0′ (우선)** | `stage2_jobs_claimA_E_L0.txt` | 실험 8과 같음 (sleep 0.02, cap original) | **E의 L0만**: 시드 1 필수, 시드 2 · 3과 C의 L0 시드 2 · 3은 선택. 결과는 실험 8 결과 폴더(`data/exp8/runs`)에 같이 저장 |
| A1 | `stage2_jobs_claimA_sleep01.txt` | **sleep 0.1** | A · C · E · F · J, L8 시드 1 · 2 · 3 + L0 시드 1 |
| A1b | `stage2_jobs_claimA_capall.txt` | sleep 0.1, **cap all** | C · E (공통 문장 목록 없이) |
| A2 | – | A1과 같음 | Classification |
| A3 | – | A1과 같음 | 처음 보는 템플릿 E′ |
| A4 | – | – | 참고: vCache + LoRA 인코더(B) |

- E는 모든 단계에서 L8 헤드 하나를 쓴다 (공통 문단 수를 모르는 실제 서비스 조건).
- A0′ 실행: `bash scripts/run_stage2_lmarena.sh scripts/stage2_jobs_claimA_E_L0.txt 3` (약 1시간, 선택 포함 약 3시간). 집계: `python -m benchmarks.prefix.prefix_stage2_report --runs ../data/exp8/runs --out ../data/exp8/report` (mvr-cache에서).
- 실행: `nohup bash scripts/run_claimA_stage2.sh 5 > data/claimA/run.log 2>&1 < /dev/null &` (배치를 순서대로, 결과는 `mvr-cache/benchmarks/prefix/claimA_stage2/`로 자동 push). 예상 시간: A0 + 채우기 약 1.5시간, A1 약 8시간, A1b 약 3시간.

데이터마다:
1. L8에서 E 히트율 > A, F (시드 3개 모두)
2. E 오류율 ≤ δ (모든 단계 · 시드)
3. L0에서 E 히트율이 C보다 0.5%p 넘게 낮지 않음

## 파일

| 파일 | 역할 |
|---|---|
| `scripts/make_lmarena_splits.py` · `scripts/lmarena_merge_duplicate_sets.py` | LmArena 분할(지문 train `ec6e778ee76f` / val `411c8e1b6412` / test `3e5cc1fe8fa8`) · 합친 정답 `id_set_merged` |
| `mvr-cache/benchmarks/prefix/make_prefix_data.py` · `prefix_pool.json` | 공통 문단 데이터 L0 · L2 · L4 · L8 |
| `scripts/make_stage2_lmarena_data.py` | 스트림(L0 · L8 test) · RL 분할기 train/val 파일 |
| `scripts/train_rl_splitter_lmarena.sh` | MVR-Cache 기준선 RL 분할기 학습 (논문 레시피) |
| `mvr-cache/benchmarks/prefix/exp2_lmarena/offline_merged/prefix_w_variant_L8.pt` | E의 M3 가중치 헤드 (LmArena L8 train으로 학습) |
| `scripts/run_stage2_lmarena.sh` · `scripts/stage2_jobs_*.txt` | 스트림 실행 (조건 A~G · J, `SLEEP` · `CAP_SCOPE` · `HEAD` · `OUT` 환경변수) |
| `scripts/run_claimA_stage2.sh` · `scripts/autosave_stage2.sh` | 보강 배치를 순서대로 실행 · 결과 자동 push |
| `mvr-cache/benchmarks/prefix/prefix_stage2_report.py` | 히트율 · 오류율 집계 |
| `mvr-cache/vcache/vcache_core/splitter/PrefixRuleSplitter.py` | 규칙 분할기 + M3 헤드 |

데이터 파일(`data/`)은 git에서 제외되어 있다. 위 스크립트로 Hugging Face 원본(`vCache/SemBenchmarkLmArena`)에서 같은 파일을 다시 만든다. 데이터 만드는 방법은 PREFIX_EXPERIMENT 부분, 실험 실행은 `scripts/run_claimA_stage2.sh`.

---

<!-- 원래 파일: mvr-cache/benchmarks/prefix/exp8_stage2/README.md -->

# exp8_stage2 README 부분: 실험 8 결과 (2단계 41K 스트림, LmArena)

설계 · 결과 · 판정: CLAIM_A 부분.

| 파일 | 내용 |
|---|---|
| `stage2_runs.csv` | 실행별 히트율 · 오류율 · 정밀도 (조건 A~G × 단계 × 시드, 앞 2,000행 제외 38,964요청) |
| `stage2_summary.csv` | 조건 × 단계 평균 · 표준편차 · 오류율 최대 |
| `claim_a_L8_comparison.csv` | 주장 A: L8에서 E(규칙 분할 + 학습 가중치)와 A · F · C의 시드별 차이 |

- `stage2_runs.csv`는 Colab VM에서 `prefix_stage2_report.py`가 출력한 표(소수 6자리)에서 복원했다. 원본 실행 JSON · timing 파일 · RL 학습 로그는 VM과 함께 없어졌다.
- `stage2_summary.csv` · `claim_a_L8_comparison.csv`는 `stage2_runs.csv`에서 계산했다.
- 조건: A/B vCache base/LoRA, C/D 규칙 분할 균등 base/LoRA, E 규칙 분할 + 학습 가중치(M3) base, F/G 논문 MVR-Cache(재학습 RL 분할기) base/LoRA.

---

<!-- 원래 파일: PREFIX_EXPERIMENT.md -->

# PREFIX_EXPERIMENT 부분: 공통 문단 데이터셋 (LmArena)

LmArena 프롬프트 앞에 **답과 무관한 공통 문장**을 붙여서, "긴 공통 템플릿 + 짧은 결정 부분" 상황을 만든 데이터셋이다. 실험 설계 · 결과는 CLAIM_A 부분에 있다.

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

---

<!-- 원래 파일: PREFIX_PLAN.md -->

# PREFIX_PLAN 부분: 공통 문단 테스트: 조각 가중치가 필요한 조건 찾기 (실험 계획)

> 🎯 **한 줄 요약**: Classification 프롬프트 앞에 답과 무관한 공통 문단을 붙여 "긴 공통 템플릿 + 짧은 결정 부분" 상황을 만들고, 공통 문단이 길어질수록 vCache · 균등 MVR · 학습 가중치의 구분력과 히트율이 어떻게 변하는지 측정한다.
>
> 기대하는 바는 vCache와 MVR-Cache에서 수치 하락, 학습 가중치에서 수치 상승.

- 이 문서는 결과를 보기 전에 고정한 **설계 원문**이다.
- 코드로 옮긴 방법, 데이터로 확인한 설계 충돌, 실행 명령은 PREFIX_EXPERIMENT 부분에 있다.

## 목적

- 기존 데이터셋 3개에서는 조각 가중치의 효과가 작았다 (오라클 천장). 이 실험은 **가중치가 효과를 내는 조건이 존재하는지**를 통제된 환경에서 확인한다.
- 실제 LLM 서비스는 질문 앞에 시스템 프롬프트 · 안내문 · 예시를 붙인다. 이 공통 부분이 길수록 문장 벡터와 균등 평균은 공통 부분에 지배된다.
- 가설: 공통 문단이 길어질수록 vCache와 균등 MVR의 구분력이 떨어지고, 학습 가중치는 공통 조각의 비중을 낮춰 구분력을 되찾는다.

## 보고 위치

- **스트레스 테스트(통제 실험)**로 보고한다. "가중치가 효과 있다"의 주 근거가 아니라 "가중치는 어떤 조건에서 필요한가"의 근거다.
- 기존 데이터셋 3개 결과(효과 작음)와 함께 제시한다.
- 데이터 구성 규칙은 결과를 보기 전에 이 문서대로 고정한다.

## 데이터 구성

### 원본

- Classification (train3k / val1k / test41k, seed 42), 하위 과제 ecommerce · amazon · commonsense
- 라벨: 원본 그대로 (`response_llama_3_8b` 문자열 일치). 공통 문단은 답에 영향을 주지 않는 문장으로만 구성한다는 가정을 둔다.

### 공통 문단

- 영어 한 문장 단위, 문장당 15~20토큰, 문장 안에 구두점은 마지막 마침표 하나만
- 답과 무관한 문장만 사용: 역할 소개, 말투, 형식, 안전 안내 등 (예: "You are a helpful assistant for an online store." · "Please keep your answer short and polite.")
- 문장 칸(slot) 8개, 칸마다 바꿔 쓴 문장 5개 → 공통 문장 풀 40개 (`prefix_pool.json`, 실험 전 고정)
- **변형 방식**: 프롬프트마다 칸별로 5개 중 하나를 고정 시드로 뽑음. 공통 부분이 요청마다 조금씩 달라지는 실제 상황(사용자 정보, 대화 이력)을 흉내 내고, "공통 부분을 문자열로 잘라내면 된다"는 대안을 막는다.
- 고정 방식(모든 프롬프트에 같은 문장)은 1단계에서만 참고로 측정

### 길이 단계

| 단계 | 붙이는 공통 문장 수 | 추가 토큰 (약) |
| --- | --- | --- |
| L0 | 0 (원본) | 0 |
| L2 | 2 | 30~40 |
| L4 | 4 | 60~80 |
| L8 | 8 | 120~160 |

### 길이 제한

- bge-base-en-v1.5 최대 입력 512토큰. 넘으면 뒤쪽 원본 프롬프트가 잘린다.
- 단계마다 (공통 문단 + 원본) 토큰 수를 전수 계산해 **잘림 0건**을 확인한다. 잘림이 생기는 단계는 실험에서 제외한다.

## 비교 조건

| 조건 | 점수 계산 | 역할 |
| --- | --- | --- |
| R | 원본(L0)의 균등 MVR | 공통 문단이 없을 때의 기준 |
| M1 | vCache (문장 벡터 1개, CLS 포함 토큰 평균) | 단일 벡터 |
| M2 | MVR 균등 가중치 (논문 식 7) | 멀티벡터 기준 |
| M3 | MVR 학습 가중치 (Linear 768 → 1 + 조각끼리 softmax, 0 초기화) | 제안 |
| M4 | MVR, 공통 문단 조각 가중치 0 (어느 조각이 공통인지 아는 규칙) | 가중치로 얻을 수 있는 상한 |

- M3 학습: 오라클 천장 실험과 같은 절차 (BCE, 클래스 비율 보정, Adam 0.001, 배치 2,048, 20 에폭, val AUC 최고 시점). **단계마다** 그 단계의 train3k로 따로 학습
- M3은 어느 조각이 공통 문단인지 정보를 받지 않는다. 조각 벡터만 보고 배운다.

## 고정 설정

| 항목 | 값 |
| --- | --- |
| 분할 | 모든 구두점 위치 분할 (상한 32). 공통 문장은 항상 각각 한 조각이 되고, 원본 부분의 분할은 단계와 무관하게 같다 |
| 후보 검색 | 문장 벡터 1개로 HNSW 후보 10개 |
| 점수 | SMaxSim, (x+1)/2 변환, 전체 벡터 행 미사용, 전체 코사인 혼합 없음 |
| 판정 | vCache, δ 0.01, cache-on-miss, sleep 0.02 (판정자 미사용) |
| 임베딩 · 라벨 | bge-base-en-v1.5 · Llama 3 8B |
| 환경 | Colab L4, MVR-Cache 저장소 + 팀 패치 (커밋 해시 기록) |

## 진행 단계

### 0단계: 데이터 만들기 · 점검

1. `prefix_pool.json` 작성, 단계별 데이터 생성 (시드 고정)
2. 점검: 단계별 토큰 수 분포와 잘림 0건, 조각 수 분포, 원본 부분 분할이 L0와 같은지

### 1단계: 오프라인 진단 (쌍 단위)

- 평가 쌍: test41k 앵커 6,000 × 문장 벡터 최근접 10 = 6만 쌍 (단계마다 다시 검색)
- 단계(L0 · L2 · L4 · L8) × 조건(M1 · M2 · M3 · M4)마다 측정
    - 검색 천장@10: 후보 10개 안에 정답 후보가 있는 비율
    - 1등 정답률: 점수 1등 후보가 정답인 비율
    - 선택된 이웃 (s, c)의 AUC
    - TPR@FPR1% (팀 쌍 단위 대리 지표)
    - 하위 과제별 같은 지표
- 고정 방식 공통 문단(L4)도 같은 지표로 1회 측정

### 2단계: 41K 스트림 (1단계 통과 시)

- 단계: L0 + 1단계에서 통과한 단계 중 가장 긴 것 1개
- 조건: M1 · M2 · M3, 시드 3개 (같은 시드끼리 짝지어 비교)
- 공식 집계 39,000행 (앞 2,000행 제외)
- 지표
    - 히트율, 오류율(≤ δ 확인), 검색 천장, 과제별 히트율
    - 지연시간: 임베딩 · HNSW 검색 · SMaxSim 재순위 · 가중치 계산 · vCache 판정 (요청당 ms)
    - 저장 벡터 수, 메모리

## 통과 기준 (1단계 → 2단계)

- 어떤 단계 L(≥ L4)에서 M2의 AUC가 L0 대비 **0.03 이상 하락**
- 그 단계에서 M3의 회복률 = (M3 − M2) / (R − M2)가 **0.5 이상** (AUC 기준)
- 둘 다 만족하는 단계가 없으면 2단계를 진행하지 않고 1단계 결과를 최종 결과로 기록

## 결과 해석 (사전 고정)

| 결과 | 해석 |
| --- | --- |
| M2가 단계가 올라가도 떨어지지 않음 | 문맥 임베딩이 공통 문단 영향을 흡수함. 이 조건에서도 가중치는 필요 없음 (음성 결과) |
| M2 하락, M3 회복, M3 ≈ M4 | 학습 가중치가 공통 조각을 스스로 낮춤. "긴 공통 템플릿 환경에서 가중치가 필요하다"는 조건부 양성 결과 |
| M2 하락, M3 일부 회복, M3 < M4 | 가중치로 얻을 수 있는 이득 중 일부만 학습됨. 모델 구조 · 학습 데이터 한계 |
| M2 하락, M4도 회복 못 함, 검색 천장 하락 | 병목이 후보 검색 단계(문장 벡터 1개)에 있음. 재순위 가중치로는 해결 불가 |
| 1단계 회복이 2단계 히트율로 이어지지 않음 | 판정 단계(cold 규칙, 경로 의존)가 이득을 막음 |

## 한계 (보고 시 함께 명시)

- 공통 문단은 답에 영향을 주지 않는다고 가정하고 원본 라벨을 그대로 쓴다. 실제 LLM은 시스템 프롬프트에 따라 답이 달라질 수 있다.
- 인공적으로 구성한 데이터라 실제 서비스 분포와 다르다.
- 실제 시스템은 시스템 프롬프트를 캐시 키에서 빼는 경우가 많다. 이 실험은 공통 부분이 요청마다 조금씩 달라 문자열로 분리할 수 없는 경우를 대상으로 한다.

## 산출물

- `prefix_pool.json`: 공통 문장 풀 (칸 8개 × 5개)
- `cls_prefix_L{0,2,4,8}.parquet`: 단계별 데이터
- `prefix_sanity.json`: 토큰 수 · 잘림 · 조각 수 점검
- `prefix_offline.csv`: 단계 × 조건 × 과제별 1단계 지표
- `prefix_w_L{2,4,8}.pt`: 단계별 학습 가중치
- (2단계) 시드별 41K 결과, 지연시간 표, `prefix_run_meta.json`

## 일정

- 2026-10-19 → 2026-10-23: 0단계 데이터 생성 · 점검
- 2026-10-26 → 2026-10-30: 1단계 오프라인 진단, 통과 여부 결정
- 2026-11-02 → 2026-11-13: 2단계 41K 3시드 (통과 시)
