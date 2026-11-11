# `cls-scale-decomp` 브랜치 노트

`prefix-stress-test`에서 딴 브랜치다. 위쪽이 이 브랜치에서 쓴 문서, 아래쪽이 부모 브랜치(`prefix-stress-test`)에서 물려받은 문서다.
각 부분의 제목 앞 이름이 원래 파일 이름이고, 절 번호는 원래 문서 그대로다 (코드 주석의 "X 부분 N절"이 이 번호를 가리킨다).

| 부분 | 내용 |
|---|---|
| DECOMP | 이 브랜치: 바닐라 vCache ↔ MVR 파이프라인 차이 분해 (CLS · 점수 범위 · 후보 수) |
| PREFIX_EXPERIMENT | 부모 브랜치 문서 (`PREFIX_EXPERIMENT.md`) |

---

<!-- 이 브랜치 문서 -->

# DECOMP 부분: 바닐라 vCache ↔ MVR 파이프라인 차이 분해

> 판정자를 쓰지 않는 실험이다. 실행 전 설계 · 판정 기준을 여기에 고정한다. 결과를 보고 기준을 바꾸지 않는다.

## 0. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 질문 | 같은 vCache 판정 규칙인데 바닐라 vCache(V)와 MVR 파이프라인(B0, 분할 0)의 히트율 · 오류율이 왜 다른가 |
| 요인 | ① [CLS] 토큰을 임베딩 평균에 넣는가 ② 점수 범위: 코사인 그대로 vs (코사인+1)/2 ③ 후보 수: 1개 vs 10개 재정렬 |
| 칸 | V · P1 ~ P5 (3절). V → P5가 바닐라 → B0이고, 한 번에 한 요인씩 바꾼다 |
| 데이터 | Classification test41k(공식 39K 집계) · LmArena test 56,048행(공식 54,048행 집계). SearchQueries는 이번에 뺀다(1회 3~5시간) |
| 고정 | 순서 1(파일 순서) · 시드 1 · δ 0.01 · cache-on-miss · bge-base-en-v1.5 · sleep 0.02 · 판정자 없음 |
| 반복 | 이번에는 없음. "차이 큼"(6절) 요인만 순서 2·3 반복과 판정자 조건으로 확장 |
| 실행 | RunPod. 데이터셋 하나당 Pod 하나, Pod 안에서는 칸을 하나씩 순서대로 |

## 1. 배경 (기존 기록)

| 기록 | 내용 |
|---|---|
| 시맨틱 캐싱 실험 기록 (Classification 45K · A5000 · 1회) | A(바닐라) 14.1% → A′(CLS 제외 + (x+1)/2, 후보 1) 16.5% (+2.4%p) → B(후보 10) 17.3% (+0.8%p) |
| 판정자 실험 A (Classification · L4 · 시드 1) | V 14.446% / 0.551% vs B0 17.244% / 0.590% → B0가 +2.80%p |
| 판정자 실험 H (LmArena · L4 · 순서 1) | 같은 δ 1%: V 18.97% / 0.113% vs B0 18.24% / 0.157%. **같은 실제 오류율**에서는 B0가 7 ~ 9%p 낮다 |
| 실험 H의 원인 후보 | 풀링 · 점수 척도 · 후보 검색 경로 (미측정) |

LmArena는 같은 δ에서 히트율보다 **오류율** 차이로 드러난다(0.113 vs 0.157%). 그래서 판정에 오류율 비를 함께 쓴다(6절).

## 2. 코드 확인

| | 바닐라 (`benchmarks/eval_sembenchmark_verified.py` → `verified.py`) | MVR 파이프라인 (`eval_sembenchmark_verified_splitter.py` → `verified_splitter.py`) |
|---|---|---|
| 후보 | HNSW 최근접 1개 (`get_knn(emb, k=1)`) | HNSW로 k개 → 조각 SMaxSim으로 재정렬해 1등 |
| 임베딩 | `EmbeddingModel.get_embedding`: [CLS] · [SEP] 포함 전체 토큰 평균 | 후보 검색은 같은 벡터, 점수는 조각 벡터(`_segment_embeds_from_pointers`: [CLS] 제외) |
| 점수 | 1 − HNSW 코사인 거리 = 코사인 그대로 | 대칭 SMaxSim → (x+1)/2 |
| 판정 | `_Algorithm` (6개 미만 EXPLORE, 항목별 로지스틱, τ) | 같은 `_Algorithm`을 가져다 씀 |

- 이 브랜치의 바닐라 경로는 main과 판정 코드가 같다(`verified.py` 변경 없음). 차이는 vllm 없이 import, `--seed`, `--filter-true-column`뿐이다.
- 점수 범위만 바꿔도 판정이 달라질 수 있는 곳: 로지스틱은 정규화가 없어 이론상 척도와 무관하지만 ① 완전 분리일 때 고정 분산값(`variance_map`) ② `round(s, 3)` ③ t_hat을 [0, 1]로 자르기는 척도에 따라 달라진다 *(추정)*.

## 3. 칸 설계

모든 P칸은 분할 0(`--rule-max-cuts 0 --rule-cap-scope all` → 조각 1개 = 문장 전체), `--candidate-selection top_k`.

| 칸 | CLS | 점수 범위 | 후보 | 의미 |
|---|---|---|---|---|
| V | 포함 | 코사인 | 1 | 바닐라 vCache (main 판정 코드) |
| P1 | 포함 | 코사인 | 1 | V를 MVR 경로로 재현 → **코드 경로 확인** (V와 같아야 함) |
| P2 | 포함 | (x+1)/2 | 1 | P2 − P1 = 점수 범위 효과 (CLS 포함일 때) |
| P3 | 제외 | 코사인 | 1 | P3 − P1 = CLS 효과 (코사인일 때) |
| P4 | 제외 | (x+1)/2 | 1 | = 기록 1의 A′. P4 − P3 · P4 − P2로 두 효과의 반대쪽 칸 |
| P5 | 제외 | (x+1)/2 | 10 | = B0 재현. P5 − P4 = 후보 수 효과 |

- 2×2(P1 ~ P4)에서 CLS 주효과 = ((P3 − P1) + (P4 − P2)) / 2, 점수 범위 주효과 = ((P2 − P1) + (P4 − P3)) / 2, 상호작용 = (P4 − P3) − (P2 − P1)
- 후보 검색 벡터는 모든 칸에서 [CLS] 포함 벡터(HNSW)다. 후보 1개 칸에서 CLS를 빼면 "1등은 CLS 포함 벡터로 찾고, 그 점수만 CLS 제외 벡터로 낸다"는 뜻이다(기록 1의 A′와 같은 정의). CLS 포함 + 후보 10은 재정렬해도 HNSW 1등과 거의 같아 칸을 두지 않는다.
- V와 P5를 같이 돌리는 이유: RunPod GPU가 기존 기록(L4)과 다를 수 있어, 비교는 **이 실험 안의 칸끼리만** 한다.

## 4. 코드 변경 (이 브랜치)

| 파일 | 변경 | 기본값 |
|---|---|---|
| `vcache/vcache_core/splitter/PrefixRuleSplitter.py` | `include_cls`: 첫 조각 평균을 [CLS]부터. 분할 0이면 조각 벡터 = 바닐라 임베딩(`pooled_knn`) | False (원본과 같음) |
| `vcache/vcache_policy/strategies/verified_splitter.py` | `score_scale`: `unit` = (x+1)/2, `raw` = 코사인 그대로(−1 ~ 1). `mix_fullcos`와 같이 쓰면 오류. 후보 1등 초기값을 −1 → −∞(raw에서도 안전) | `unit` (원본과 같음) |
| `benchmarks/eval_sembenchmark_verified_splitter.py` | 인자 `--segment-include-cls` · `--score-scale {unit,raw}` | 원본과 같음 |
| `scripts/decomp/run_decomp.sh` | 6칸 실행기. 칸마다 `<칸>.json`(per_sample) · `<칸>.log` · `<칸>.meta.json`(명령 · 커밋 · GPU · 버전 · 데이터 지문). 끝난 칸은 건너뜀 | |
| `scripts/decomp/report_decomp.py` | 공식 집계(앞 2,000행 제외), 오류율 95% 상한, 요인 효과, 6절 판정, P1 ↔ V 요청별 일치율. `report.md` · `report.csv` | |
| `scripts/decomp/selftest.py` | 본 실행 전 점검: 풀링 · 점수 범위 단위 시험, `--with-model`이면 실제 BGE로 "CLS 포함 분할 0 벡터 = 바닐라 벡터" 확인 | |

## 5. 실행 (RunPod)

1. Pod: **두 대 모두 같은 GPU 종류 · 같은 이미지**. Pod 1 = Classification, Pod 2 = LmArena. 한 Pod에 다른 실험을 같이 올리지 않는다(관측 갱신이 백그라운드 스레드라 느려지면 판정이 달라질 수 있다 — 평가 코드의 `--sleep` 주석).
2. 설치 (저장소 루트):
   ```bash
   git clone -b cls-scale-decomp https://github.com/sgwag6799/MVR-Cache.git && cd MVR-Cache/mvr-cache
   pip install -q tokenizers transformers pyarrow hnswlib statsmodels rl4co tensordict pandas tqdm scikit-learn
   python ../scripts/decomp/selftest.py --with-model
   ```
3. 데이터: GitHub에 올리지 않는다. Drive `mvr_splits/`에서 Pod로 복사한다.
   - Classification: `cls_seed42_nodedup/test41k.parquet` (41,000행, 기존 실험과 같은 파일)
   - LmArena: `lmarena_seed42/lmarena_test_seed42_repro.parquet` (56,048행, 지문 `1f38e5aad950`)
4. 실행 (tmux 안에서):
   ```bash
   DATA=/workspace/data/test41k.parquet DS=cls OUT=/workspace/decomp/cls_o1 bash scripts/decomp/run_decomp.sh
   DATA=/workspace/data/lmarena_test_seed42_repro.parquet DS=lm OUT=/workspace/decomp/lm_o1 bash scripts/decomp/run_decomp.sh
   python scripts/decomp/report_decomp.py /workspace/decomp/cls_o1
   ```
5. 시간 *(추정, L4 기준)*: Classification 칸당 약 30 ~ 40분 × 6 ≈ 3.5시간, LmArena 칸당 약 55분 × 6 ≈ 5.5시간. 두 Pod 병렬.
6. 끝나면 `OUT` 폴더(json · log · meta · report)를 Drive에 올린다.

## 6. 판정 기준 (실행 전 기록)

| 확인 | 기준 | 결과에 따른 처리 |
|---|---|---|
| 유효성 | 모든 칸의 데이터 지문 · GPU · 커밋이 같음 | 다르면 비교 무효, 다시 돌림 |
| 코드 경로 (P1 ↔ V) | 요청별 히트 일치율 ≥ 99% 이고 \|Δ히트율\| < 0.4%p | 통과면 V와 P1을 같은 출발점으로 본다. 미통과면 2×2는 P1 기준으로만 읽고, 경로 차이(HNSW 점수 vs torch 코사인의 반올림 등)를 먼저 확인한다 |
| 요인별 "차이 큼" | \|Δ히트율\| ≥ 1.0%p (재실행 변동 약 0.4%p의 2.5배) **또는** 오류율 비가 0.8 ~ 1.25 밖 | "차이 큼"인 요인만 7절로 확장. 아니면 "1회 기준 작음"으로 보고 해석하지 않는다 |
| 보고 | 칸별 히트율 · 오류율(95% 상한), 요인 효과, 2×2 주효과 · 상호작용, P1 ↔ V 일치율 | 판정에는 히트율 · 오류율 비만 쓴다 |

## 7. 확장 계획 (6절 결과에 따라)

- "차이 큼" 요인이 있는 데이터셋: 그 요인의 두 칸을 순서 2·3(기존 실험 H와 같은 `default_rng(1000 + k)` 섞기)으로 반복
- 같은 요인을 판정자 조건(대체형)에서도 확인할지는 반복 결과를 보고 정한다
- LmArena처럼 오류율 차이가 큰 경우, 같은 실제 오류율 비교가 필요하면 해당 칸의 δ 곡선 점(δ 0.02 · 0.03)을 추가한다

## 8. 한계 · 실행 전 확인할 것

- **B0 인자 확인**: 팀 B0 기록(`judge_a/runs/B0_s1` 의 meta)의 후보 방식이 `top_k`인지 `multivector_top_k`인지, `--include-full-embedding` · `--mix-fullcos` 여부를 확인한다. 분할 0이면 조각 벡터 = 전체 벡터라 앞의 두 옵션은 점수가 같지만, `multivector_top_k`면 후보를 **CLS 제외 벡터**로 찾으므로 P5와 다르다 → 그 경우 칸 P6(`--candidate-selection multivector_top_k --candidate-k 10`)을 추가한다
- 순서 1 · 1회 결과다. 1%p보다 작은 차이는 해석하지 않는다
- δ 하나(0.01)라 같은 실제 오류율 비교는 하지 않는다. 오류율 비로 보완한다
- GPU가 기존 기록(L4)과 다르면 기존 V · B0 수치와 직접 비교하지 않는다
- SearchQueries는 이번 범위 밖이다

---

<!-- 부모 브랜치(prefix-stress-test) 문서 -->

<!-- 원래 파일: PREFIX_EXPERIMENT.md -->

# PREFIX_EXPERIMENT 부분: 공통 문단 스트레스 테스트 (`prefix-stress-test` 브랜치)

> **답변은 한국어로 해 주세요.** `main`(원저자 코드, `8cfaa0b`)에서 딴 브랜치다. `gpu-production`(규칙 분할 예외 1~6, 로그 A~F)과 `dialect-cache-experiment`와는 별개다.
> 설계 원문: 노션 "공통 문단 테스트: 조각 가중치가 필요한 조건 찾기". 이 문서는 그 설계를 코드로 옮긴 방법과, 데이터로 확인해 보니 설계와 충돌한 부분을 적는다.

## 0. 한눈에 보기

| | 내용 |
|---|---|
| 질문 | 프롬프트 앞에 답과 무관한 공통 문단이 길게 붙으면 단일 벡터·균등 MVR의 구분력이 떨어지고, 학습 가중치가 그걸 되찾는가 |
| 데이터 | Classification train3k / val1k / test41k (팀 분할 cls_seed42), 공통 문단 L0·L2·L4·L8 |
| 조건 | M1 단일 벡터 · M2 균등 MVR · M3 학습 가중치 · M4 공통 조각 가중치 0(상한), R = L0의 M2 |
| 1단계 | 오프라인 쌍 진단 (앵커 6,000 × 최근접 10) → 통과 기준 판정 |
| 2단계 | 통과 시 41K 스트림에서 vCache 실행 (M1·M2·M3, 시드 3개) |

## 1. 파일

| 파일 | 역할 |
|---|---|
| `mvr-cache/benchmarks/prefix/prefix_pool.json` | 공통 문장 풀: 칸 8개 × 문장 5개. 모두 15~18토큰, 구두점은 끝 마침표 하나 (검사 완료). L8 평균 약 129토큰 |
| `mvr-cache/vcache/vcache_core/splitter/prefix_split.py` | **분할 규칙 한 곳에 모음.** 0·1·2단계가 모두 이 함수로 자른다 (자르는 위치, 상한, 공통 조각 표시, 정답 정규화) |
| `mvr-cache/benchmarks/prefix/make_prefix_data.py` | 0단계: 단계별 데이터 만들기 (모델 불필요, 토크나이저만) |
| `mvr-cache/benchmarks/prefix/prefix_sanity.py` | 0단계 점검: 토큰 수·잘림·조각 수·원본 부분 분할 불변 (모델 불필요) |
| `mvr-cache/benchmarks/prefix/prefix_offline.py` | 1단계: 인코딩 → 쌍 → M1~M4 점수 → 지표 → M3 학습 → 통과 판정 (GPU) |
| `mvr-cache/vcache/vcache_core/splitter/PrefixRuleSplitter.py` | 2단계: 같은 규칙으로 자르는 분할기 + M3 가중치 헤드 |
| `mvr-cache/benchmarks/prefix/prefix_stream_report.py` | 2단계 집계: 조건 × 시드 × 과제, 같은 시드 짝 차이 |
| `eval_sembenchmark_verified_splitter.py` (수정) | `--splitter-mode prefix_rule`, `--rule-max-cuts`, `--rule-cap-scope`, `--prefix-pool`, `--segment-head`, `--filter-true-column`, `--seed` |
| `eval_sembenchmark_verified.py` (수정) | M1용: `--filter-true-column`, `--seed` |
| `verified_splitter.py` (수정) | 분할기가 `segment_weights`를 주면 가중 MaxSim (없으면 원본과 동일) |
| `inference_engine/*` (수정) | vllm 없이 import 되게 (팀 패치·gpu-production과 같은 수정) |

## 2. 설계를 코드로 옮긴 방법

### 2.1 공통 문단
- 프롬프트마다 칸 8개 각각에서 문장 하나를 **(시드 42, dataset_name:id)로 정한 난수**로 고른다. 8칸을 항상 다 뽑아 두고 L칸만 쓰므로 **L2의 문장 = L8의 앞 두 문장**이다 (단계 사이 비교가 깔끔).
- 붙이는 방식: `공통 문장들(공백으로 연결) + " " + 원본 프롬프트`. 원본 프롬프트·라벨·다른 열은 그대로.
- 고정 방식: 모든 프롬프트가 칸마다 0번 문장. L4만 만든다.

### 2.2 분할 (설계: "모든 구두점 위치 분할, 상한 32")
- 자르는 토큰: `. , ! ? ; :` (영문 6개). 위치 1 ~ 길이−3 (`[CLS]`, `[SEP]` 바로 앞 문장 끝 구두점, `[SEP]` 제외). 팀 `rule_pointers`와 같은 규칙.
- 조각 벡터: 구간별 토큰 평균(`[CLS]` 제외, 마지막 조각에 `[SEP]` 포함) = `MaxSimSplitter._segment_embeds_from_pointers`.
- 전체 문장 행 없음, 전체 코사인 혼합 없음, 점수는 대칭 가중 SMaxSim → (x+1)/2.
- 단일 벡터(M1·후보 검색): `[CLS]`·`[SEP]` 포함 토큰 평균 = `EmbeddingModel.get_embedding`.

### 2.3 조건
- **M3:** Linear(768→1), 0 초기화, 프롬프트 안 조각끼리 softmax. 단계마다 그 단계의 train으로 학습. 학습 쌍 = train 각 행 × train 안 최근접 10 (검증 = val 각 행 × val 안 최근접 10). BCE(양성 비율 보정), Adam 1e-3, 배치 2,048, 20 에폭, val AUC 최고 에폭. 로짓은 `a·s + b`(a, b도 학습)로 만들며 순위에는 영향 없음. **어느 조각이 공통 문단인지는 받지 않는다.**
- **M4:** 공통 문단 조각의 가중치 0, 나머지 균등 (공통 문단 끝 위치를 알고 있는 규칙).
- 정답 c: 응답을 vCache 판정과 같은 방식으로 정규화(공백·끝 마침표·따옴표 제거, 소문자)한 뒤 일치.

### 2.4 1단계 지표 (단계 × 조건 × 과제 ALL·3개)
`ceiling10`(후보 10개 안 정답 존재), `top1_acc`, **`auc_selected`**(앵커마다 1등 후보의 (s, c)로 AUC), `auc_pairs`(6만 쌍 전체 AUC), `tpr_at_fpr1`(6만 쌍, 팀 대리 지표). 앵커는 모든 단계에서 같은 행(시드 0), 최근접은 단계마다 다시 검색.

### 2.5 통과 기준 (1→2단계, `prefix_gate.json`)
- L ≥ 4 에서 `R − M2 ≥ 0.03` 이고 `(M3 − M2)/(R − M2) ≥ 0.5` (기본 지표 `auc_selected`)
- 통과한 단계 중 가장 긴 것을 2단계에 쓴다 (`stage2_level`)

## 3. 0단계 점검 결과 (샘플: test 2,000 / train 500 / val 500, 토크나이저만 사용)

| 항목 | 결과 |
|---|---|
| 공통 문장마다 정확히 한 조각 | ✅ 모든 단계에서 오류 0 |
| **L0에서 이미 512토큰 초과(잘림)** | test 3.5% (70/2,000), L8에서 6.7% |
| L8에서도 512토큰 이하인 행 (`fits_all_levels`) | 93.3% (아마존 92.5%, 상품 분류 92.5%, 상식 100%) |
| `fits_all_levels` 행 + 상한을 원본 부분에만 적용 | **원본 부분 분할이 L0와 100% 같음** |
| 상한을 맨 앞부터 셀 때 (`cap_scope all`) | L8에서 13% 행의 원본 분할이 바뀜 |
| 상한 32에 걸리는 행 | L0 test 약 8% |
| 조각 수 평균 (test) | L0 10.7 → L2 12.7 → L4 14.7 → L8 18.6 |
| 2단계 분할기의 공통 문단 자동 인식 | 15,000행 모두 기록값과 일치 |

## 4. 설계와 다르게 정한 것 (팀 확인 필요)

1. **잘림 규칙:** 설계는 "잘림이 생기는 단계는 제외"인데 L0부터 잘림이 있어 그대로 적용하면 모든 단계가 빠진다. → **L8에서도 잘리지 않는 행(약 93%)만 모든 단계에서 똑같이 사용**한다 (`--row-filter fits_all_levels`, 2단계는 `--filter-true-column fits_all_levels`). 그래서 2단계 스트림은 41,000행이 아니라 약 38,000행이고, 앞 2,000행을 빼면 공식 집계는 약 36,000행이다.
2. **상한 32의 범위:** 맨 앞부터 세면 공통 문단이 상한을 써 버려 "원본 부분 분할은 단계와 무관하게 같다"가 깨진다. → **상한은 원본 부분의 자르는 위치에만 적용**한다 (`cap_scope original`). 공통 문단 조각은 최대 8개라 메모리 영향 작음.
3. **통과 기준의 "AUC":** 설계에 어떤 AUC인지 없어서 **`auc_selected`(선택된 이웃의 (s, c) AUC)**로 정했다. `auc_pairs`도 함께 기록하므로 바꾸려면 `prefix_offline.csv`로 다시 판정하면 된다.
4. **환경:** 설계는 "저장소 + 팀 패치"인데 이 브랜치는 **main + 이 실험에 필요한 최소 수정**이다. 팀 패치의 consensus 정책 등은 들어 있지 않다. 분할 규칙·가중치 헤드 구조는 팀 패치와 같게 맞췄다.
5. **시드:** main은 vCache 탐색/활용 난수를 고정하지 않는다. `--seed`로 전역 난수를 고정하지만 관측 갱신이 백그라운드 스레드라 완전히 재현되지는 않는다.
6. **2단계 저장 벡터 수·메모리:** main 평가 스크립트는 기록하지 않는다. 지연시간은 `--timing-output`(단계별 합계 → 요청당 평균)으로 본다.

## 5. Colab 실행 (L4 GPU 권장, T4도 가능)

### 셀 1: 설치
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone -b prefix-stress-test https://github.com/sgwag6799/MVR-Cache.git /content/MVR-Cache
!pip install -q tokenizers transformers pyarrow hnswlib statsmodels rl4co tensordict
%cd /content/MVR-Cache/mvr-cache
```

### 셀 2: 0단계 (데이터 + 점검, 수 분)
```python
SPL = '/content/drive/MyDrive/mvr_splits'
OUT = '/content/drive/MyDrive/mvr_eval/prefix'
!python -m benchmarks.prefix.make_prefix_data \
  --train {SPL}/cls_seed42/train.parquet --val {SPL}/cls_seed42/val.parquet \
  --test {SPL}/cls_seed42_nodedup/test41k.parquet --out {OUT}/data
!python -m benchmarks.prefix.prefix_sanity --data {OUT}/data
```

### 셀 3: 1단계 (먼저 `--quick 300`으로 확인 → 본 실행)
```python
!python -m benchmarks.prefix.prefix_offline --data {OUT}/data --out {OUT}/offline_quick --quick 300 --epochs 2
!python -m benchmarks.prefix.prefix_offline --data {OUT}/data --out {OUT}/offline
!cat {OUT}/offline/prefix_gate.json
```

### 셀 4: 2단계 (통과한 단계 L, 시드 1·2·3)
```bash
D=$OUT/data/variant/L8/test.parquet   # prefix_gate.json 의 stage2_level 로 바꿀 것 (L0도 같은 방식으로 실행)
COMMON="--dataset $D --llm-col response_llama_3_8b --similarity-evaluator string --delta 0.01 --sleep 0.02 --filter-true-column fits_all_levels"
# M1 (vCache, 단일 벡터)
python benchmarks/eval_sembenchmark_verified.py $COMMON --device cuda --seed 1 --output-json $OUT/stream/L8_M1_s1.json
# M2 (균등 MVR)
python benchmarks/eval_sembenchmark_verified_splitter.py $COMMON --splitter-mode prefix_rule --rule-max-cuts 32 \
  --rule-cap-scope original --prefix-pool benchmarks/prefix/prefix_pool.json \
  --candidate-selection top_k --candidate-k 10 --use-cached-candidate-segments --splitter-device cuda \
  --seed 1 --timing-output $OUT/stream/timing.txt --output-json $OUT/stream/L8_M2_s1.json
# M3 (학습 가중치): M2 명령 + --segment-head $OUT/offline/prefix_w_variant_L8.pt, 출력 L8_M3_s1.json
python -m benchmarks.prefix.prefix_stream_report --dataset $D --filter-true-column fits_all_levels \
  M1:1=$OUT/stream/L8_M1_s1.json M2:1=$OUT/stream/L8_M2_s1.json M3:1=$OUT/stream/L8_M3_s1.json ...
```
- 1회 약 1시간대(41K 기준 예상). 끊길 수 있으니 출력은 Drive에 둔다.

## 6. 아직 검증하지 않은 것
- **GPU 코드(1단계 인코딩·M3 학습, 2단계 분할기)는 실제로 돌려 보지 않았다.** 문법 검사와, 모델 없이 도는 부분(데이터 생성·점검·공통 문단 인식)만 샘플로 확인했다 → 셀 3의 `--quick 300`부터 실행할 것.
- 학습·평가 쌍 구성은 설계("오라클 천장 실험과 같은 절차")를 코드에서 직접 확인하지 못해 2.3절처럼 정했다. 팀 오라클 코드와 다르면 맞춰야 한다.
