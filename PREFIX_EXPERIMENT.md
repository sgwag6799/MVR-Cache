# 공통 문단 스트레스 테스트 (`prefix-stress-colab` 브랜치)

> **답변은 한국어로 해 주세요.** `main`(원저자 코드, `8cfaa0b`)에서 딴 브랜치다. `prefix-stress-test`의 실험 코드 전체 + Colab(Python 3.13) 호환 수정 + 로컬 데이터 준비. `gpu-production`(규칙 분할 예외 1~6, 로그 A~F)과 `dialect-cache-experiment`와는 별개다.
> 설계 원문: [PREFIX_PLAN.md](PREFIX_PLAN.md) (노션 "공통 문단 테스트: 조각 가중치가 필요한 조건 찾기"). 이 문서는 그 설계를 코드로 옮긴 방법과, 데이터로 확인해 보니 설계와 충돌한 부분을 적는다.
> **새 세션에서 이어서 할 때는 7절부터 읽는다.**

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

## 7. VSCode + Colab 런타임에서 실행 (`prefix-stress-colab`)

Colab 런타임에 VSCode로 붙어 터미널에서 돌리는 방식. Drive 없이 HF 원본에서 팀 분할을 그대로 다시 만든다. **런타임이 끊기면 `/root` 아래가 모두 사라지므로** 코드는 push, 결과는 따로 백업한다.

### 7.1 런타임마다 1회
```bash
git clone -b prefix-stress-colab https://github.com/sgwag6799/MVR-Cache.git /root/MVR-Cache
cd /root/MVR-Cache && source colab_setup.sh          # 의존성 + PYTHONPATH + HF 경로
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkClassification','train.parquet',repo_type='dataset',local_dir='data/raw')"
python scripts/make_cls_splits.py                    # 팀 분할 재현 + 지문 검사
```
- `colab_setup.sh`: Colab의 torch·numpy·transformers를 고정한 채 필요한 패키지만 설치. hnswlib는 numpy 2로 빌드.
- `scripts/make_cls_splits.py`: `df.sample(frac=1, random_state=42)` → train 3,000 / val 1,000 / test 41,000 (섞인 순서 그대로). `gpu-production` manifest의 지문 `35365cc8e27f` / `a472edba40b2` / `f215310197be`와 일치해야 통과 (HF 리비전 `373719c`).
- 결과: `data/cls_seed42/{train,val}.parquet`, `data/cls_seed42_nodedup/test41k.parquet` (`data/`는 git 제외).

### 7.2 단계 실행 (5절과 같은 명령, 경로만 로컬)
```bash
cd /root/MVR-Cache/mvr-cache
SPL=/root/MVR-Cache/data; OUT=/root/MVR-Cache/data/prefix
python -m benchmarks.prefix.make_prefix_data --train $SPL/cls_seed42/train.parquet --val $SPL/cls_seed42/val.parquet \
  --test $SPL/cls_seed42_nodedup/test41k.parquet --out $OUT/data
python -m benchmarks.prefix.prefix_sanity --data $OUT/data
```
- 오래 걸리는 실행은 `tmux` 또는 `nohup ... > log 2>&1 &`로 (VSCode 연결이 끊겨도 계속 돌게).

### 7.3 이 브랜치에서 main 대비 추가로 고친 것 (Colab 호환)
| 파일 | 수정 |
|---|---|
| `inference_engine/strategies/lang_chain.py` | langchain 1.x: `langchain.schema` → `langchain_core.messages` |
| `embedding_engine/strategies/lang_chain.py` | `HF_ENDPOINT`를 hf-mirror.com으로 강제로 덮어쓰지 않음 (설정값 우선) |
| `splitter/MaxSimEnv.py`, `rl-training-algorithm/MaxSimEnv.py` | torchrl ≥ 0.11의 `Bounded`/`Composite`/`UnboundedContinuous`/`Categorical` 별칭 |

### 7.4 확인된 사실 · 주의
- 현재 런타임 GPU는 **T4** (설계는 L4). 2단계 지연시간은 GPU 이름과 함께 기록.
- `EmbeddingModel.encode()`는 패딩 토큰까지 평균한다 → 같은 문장도 배치 구성에 따라 벡터가 달라짐 (실측 cos 0.958). 한 건씩 인코딩하면 영향 없음. 오프라인 배치 인코딩 코드가 마스크 평균인지 확인할 것.
- 원본 test41k에서 이미 512토큰 초과 1,438행(3.5%) → 4절 1번(`fits_all_levels`) 규칙 근거.
- 같은 프롬프트 6,315개(정확히 일치). 변형 방식 시드가 `(dataset_name, id)` 기준이라 같은 문장이라도 id가 다르면 다른 공통 문단이 붙는다.

## 8. 1단계 결과 (2026-10-25, 커밋 `0287b5e`, Tesla T4)

전체 실행: `fits_all_levels` 행만 (train 2,791 / val 935 / test 38,304), 앵커 6,000 × 최근접 10, M3 20 에폭. 원본 출력은 [`mvr-cache/benchmarks/prefix/stage1_out/`](mvr-cache/benchmarks/prefix/stage1_out/).

### 8.1 통과 기준: **미달 → 2단계 진행하지 않음** (설계 규칙대로 1단계 결과가 최종)

| 단계 | R (L0 M2) | M2 | M3 | M4 | 하락 R−M2 | 회복률 | 통과 |
|---|---|---|---|---|---|---|---|
| L4 | 0.6972 | 0.6537 | 0.6636 | 0.6616 | 0.043 ✅ | 0.23 ❌ | ❌ |
| L8 | 0.6972 | 0.6241 | 0.6501 | 0.6413 | 0.073 ✅ | 0.36 ❌ | ❌ |

### 8.2 전체 지표 (과제 ALL)

`auc_selected` (기본 지표)

| | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| L0 | 0.6970 | 0.6972 | 0.7001 | 0.6972 |
| L2 | 0.6451 | 0.6390 | 0.6545 | 0.6467 |
| L4 | 0.6474 | 0.6537 | 0.6636 | 0.6616 |
| L8 | 0.6234 | 0.6241 | 0.6501 | 0.6413 |
| 고정 L4 | 0.6390 | 0.6374 | 0.6485 | 0.6439 |

`top1_acc`

| | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| L0 | 0.8125 | 0.8120 | 0.8080 | 0.8120 |
| L2 | 0.7933 | 0.7975 | 0.7968 | 0.7972 |
| L4 | 0.7765 | 0.7707 | 0.7812 | 0.7765 |
| L8 | 0.7445 | 0.7443 | 0.7478 | 0.7565 |
| 고정 L4 | 0.7872 | 0.7855 | 0.7855 | 0.7867 |

`tpr_at_fpr1`

| | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| L0 | 0.104 | 0.107 | 0.104 | 0.107 |
| L2 | 0.086 | 0.088 | 0.099 | 0.096 |
| L4 | 0.081 | 0.085 | 0.100 | 0.097 |
| L8 | 0.065 | 0.068 | 0.093 | 0.087 |

- `ceiling10`은 모든 단계·조건에서 0.982~0.983 (검색 천장은 공통 문단 영향 없음).
- `auc_pairs`(6만 쌍 전체)는 L0 0.61 → L2 이상 0.48~0.53 (M3가 항상 가장 높음).

### 8.3 해석 (5절 사전 고정 표 기준)

- **M2 하락은 확인됨**: L0 → L8에서 `auc_selected` −0.073, `top1_acc` −0.068. 공통 문단이 길수록 단일 벡터(M1)와 균등 MVR(M2)이 함께 떨어진다. M1과 M2는 모든 단계에서 거의 같다.
- **M3는 일부만 회복 (L8 36%)하지만 M3 ≥ M4.** 사전 표의 "M3 일부 회복, M3 < M4" 칸과 맞지 않는다. 공통 조각을 정확히 빼는 상한(M4)조차 R에 못 미친다 → 남는 차이는 공통 조각 자체가 아니라 **문맥 임베딩 오염**(원본 조각 벡터가 같은 입력 안의 공통 문단에 attention으로 영향받음)으로 보인다. M3가 M4보다 높은 것은 원본 부분 안의 조각 가중치까지 조정하기 때문 (L0에서도 M3 > M2).
- **검색 천장 하락 없음** → 병목은 후보 검색이 아니라 재순위 점수.
- 과제별: 하락과 M3 이득은 대부분 ecommerce(L8 M2 0.794 → M3 0.823)에서 나온다. amazon·commonsense는 L0부터 `auc_selected`가 0.52~0.53 수준이라 구분력 자체가 낮다.

### 8.4 결과를 읽을 때 주의할 점

- **M3 학습 val AUC가 L2 이상에서 0.45~0.48 (0.5 미만)**이고 선택 에폭이 L4 18, L8 19 (20 에폭 끝 근처)다. 학습이 끝까지 개선 중이었을 수 있어 M3는 과소학습일 가능성이 있다. 설계 고정값(20 에폭)대로 보고하고, 에폭을 늘린 실행은 탐색 결과로만 따로 표시할 것.
- 차이의 크기: 앵커 6,000개 기준 AUC 표준오차는 대략 0.01 수준. L4의 M3−M2(+0.010)는 잡음 범위, L8의 +0.026은 그보다 크다. 부트스트랩 신뢰구간은 아직 계산하지 않았다 (쌍별 점수를 저장하지 않음).
- GPU는 설계(L4)와 다른 T4. 1단계 지표는 GPU와 무관하다 (fp16 수치 차이 정도).

## 9. 탐색 실행 (설계 밖, 2026-10-25, 커밋 `409e364`) — 8절 결과를 대체하지 않음

8.4절의 "M3 과소학습" 의심과 "흔한 어구 가중치 낮추기(라벨 없는 방식)"를 확인하려고 돌린 것. **통과 판정과 최종 결과는 8절(설계 고정값)이다.**

- 바뀐 것: M3 **200 에폭**(약 2,800 스텝, 8절은 20 에폭 ≈ 280 스텝), 조건 추가 **M5 idf**·**M6 centroid** (`--extra-conditions M5 M6`, `gpu-production` RuleSplitter와 같은 식, 통계는 각 단계 train에서), 쌍별 점수 저장(`--save-scores`) → 앵커 단위 부트스트랩 2,000회.
- 원본 출력: [`mvr-cache/benchmarks/prefix/stage1_explore/`](mvr-cache/benchmarks/prefix/stage1_explore/) (`bootstrap_auc_selected.csv`, `scores_*.npz` 포함)

### 9.1 `auc_selected` (과제 ALL)

| | M1 | M2 | M3 (200ep) | M4 | M5 idf | M6 centroid |
|---|---|---|---|---|---|---|
| L0 | 0.6970 | 0.6972 | **0.7149** | 0.6972 | 0.6979 | 0.6963 |
| L2 | 0.6451 | 0.6390 | 0.6527 | 0.6467 | 0.6414 | 0.6389 |
| L4 | 0.6474 | 0.6537 | 0.6632 | 0.6616 | 0.6535 | 0.6536 |
| L8 | 0.6234 | 0.6241 | **0.6558** | 0.6413 | 0.6264 | 0.6239 |
| 고정 L4 | 0.6390 | 0.6374 | 0.6498 | 0.6439 | 0.6415 | 0.6359 |

같은 기준으로 판정하면 L4 회복률 0.22, L8 0.43 → 여전히 미달.

### 9.2 짝지은 차이와 95% 부트스트랩 구간 (`auc_selected`)

| 차이 | L0 | L2 | L4 | L8 |
|---|---|---|---|---|
| M3 − M2 | **+0.018** [+0.010, +0.026] | **+0.014** [+0.005, +0.022] | +0.010 [−0.000, +0.019] | **+0.032** [+0.022, +0.041] |
| M4 − M2 | 0 | **+0.008** [+0.002, +0.013] | **+0.008** [+0.001, +0.015] | **+0.017** [+0.010, +0.024] |
| M5 − M2 (idf) | +0.001 [−0.003, +0.004] | +0.002 [−0.002, +0.007] | −0.000 [−0.005, +0.004] | +0.002 [−0.002, +0.006] |
| M6 − M2 (centroid) | −0.001 [−0.002, +0.000] | −0.000 [−0.002, +0.002] | −0.000 [−0.002, +0.002] | −0.000 [−0.003, +0.002] |
| M3 − M4 | +0.018 | +0.006 [−0.002, +0.014] | +0.002 [−0.006, +0.009] | **+0.014** [+0.006, +0.022] |
| R − M2 | — | **+0.058** [+0.043, +0.072] | **+0.044** [+0.028, +0.058] | **+0.073** [+0.057, +0.089] |

### 9.3 읽은 것

1. **라벨 없는 가중치(M5 idf, M6 centroid)는 효과 없음.** 모든 단계에서 구간이 0을 포함한다. idf는 L8에서 공통 조각이 가중치에서 차지하는 비율을 54% → 41%로 낮추지만(500행 확인) 점수 순위는 거의 그대로다. centroid는 비율 자체를 거의 못 바꾼다(51%).
   - 해석: 공통 조각은 질의와 후보 양쪽에 거의 같은 문장으로 있어서 서로 cos ≈ 1로 짝지어진다 → 모든 후보에 비슷한 크기의 상수를 더하는 셈이라, 줄여도 후보 간 순서는 크게 안 바뀐다. 구분력 손실의 주원인은 공통 조각 자체가 아니라 **원본 조각 벡터의 문맥 오염**이라는 8.3절 해석과 맞는다 (M4도 R의 일부만 회복).
2. **M3는 에폭을 늘리면 L0·L8에서 오른다** (L0 0.700 → 0.715, L8 0.650 → 0.656). 그러나 **L0에서도 +0.018**이 나므로, L8의 +0.032 중 공통 문단 때문에 생긴 몫은 약 +0.014뿐이다. 학습 가중치의 이득 대부분은 공통 문단과 무관한 "원본 조각 재가중"이다.
3. **M3는 200 에폭에서도 수렴하지 않았다.** 선택 에폭이 L0 194, 나머지 모두 199(마지막), train BCE가 계속 감소. 선택 기준인 val AUC(val 안 모든 쌍)는 L2 이상에서 0.45~0.48로 0.5 미만이고 에폭 19 → 199 동안 거의 안 변한다(예: L8 0.4757 → 0.4779) → **선택 기준이 사실상 "마지막 에폭"으로 작동**한다. val 지표(쌍 전체 AUC)와 판정 지표(`auc_selected`)가 다른 것도 문제. 다음에 고칠 것: val 선택 지표를 `auc_selected`로 맞추기, 에폭 대신 수렴(조기 종료) 기준.
4. 1등 정답률(`top1_acc`)은 조건 간 차이가 작다 (L8: M2 0.744, M3 0.748, M4 0.757). AUC 이득이 "1등 후보 선택"보다 "1등 후보의 점수 보정"에서 주로 나온다는 뜻이며, 2단계 히트율로 이어질지는 1단계만으로 알 수 없다.
