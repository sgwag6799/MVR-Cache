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

### 9.4 M3 에폭 선택 옵션 (9.3-3 대응, 커밋 이후 추가)

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--select-metric {auc_pairs, auc_selected}` | `auc_pairs` (원래 규칙) | val에서 에폭을 고르는 지표. `auc_selected`는 판정 지표와 같은 "행마다 1등 후보의 (s, c) AUC" |
| `--patience N` | `0` (끔) | N 에폭 동안 val 최고값이 안 바뀌면 학습 중단 |

- 옵션을 안 주면 8절과 결과가 완전히 같다 (빠른 실행으로 차이 0.0 확인).
- meta의 `m3_training`에 `select_metric`, `epochs_run`, 에폭별 `val_auc_pairs`·`val_auc_selected`가 함께 남는다.
- val은 935행이라 `auc_selected` 한 값의 잡음이 크다 (표준오차 약 0.02). patience를 너무 작게 잡으면 첫 몇 에폭에서 멈출 수 있으니 20 이상 권장.

```bash
python -m benchmarks.prefix.prefix_offline --data $OUT/data --out $OUT/explore_sel \
  --epochs 400 --select-metric auc_selected --patience 30 --extra-conditions M5 M6 --save-scores
```

### 9.5 에폭 선택을 `auc_selected` + 조기 종료로 바꾼 실행 (커밋 `741860a`)

`--epochs 400 --select-metric auc_selected --patience 30 --extra-conditions M5 M6 --save-scores`. 출력: [`stage1_explore_sel/`](mvr-cache/benchmarks/prefix/stage1_explore_sel/).

| 단계 | M2 | M3 20ep (8절) | M3 200ep (9.1) | **M3 auc_selected·patience 30** | 돈 에폭 / 고른 에폭 | M4 |
|---|---|---|---|---|---|---|
| L0 | 0.6972 | 0.7001 | 0.7149 | 0.6997 | 55 / 24 | 0.6972 |
| L2 | 0.6390 | 0.6545 | 0.6527 | 0.6505 | 33 / 2 | 0.6467 |
| L4 | 0.6537 | 0.6636 | 0.6632 | 0.6646 | 58 / 27 | 0.6616 |
| L8 | 0.6241 | 0.6501 | 0.6558 | 0.6511 | 113 / 82 | 0.6413 |
| 고정 L4 | 0.6374 | 0.6485 | 0.6498 | 0.6457 | 86 / 55 | 0.6439 |

- 판정: L4 회복률 0.25, L8 0.37 → 미달 (세 가지 에폭 설정 모두 0.22~0.43).
- **val의 `auc_selected`도 L2 이상에서 0.47~0.50 (0.5 근처)**: val 935행에서는 M3 1등 후보 점수가 정답 여부를 거의 구분하지 못한다. test(앵커 6,000)에서는 0.65 수준이라 val이 작고 잡음이 커서 에폭 선택 신호로 약하다.
- 결론: 에폭 수·선택 기준을 바꿔도 M3는 0.650~0.665 범위. **"과소학습 때문에 미달"이라는 설명은 지지되지 않는다.** Linear(768→1) 조각 가중치로 얻을 수 있는 회복은 대략 이 정도이고, 남는 차이는 문맥 오염(공통 문단이 원본 조각 벡터 자체를 바꿈)일 가능성이 크다 → 다음 확인: 원본 부분만 따로 인코딩한 조각 벡터로 M4 계산.

## 10. 문맥 오염 확인 (설계 밖 탐색, 2026-10-25)

스크립트: [`prefix_contamination.py`](mvr-cache/benchmarks/prefix/prefix_contamination.py). 출력: [`stage1_contamination/`](mvr-cache/benchmarks/prefix/stage1_contamination/). 테스트 앵커 6,000개 (8절과 같은 행), 부트스트랩 2,000회.

M4와 R의 차이를 두 가지로 나눈다. 단계마다 2 × 2.

- **조각 벡터**: ctx = 단계 L 인코딩에서 원본 조각만 (공통 문단을 앞에 두고 인코딩) / iso = 원본만 따로 인코딩 (= 같은 행의 L0 조각. 원본 부분 분할이 단계와 같음을 확인: 정렬 비율 1.000)
- **후보 쌍**: L = 단계 L 문장 벡터로 찾은 이웃 10개 / L0 = L0 문장 벡터로 찾은 이웃 (R이 쓰는 쌍)
- iso × L0 = R. ctx × L ≈ M4 (8절 M4는 후보 쪽 공통 조각이 max 대상에 남아 있어 값이 조금 다르다: L4 0.6616 vs 0.6648)

### 10.1 `auc_selected` (과제 ALL), R과의 차이 [95% 구간]

| 단계 | 조각 cos (ctx vs iso) | 이웃 겹침 | M2 (L 쌍) | ctx × L (≈M4) | **iso × L (후보만 바뀜)** | **ctx × L0 (벡터만 바뀜)** | R |
|---|---|---|---|---|---|---|---|
| L2 | 0.956 | 0.48 | −0.058 [−0.072, −0.045] | −0.051 [−0.064, −0.037] | +0.002 [−0.007, +0.012] | −0.046 [−0.059, −0.034] | 0.6972 |
| L4 | 0.935 | 0.42 | −0.044 [−0.059, −0.029] | −0.032 [−0.046, −0.019] | +0.006 [−0.005, +0.016] | −0.037 [−0.050, −0.023] | 0.6972 |
| L8 | 0.873 | 0.32 | −0.073 [−0.089, −0.057] | −0.050 [−0.065, −0.035] | +0.014 [+0.001, +0.027] | −0.043 [−0.058, −0.028] | 0.6972 |
| 고정 L4 | 0.945 | 0.56 | −0.046 [−0.060, −0.032] | −0.037 [−0.050, −0.024] | +0.005 [−0.002, +0.012] | −0.039 [−0.052, −0.027] | 0.6854 |

- 고정 L4는 두 데이터 모두 512토큰 안에 드는 38,303행 기준이라 R이 다르다.
- 검색 천장@10은 모든 칸 0.982~0.985로 거의 같다.

### 10.2 읽은 것

1. **후보 집합 변화는 원인이 아니다.** 단계 L의 이웃은 L0 이웃과 32~56%만 겹치지만, 그 후보를 원본만 인코딩한 벡터로 점수 매기면 R과 같거나 오히려 높다 (L8 +0.014).
2. **남는 손실은 거의 전부 문맥 오염이다.** 원본 조각을 공통 문단과 함께 인코딩하기만 해도 (후보는 R과 같음) −0.037 ~ −0.046. 공통 문단이 길수록 원본 조각 벡터가 더 바뀐다 (cos 0.956 → 0.873).
3. 따라서 공통 조각 가중치를 0으로 두는 M4도 R까지 회복할 수 없고, 같은 조각 벡터 위에서 가중치만 학습하는 M3의 회복률이 0.5에 못 미치는 것은 구조적 한계다. 사전 고정 해석표 기준 "M2 하락, M3 일부 회복, M3 < M4"를 넘어, **상한 M4 자체가 문맥 오염 때문에 R보다 낮다**는 것이 확인됐다.
4. 시사점: 이 조건에서 이득을 보려면 재순위 가중치가 아니라 **인코딩 단계에서 공통 부분을 분리**해야 한다 (예: 조각별 따로 인코딩, 공통 부분을 제거하고 인코딩, 어텐션 마스크로 공통 부분을 가림). 이는 지연시간·구현 비용이 다른 별도 설계다.

```bash
python -m benchmarks.prefix.prefix_contamination --data $OUT/data --out $OUT/contamination
```

## 11. 공통 문장 풀 × 라벨 모델 (설계 밖 탐색, 2026-10-27)

스크립트: [`scripts/run_prefix_labels_tasks.sh`](scripts/run_prefix_labels_tasks.sh). 출력: [`results_labels_tasks/`](mvr-cache/benchmarks/prefix/results_labels_tasks/). 풀: 공용 `prefix_pool.json` / 하위 과제별 `prefix_pool_task.json`. 라벨: `response_llama_3_8b` / `response_llama_3_70b`. 조합마다 1단계 오프라인(설계 기본값 + M5·M6, `--save-scores`) + 문맥 오염 2×2.

| 풀 · 라벨 | R | L4 하락 / 회복률 | L8 하락 / 회복률 | 통과 | L8 ctx × L0 (벡터만 바뀜) | L8 iso × L (후보만 바뀜) |
|---|---|---|---|---|---|---|
| 공용 · 8B | 0.697 | 0.043 / 0.23 | 0.073 / 0.36 | ❌ | −0.043 | +0.014 |
| 공용 · 70B | 0.729 | 0.069 / 0.33 | 0.098 / 0.30 | ❌ | −0.056 | +0.012 |
| 과제별 · 8B | 0.701 | 0.092 / 0.29 | 0.120 / 0.32 | ❌ | −0.060 | +0.033 |
| 과제별 · 70B | 0.717 | 0.104 / 0.30 | 0.135 / 0.34 | ❌ | −0.062 | +0.045 |

- 4개 조합 모두 미달, 모두 M3 > M4, 손실의 주원인은 문맥 오염 (10절과 같은 결론).
- 과제별 풀은 하락이 더 크다. 후보 집합은 오히려 좋아지고(iso × L > R), 원본 조각 벡터는 더 많이 바뀐다.

## 12. 실험 1: 과제 안에서 잰 통과 판정 (설계 밖, 2026-10-28, 브랜치 `prefix-exp1-within-task-gate`)

스크립트: [`prefix_within_task_gate.py`](mvr-cache/benchmarks/prefix/prefix_within_task_gate.py) (GPU 불필요, `--save-scores` 점수 사용). 출력: [`exp1_within_task/`](mvr-cache/benchmarks/prefix/exp1_within_task/).

**왜:** 8·11절의 `auc_selected`는 세 과제의 앵커를 한데 섞어 계산한다. 과제마다 점수대와 정답률이 크게 다르다 (L0 M2 1등 정답률: ecommerce 0.95, amazon 0.76, commonsense 0.53). 그래서 섞은 AUC는 "ecommerce를 다른 과제보다 높게 매기는" 점수에도 보상을 준다. 공통 문단이 과제 사이 점수 차이를 줄이면, 과제 안에서는 아무것도 안 바뀌어도 구분력이 떨어진 것처럼 보인다.

**방법:** 과제마다 `auc_selected`를 따로 계산하고, 세 과제 값의 평균(**macro**, 과제마다 같은 비중)으로 같은 기준(하락 ≥ 0.03, 회복률 ≥ 0.5)을 적용한다. ecommerce 단독도 함께 보고한다. 구간은 과제별로 층화한 앵커 부트스트랩 1,000회.
- 공용 · 8B는 8절 실행이 쌍별 점수를 저장하지 않아 같은 설정으로 다시 돌렸다 (`exp1_within_task/scores_prefix_8b/`, L4 GPU). 8절과 차이 ≤ 0.0005 (fp16 수준).
- 섞은 값(pooled)은 `prefix_gate.json`과 일치함을 확인했다.
- **주의:** macro 기준은 11절의 과제별 수치를 본 뒤 정했다. 탐색 결과이며 8절 판정을 대체하지 않는다.

### 12.1 하락: 섞은 값 vs 과제 안

| 풀 · 라벨 | L8 하락 pooled | L8 하락 macro [95%] | L8 하락 ecommerce [95%] |
|---|---|---|---|
| 공용 · 8B | 0.073 | 0.023 [+0.001, +0.045] | 0.028 [+0.002, +0.054] |
| 공용 · 70B | 0.098 | 0.046 [+0.022, +0.069] | 0.014 [−0.015, +0.043] |
| 과제별 · 8B | 0.120 | 0.034 [+0.009, +0.059] | 0.041 [+0.018, +0.064] |
| 과제별 · 70B | 0.135 | 0.034 [+0.006, +0.058] | 0.040 [+0.008, +0.072] |

→ 과제 안의 하락은 섞은 값의 **1/4 ~ 1/2**. 8·11절 하락의 절반 이상은 과제 사이 점수 차이가 줄어든 효과다.

### 12.2 판정

| 풀 · 라벨 | 단계 | macro 하락 / 회복률 [95%] | macro M3−M2 [95%] | ecommerce 하락 / 회복률 [95%] | ecommerce M3−M2 [95%] | 통과 (macro / ecom) |
|---|---|---|---|---|---|---|
| 공용 · 8B | L4 | 0.015 / 0.35 | +0.005 [−0.009, +0.019] | 0.012 / −0.15 | −0.002 [−0.013, +0.010] | ❌ / ❌ |
| 공용 · 8B | L8 | 0.023 / 0.53 [−0.01, 2.87] | +0.012 [−0.000, +0.026] | 0.028 / 1.06 [0.48, 4.27] | +0.029 [+0.014, +0.046] | ❌ / ❌ (하락 0.028) |
| 공용 · 70B | L4 | 0.040 / **0.57** [0.26, 1.07] | +0.023 [+0.009, +0.037] | 0.010 / 0.45 | +0.004 [−0.011, +0.018] | ✅ / ❌ |
| 공용 · 70B | L8 | 0.046 / 0.29 [−0.02, 0.67] | +0.014 [−0.001, +0.026] | 0.014 / 1.62 | +0.023 [+0.007, +0.038] | ❌ / ❌ |
| 과제별 · 8B | L4 | 0.042 / 0.13 | +0.005 [−0.010, +0.019] | 0.026 / 0.48 | +0.013 [−0.001, +0.024] | ❌ / ❌ |
| 과제별 · 8B | L8 | 0.034 / 0.13 [−0.48, 0.63] | +0.004 [−0.010, +0.018] | 0.041 / **0.60** [0.26, 1.23] | +0.025 [+0.009, +0.041] | ❌ / ✅ |
| 과제별 · 70B | L4 | 0.044 / 0.41 | +0.018 [+0.004, +0.032] | 0.026 / 0.76 | +0.020 [+0.002, +0.037] | ❌ / ❌ |
| 과제별 · 70B | L8 | 0.034 / **0.72** [0.33, 2.56] | +0.024 [+0.011, +0.038] | 0.040 / **0.97** [0.54, 2.93] | +0.039 [+0.024, +0.056] | ✅ / ✅ |

### 12.3 읽은 것

1. **섞어서 잰 하락은 과장되어 있다.** 과제 안에서 보면 하락은 0.01~0.05이고, 몇몇 칸은 기준(0.03)에 못 미친다.
2. **과제 안에서 보면 회복률은 0.5 근처이거나 그 이상인 칸이 있다.** macro 기준 통과 2칸(공용·70B L4, 과제별·70B L8), ecommerce 기준 통과 2칸(과제별 8B·70B L8). 섞은 값으로는 모두 미달이었다.
3. **그러나 결론을 뒤집기엔 잡음이 크다.** 하락이 작아서 회복률의 분모가 작고, 회복률 구간이 대부분 0.5를 양쪽으로 넘는다. amazon·commonsense는 L0부터 0.52~0.60으로 신호가 거의 없어 macro 값을 흔든다.
4. 가장 안정적인 것은 **ecommerce의 M3 − M2** (L8 모든 조합에서 +0.023~+0.039, 구간이 0 위)와 **과제별·70B L8** (macro·ecommerce 모두 통과, M3−M2 구간이 0 위).
5. 다음: 유사도가 정답을 결정하는 단일 과제 데이터(실험 2, SemBenchmarkLmArena `ID_Set`)에서 이 기준을 **결과 보기 전에 고정**하고 다시 잰다.

```bash
python -m benchmarks.prefix.prefix_within_task_gate --out benchmarks/prefix/exp1_within_task \
  "공용·8B=benchmarks/prefix/exp1_within_task/scores_prefix_8b" \
  "공용·70B=benchmarks/prefix/results_labels_tasks/prefix/offline_70b" \
  "과제별·8B=benchmarks/prefix/results_labels_tasks/prefix_task/offline_8b" \
  "과제별·70B=benchmarks/prefix/results_labels_tasks/prefix_task/offline_70b"
```

## 13. 실험 2: 유사도가 정답을 결정하는 데이터 (SemBenchmarkLmArena, 브랜치 `prefix-exp2-lmarena`)

**왜:** 12절에서 amazon(답 2개)·commonsense(답이 질문 내용에 달림)는 L0부터 `auc_selected` 0.52~0.60으로 신호가 거의 없어, 하락·회복률이 잡음에 묻혔다. 패러프레이즈 묶음(`ID_Set`)이 있는 단일 과제 데이터로 같은 1단계 진단을 다시 한다.

### 13.1 사전 고정 (결과 보기 전, 이 커밋에서 고정)

| 항목 | 내용 |
|---|---|
| 데이터 | `vCache/SemBenchmarkLmArena` 63,796행 · `ID_Set` 3,500개 (중앙값 20행/묶음) |
| 분할 | [`scripts/make_lmarena_splits.py`](scripts/make_lmarena_splits.py): 묶음 단위로 `random.Random(42)` 순서대로 train ≥ 3,000 / val ≥ 1,000 / test ≥ 41,000행. 남는 묶음은 사용 안 함. 같은 묶음은 한 분할에만 있음 |
| 정답 c | 같은 `ID_Set` (`--resp-col id_set_label`) |
| 공통 문단 | 공용 `prefix_pool.json` (일반 어시스턴트 지시문이라 채팅 프롬프트에 자연스러움), 변형 L0·L2·L4·L8 + 고정 L4 |
| 나머지 | 8절과 같은 설정: 분할 규칙, `fits_all_levels`, 앵커 6,000 × 최근접 10, M3 20 에폭, + M5·M6, `--save-scores`, 문맥 오염 2×2 |
| 판정 | 8절과 같은 기준 (`auc_selected`, L ≥ 4에서 하락 ≥ 0.03 그리고 회복률 ≥ 0.5). 과제가 하나라 섞임 문제 없음 |
| 보조 | `auc_pairs`, `top1_acc`, `ceiling10`, 부트스트랩 구간. 1등 후보 정답률이 매우 높으면(오답 1등 < 5%) `auc_selected`의 오답 표본이 적으므로 `auc_pairs`를 함께 해석 |

### 13.2 결과 (사전 고정 설정 그대로, 2026-10-28, NVIDIA L4)

출력: [`exp2_lmarena/`](mvr-cache/benchmarks/prefix/exp2_lmarena/). `fits_all_levels` 행: train 3,007 / val 1,000 / test 40,964. LmArena 프롬프트는 짧다 (test 평균 47토큰, 조각 4.3개 → L8 176토큰, 12.3개).

| 단계 | ceiling10 | top1_acc M2 / M3 / M4 | `auc_selected` M2 / M3 / M4 | `auc_pairs` M2 / M3 / M4 |
|---|---|---|---|---|
| L0 (R) | 0.995 | 0.953 / 0.953 / 0.953 | 0.472 / 0.470 / 0.472 | 0.554 / 0.554 / 0.554 |
| L2 | 0.993 | 0.943 / 0.945 / 0.946 | 0.441 / 0.442 / 0.443 | 0.522 / 0.546 / 0.552 |
| L4 | 0.991 | 0.935 / 0.941 / 0.941 | 0.450 / 0.475 / 0.472 | 0.554 / 0.607 / 0.609 |
| L8 | 0.986 | 0.914 / 0.926 / 0.929 | 0.496 / 0.522 / 0.513 | 0.568 / 0.650 / 0.656 |
| 고정 L4 | 0.994 | 0.954 / 0.954 / 0.954 | 0.482 / 0.481 / 0.484 | 0.576 / 0.576 / 0.574 |

- **판정: ❌** (L4 하락 0.022 < 0.03, L8은 M2가 R보다 높음 → 하락 조건 불충족). 사전 표로는 "M2가 떨어지지 않음 = 음성 결과" 칸.
- **그러나 13.1에서 미리 적은 경고에 해당한다.** L0 1등 오답이 4.7%(287개)뿐이고, `auc_selected`가 0.5 미만이다 (1등 오답이 1등 정답보다 점수가 높음).
- **원인: 데이터 라벨 잡음.** 글자까지 같은 프롬프트 186개가 서로 다른 `ID_Set`에 들어 있다 (test 661행). 이 쌍은 점수 1.000인데 "오답"으로 세어져 1등 오답 목록 맨 위를 차지한다. 예: `"How many sisters does each of Alice's brothers have?"` 가 묶음 1281과 1264에 각각 있음. 그래서 `tpr_at_fpr1`도 거의 0이다.
- 라벨 잡음과 무관하게 볼 수 있는 것: **top1_acc는 L0 0.953 → L8 0.914로 떨어지고**, M3·M4가 그중 약 1/3을 되찾는다 (0.926 · 0.929). 고정 L4는 하락이 없다.
- 문맥 오염 2×2 (L8, R과의 차이): ctx × L0 −0.015 [−0.055, +0.024], iso × L +0.109 [+0.076, +0.141]. 구간이 넓고 기준값(R)이 잡음에 오염돼 있어 해석을 보류한다.

### 13.3 민감도 분석 (결과를 본 뒤 추가, 탐색)

13.2의 라벨 잡음을 고친 정답으로 같은 실행을 한 번 더 한다. **이 분석은 결과를 본 뒤 정했다.**
- 정답 c: 글자까지 같은 프롬프트를 공유하는 `ID_Set`들을 하나로 합친 묶음 (union-find, 원본 63,796행 전체 기준). 열 이름 `id_set_merged`
- 분할 · 공통 문단 · 나머지 설정은 13.1과 같다 (분할은 다시 만들지 않음)

### 13.4 민감도 분석 결과 (합친 정답, 탐색)

출력: [`exp2_lmarena/offline_merged/`](mvr-cache/benchmarks/prefix/exp2_lmarena/offline_merged/), [`contamination_merged/`](mvr-cache/benchmarks/prefix/exp2_lmarena/contamination_merged/). `ID_Set` 3,500 → 합친 묶음 3,303.

| 단계 | top1_acc M2 / M3 / M4 | `auc_selected` M2 / M3 / M4 | `auc_pairs` M2 / M3 / M4 | `tpr_at_fpr1` M2 / M3 |
|---|---|---|---|---|
| L0 (R) | 0.992 | 0.734 | 0.826 | 0.162 |
| L2 | 0.985 / 0.986 / 0.987 | 0.613 / 0.616 / 0.621 | 0.722 / 0.751 / 0.756 | 0.033 / 0.040 |
| L4 | 0.977 / 0.983 / 0.984 | 0.613 / 0.659 / 0.663 | 0.721 / 0.789 / 0.791 | 0.030 / 0.040 |
| L8 | 0.950 / 0.965 / 0.968 | 0.616 / **0.694** / 0.691 | 0.667 / **0.769** / 0.776 | 0.039 / 0.053 |
| 고정 L4 | 0.991 / 0.991 / 0.991 | 0.805 / 0.793 / 0.797 | 0.847 / 0.843 / 0.846 | 0.203 / 0.196 |

회복률과 95% 구간 (앵커 부트스트랩 500회)

| 단계 | 지표 | 하락 R−M2 | M3 회복률 | M4 회복률 | M3 − M4 |
|---|---|---|---|---|---|
| L4 | `auc_selected` | 0.120 [0.062, 0.182] | 0.38 [0.11, 0.89] | 0.42 [0.13, 0.89] | −0.004 [−0.022, +0.012] |
| L4 | `auc_pairs` | 0.104 [0.086, 0.125] | **0.65** [0.55, 0.79] | 0.67 [0.56, 0.82] | −0.002 |
| L8 | `auc_selected` | 0.117 [0.054, 0.182] | **0.66** [0.37, 1.37] | 0.63 [0.33, 1.27] | +0.003 [−0.013, +0.022] |
| L8 | `auc_pairs` | 0.159 [0.136, 0.185] | **0.65** [0.56, 0.75] | 0.69 [0.60, 0.81] | −0.007 |

- **판정 (기준 지표 `auc_selected`): L8 통과** (하락 0.117, 회복률 0.66). L4는 회복률 0.38로 미달.
- 단 1등 오답이 L0 49개, L8 299개뿐이라 `auc_selected` 구간이 넓다. 6만 쌍 전체의 **`auc_pairs`로 보면 L4·L8 모두 회복률 0.65이고 구간 [0.55, 0.79]가 0.5 위**에 있다.
- **M3 ≈ M4** (차이 −0.007 ~ +0.003): 사전 해석표의 "M2 하락, M3 회복, M3 ≈ M4 → 학습 가중치가 공통 조각을 스스로 낮춤, 조건부 양성" 칸. Classification(8·11·12절)에서는 나오지 않던 패턴이다.
- 문맥 오염 (L8, R과의 차이): ctx × L0 −0.123 [−0.205, −0.046], iso × L +0.126 [+0.081, +0.174]. 오염 손실은 여전히 크다. 그래도 가중치가 효과를 보는 이유는, 프롬프트가 짧아 (원본 조각 평균 4.3개) L8에서 공통 조각이 조각 12개 중 8개를 차지하므로 공통 조각을 낮추는 것만으로 얻을 몫이 크기 때문으로 보인다 (해석).
- 고정 L4는 하락이 없다 (M2 0.805 > R). 모든 프롬프트에 같은 문장이면 점수에 상수가 더해질 뿐이다.
- 13.2의 `auc_selected` 0.47은 라벨 잡음 때문이었음이 확인된다 (합친 뒤 0.73).

### 13.5 정리

- 사전 고정 실행(13.2)은 라벨 잡음 때문에 판정할 수 없었다. 잡음을 고친 탐색 실행(13.4)에서는 **L8 통과, M3 ≈ M4, `auc_pairs` 회복률 0.65 [0.55, 0.79]**.
- 즉 "공통 문단 비중이 큰(짧은 프롬프트) 환경에서는 학습 가중치가 필요하다"는 조건부 양성 결과의 근거가 처음 나왔다. 다만 라벨 정의를 결과 후에 고쳤으므로 확정하려면 **합친 정답을 사전 고정한 2단계(41K 스트림) 또는 다른 데이터에서의 재현**이 필요하다.
- 다음: 실험 3 (인코딩 단계에서 공통 부분 분리), Classification과 LmArena 두 데이터에서.

```bash
python scripts/make_lmarena_splits.py --raw data/raw_lmarena/train.parquet --out data/lmarena
python -m benchmarks.prefix.make_prefix_data --train ../data/lmarena/train.parquet --val ../data/lmarena/val.parquet \
  --test ../data/lmarena/test.parquet --out ../data/lmarena/prefix/data            # (mvr-cache 에서)
python scripts/lmarena_merge_duplicate_sets.py --data data/lmarena/prefix/data     # 13.3 민감도용
python -m benchmarks.prefix.prefix_offline --data ../data/lmarena/prefix/data --out ../data/lmarena/prefix/offline_merged \
  --resp-col id_set_merged --extra-conditions M5 M6 --save-scores
python -m benchmarks.prefix.prefix_contamination --data ../data/lmarena/prefix/data --out ../data/lmarena/prefix/contamination_merged \
  --resp-col id_set_merged
```

## 14. 실험 3: 인코딩 단계에서 공통 문단 분리 (설계 밖, 2026-10-28, 브랜치 `prefix-exp3-encode-separation`)

**왜:** 10·11·13절에서 손실의 주원인은 문맥 오염이었다 (공통 문단을 앞에 두고 인코딩하면 원본 조각 벡터가 바뀜). 가중치는 같은 벡터 위에서만 움직이므로, 인코딩 자체를 바꿔 본다.

### 14.1 설계 (실행 전 고정)

스크립트: [`prefix_encode_sep.py`](mvr-cache/benchmarks/prefix/prefix_encode_sep.py). 단계 L의 프롬프트는 그대로 두고 인코딩 방식만 바꾼다. 점수는 M2(균등) · M4(공통 조각 0), 후보는 조건마다 자기 문장 벡터로 다시 검색, R = L0의 M2.

| 조건 | 공통 문단 경계 | 인코딩 |
|---|---|---|
| `full` | 정답 경계 | 지금 방식 (= 1단계 M2 · M4) |
| `mask_oracle` | 정답 경계 (`prefix_chars`) | 한 번의 forward, 블록 어텐션 마스크: `[CLS]` · 원본 · `[SEP]`은 공통 토큰을 못 봄 (공통 토큰은 다 봄). 위치 번호는 그대로. 문장 벡터는 `[CLS]` + 원본 + `[SEP]` 평균 |
| `mask_freq` | **라벨 · 풀 파일 없이**: 단계의 train 프롬프트 1% 이상에 나오는 문장이 앞에서 이어지는 구간 (`. ! ?` + 공백으로 문장 분리) | 위와 같은 마스크 |
| `strip_freq` | 위와 같은 빈도 경계 | 경계 뒤 텍스트만 따로 인코딩 (`strip_oracle`은 L0와 정확히 같으므로 생략) |

- 데이터: Classification 공용 풀 · 8B 라벨 (8절과 같은 데이터), LmArena · 합친 정답 (13.4)
- 지표: `auc_selected`, `auc_pairs`, `top1_acc`, `ceiling10`, R과의 차이 부트스트랩 1,000회. 빈도 경계가 정답 경계와 같은 비율
- 마스크 동작 확인: 마스크를 켜면 공통 문장 표현만 바꿨을 때 원본 토큰 출력 cos 0.9995 (토큰 수 1개 차이로 위치가 밀린 몫), 끄면 0.954
- 고정 방식은 행 집합이 L0와 달라 이번에는 제외

### 14.2 결과 (2026-10-28, NVIDIA L4)

출력: [`exp3_encode_sep/`](mvr-cache/benchmarks/prefix/exp3_encode_sep/). 값은 `auc_pairs`이고 괄호 안은 R과의 차이 [95%]. `auc_selected`도 방향이 같다 (json 참고).

**LmArena (합친 정답, R = 0.826)**

| 단계 | M2 full | M4 full | M2 mask | **M4 mask** | **strip_freq** |
|---|---|---|---|---|---|
| L2 | 0.722 (−0.103) | 0.756 (−0.070) | 0.790 (−0.035) | 0.823 (−0.002 [−0.010, +0.006]) | 0.826 (0) |
| L4 | 0.721 (−0.104) | 0.791 (−0.035) | 0.780 (−0.046) | 0.823 (−0.002 [−0.009, +0.005]) | 0.826 (0) |
| L8 | 0.667 (−0.159) | 0.776 (−0.049) | 0.795 (−0.030) | **0.842 (+0.017 [+0.006, +0.027])** | 0.826 (0) |

- 빈도 경계가 정답 경계와 **100% 일치** → `mask_freq` = `mask_oracle`, `strip_freq` = R (원본만 남으므로 L0와 같음).
- **M4 + 마스크는 모든 단계에서 R과 같거나 높다.** 가중치만으로는 못 넘던 문맥 오염 손실이 마스크로 사라진다. 1등 정답률도 0.991~0.992로 R과 같음.
- 마스크만 쓰고 공통 조각을 남기면(M2 mask) 아직 −0.03~−0.05: 공통 조각끼리 cos ≈ 1로 짝지어지는 몫은 가중치(M4)로 빼야 한다. **마스크 + 가중치를 같이 써야** 회복된다.

**Classification (공용 풀 · 8B, R = 0.613, `auc_selected` R = 0.697)**

| 단계 | M2 full | M4 full | M4 mask_oracle | M4 mask_freq | strip_freq |
|---|---|---|---|---|---|
| L2 | 0.508 (−0.105) | 0.517 (−0.096) | 0.546 (−0.068) | 0.667 (+0.054) | 0.658 (+0.045) |
| L4 | 0.505 (−0.108) | 0.519 (−0.094) | 0.535 (−0.078) | 0.666 (+0.053) | 0.658 (+0.045) |
| L8 | 0.483 (−0.130) | 0.497 (−0.116) | 0.570 (−0.043) | 0.656 (+0.042) | 0.658 (+0.045) |

- **빈도 경계는 모든 행에서 공통 문단보다 더 길다 (over 100%).** 원본 프롬프트의 첫 문장이 과제마다 같은 질문문이라 함께 떼어지기 때문이다: ecommerce "Which category does the text belong to?", amazon "Is the review friendly?", commonsense "What is the main subject of the following question?" (평균 34자).
- 즉 **Classification 데이터는 원래부터 과제별 공통 문단을 갖고 있다.** L0에 같은 처리를 해도 `strip_freq` 값(0.658, `auc_selected` 0.716)이 나오며 단계와 무관하게 같다 → 공정한 기준은 "L0 + 같은 처리"이고, 그 기준으로는 **모든 단계에서 100% 회복**이다. R보다 높은 몫(+0.045)은 원래 데이터의 질문문을 뗀 효과다.
- `mask_oracle`(추가한 공통 문단만 가림)은 일부만 회복 (L8 −0.043, L4 −0.078). LmArena와 달리 R에 못 미친다. 원본이 길어(평균 427자) 공통 문단 길이만큼 밀린 위치 번호의 영향이 클 수 있다 (추정, 미확인).

### 14.3 읽은 것

1. **문맥 오염은 인코딩에서 분리하면 사라진다.** LmArena에서 마스크 + M4, 두 데이터에서 잘라내기가 R 수준까지 (또는 그 이상) 회복한다. 재순위 가중치의 회복률 0.2~0.65와 비교된다.
2. **경계는 라벨 없이 찾을 수 있다.** 같은 문장이 프롬프트 1% 이상에 반복되는지만 보면 충분했다 (단, 이 실험의 공통 문장은 칸마다 변형 5개라 각 문장이 약 20%씩 반복돼 찾기 쉬운 조건이다. 변형이 많거나 표현이 조금씩 다른 실제 템플릿에서는 더 어렵다).
3. **부수 발견:** Classification 원본에도 과제별 공통 질문문이 있고, 이를 떼면 L0에서도 `auc_selected` 0.697 → 0.716, `auc_pairs` 0.613 → 0.658로 오른다. MVR 본 실험(공통 문단을 붙이지 않은 상태)에도 해당하는 개선이다.
4. 비용: 마스크는 forward 1번 그대로(마스크만 바꿈), 잘라내기는 오히려 입력이 짧아진다. 둘 다 경계 탐지(문장 빈도표 조회)가 추가된다.

```bash
python -m benchmarks.prefix.prefix_encode_sep --data ../data/lmarena/prefix/data --out ../data/exp3/lmarena --resp-col id_set_merged
python -m benchmarks.prefix.prefix_encode_sep --data ../data/prefix/data --out ../data/exp3/cls_prefix_8b --resp-col response_llama_3_8b
```

## 15. 실험 4: 가중치 식 바꾸기 (설계 밖, 2026-10-28, 브랜치 `prefix-exp4-weighting`)

**왜:** 지금 M3 식(`w = softmax(Linear(조각))`, `점수 = ½Σ w_q·max cos + ½Σ w_c·max cos`)에는 두 가지 빈틈이 있다. ① 가중치가 합산에만 곱해져, 상대 프롬프트의 공통 조각이 max 짝으로 뽑히는 것은 못 막는다. ② 모든 쌍에 BCE를 걸지만 판정은 앵커별 1등 후보로 한다.

### 15.1 설계 (실행 전 고정)

`prefix_offline.py`에 옵션 2개 추가 (기본값이면 8절과 같음):

| 옵션 | 내용 |
|---|---|
| `--maxsim-gate` | max 전에 상대 조각의 cos(0~1로 변환)에 `w / max(w)`를 곱함 → 가중치 낮은 조각은 최대 짝이 될 수 없음. 균등 가중치(M2)는 값이 그대로 (확인함). M3 · M4 · M5 · M6 모두 적용 |
| `--loss listwise` | M3 학습: train 행마다 후보 10개에 softmax(`a·s`), 정답 후보 확률 합의 −log. 정답 · 오답이 모두 있는 행만 사용 |

- 비교: 기본(bce, 게이트 없음) / 게이트 / listwise / 게이트 + listwise
- 데이터: Classification 공용 풀 · 8B (기본 = 12절 재실행 `exp1_within_task/scores_prefix_8b`), LmArena 합친 정답 (기본 = 13.4)
- 단계 L0 · L4 · L8 (통과 판정에 필요한 단계만), 나머지 설정은 8절과 같음, `--save-scores`
- 판정: 8절 기준 (`auc_selected`) + `auc_pairs` 회복률, M3 − M4

### 15.2 결과 (2026-10-28, NVIDIA L4)

출력: [`exp4_weighting/`](mvr-cache/benchmarks/prefix/exp4_weighting/). M3 회복률 [95%, 앵커 부트스트랩 300회]. 기본 = 15.1의 기본 실행.

| 데이터 | 설정 | L4 회복률 `auc_selected` | L4 회복률 `auc_pairs` | L8 회복률 `auc_selected` | L8 회복률 `auc_pairs` | L8 M3 − M4 (`auc_pairs`) |
|---|---|---|---|---|---|---|
| LmArena | **기본** | 0.38 [0.10, 0.89] | **0.65** [0.54, 0.80] | **0.66** [0.36, 1.39] | **0.65** [0.56, 0.75] | −0.007 |
| LmArena | 게이트 | −0.23 | 0.15 | −0.24 | 0.08 | −0.121 |
| LmArena | listwise | 0.24 | 0.41 | 0.40 | 0.42 | −0.044 |
| LmArena | 게이트 + listwise | −0.25 | 0.20 | −0.38 | 0.11 | −0.116 |
| Classification | **기본** | 0.23 [0.03, 0.44] | 0.23 | 0.36 [0.24, 0.48] | 0.15 | +0.006 |
| Classification | 게이트 | −0.01 | 0.13 | −0.19 | 0.05 | −0.014 |
| Classification | listwise | 0.13 | 0.14 | 0.25 | 0.10 | −0.001 |
| Classification | 게이트 + listwise | 0.05 | 0.13 | −0.07 | 0.05 | −0.014 |

- **두 변경 모두 M3를 낫게 하지 못했다.** 기본(BCE, 게이트 없음)이 두 데이터 · 두 단계 모두 가장 높다.
- **게이트는 M3를 크게 떨어뜨린다** (LmArena L8 `auc_pairs` 0.769 → 0.679, L0에서도 0.734 → 0.702). `w / max(w)`로 max를 가리면, 학습 초기에 가중치가 조금만 낮아진 조각도 짝 후보에서 빠져 학습이 엉뚱한 쪽으로 간다 (추정). 반면 **정답 경계를 아는 M4에는 게이트가 조금 도움이 된다** (LmArena L4·L8 `auc_pairs` 0.791 · 0.776 → 0.801 · 0.800, 10절의 "후보 쪽 공통 조각까지 빼기"와 같은 효과).
- **listwise는 BCE보다 낮다.** 정답 · 오답이 모두 있는 train 행만 쓰므로 학습 행이 크게 줄어든다: LmArena 172~453행(전체 3,007), Classification 1,666~2,132행(2,791). LmArena에서 특히 불리하다.
- 결론: 가중치 식에서 얻을 여지는 작다. 14절(인코딩 분리)이 훨씬 효과가 크다.

## 16. 실험 5: 문장 단위 분할 (설계 밖, 2026-10-28, 브랜치 `prefix-exp5-sentence-split`)

**왜:** 지금 분할은 `. , ! ? ; :` 모두에서 잘라 2~5토큰짜리 짧은 조각이 많다. 짧은 조각은 평균 낼 토큰이 적어 앞 문맥(공통 문단)에 더 휩쓸릴 수 있다 (가설).

### 16.1 설계 (실행 전 고정)

- `prefix_split.cut_points(punct=...)`, `prefix_offline.py` · `prefix_contamination.py`에 `--cut-punct` 추가 (기본값이면 기존과 같음). 이번 실행은 `--cut-punct '.!?'`
- 공통 문장은 끝 마침표 하나뿐이라 여전히 문장 하나 = 조각 하나. 원본 부분만 조각이 길어진다. 상한 32 · `cap_scope original`은 그대로
- 데이터: Classification 공용 풀 · 8B, LmArena 합친 정답. 오프라인은 L0 · L4 · L8 (`--save-scores`), 문맥 오염은 L4 · L8
- 비교 대상: 같은 데이터의 기존 분할 결과 (`exp1_within_task/scores_prefix_8b`, `exp2_lmarena/offline_merged`, `contamination_merged`, 10절)
- 볼 것: R 자체의 변화, M2 하락, M3 회복률, 조각 cos (ctx vs iso), ctx × L0

### 16.2 결과 (2026-10-28, NVIDIA L4)

출력: [`exp5_sentence_split/`](mvr-cache/benchmarks/prefix/exp5_sentence_split/). 기존 = 6개 구두점 분할. 회복률 [95%, 부트스트랩 300회].

| 데이터 | 분할 | R `auc_selected` | L8 M2 → M3 | L8 회복률 `auc_selected` | L8 회복률 `auc_pairs` | L8 조각 cos (ctx vs iso) | L8 ctx × L0 |
|---|---|---|---|---|---|---|---|
| LmArena | 기존 | 0.734 | 0.616 → 0.694 | 0.66 [0.36, 1.39] | 0.65 [0.56, 0.75] | 0.844 | −0.123 |
| LmArena | 문장 | 0.737 | 0.604 → 0.669 | 0.49 [0.26, 0.94] | 0.63 [0.55, 0.73] | 0.846 | −0.112 |
| Classification | 기존 | 0.697 | 0.624 → 0.650 | 0.36 [0.24, 0.48] | 0.15 [0.13, 0.17] | 0.873 | −0.043 |
| Classification | 문장 | 0.699 | 0.624 → 0.652 | 0.37 [0.25, 0.49] | 0.15 [0.13, 0.17] | 0.873 | −0.044 |

- **차이 없음.** R, 하락, 회복률, 조각 cos, 문맥 오염 손실(ctx × L0)이 모두 구간 안에서 같다. L4도 같다 (Classification 회복률 0.23 → 0.34, 구간 겹침).
- 가설("짧은 조각이 문맥에 더 휩쓸린다")은 지지되지 않는다. 조각 벡터의 변화(cos)는 조각 길이가 아니라 공통 문단 길이로 정해진다 (L4 0.91~0.94, L8 0.85~0.87, 분할과 무관).
- 판정은 두 분할 모두 같다: LmArena L8 `auc_pairs` 기준으로는 회복, Classification은 미달.

## 17. 실험 6: 공통 문단 위치 · 긴 입력 인코더 (설계 밖, 2026-10-28, 브랜치 `prefix-exp6-position-encoder`)

### 17.1 공통 문단을 원본 뒤에 붙이기 (설계, 실행 전 고정)

**왜:** 실제 프롬프트에서 공통 지시문은 앞(시스템 프롬프트)에도 뒤("답은 한 단어로" 같은 형식 지시)에도 붙는다. 앞에 붙였을 때의 결과(문맥 오염이 주원인)가 위치에 따라 달라지는지 본다.

- `make_prefix_data.py --position suffix`: `원본 + " " + 공통 문장들`. 공통 문장 선택 · 단계 · 풀은 같다 (공용 풀). 열 `prefix_position`, `prefix_start` 추가
- 분할 (`prefix_split.suffix_cut_points`): 상한 32는 원본 부분에만, 공통 문장 안은 모두 자름, **원본 끝에서 항상 한 번 자름** (원본과 공통 문장이 한 조각에 섞이지 않게). 이렇게 하면 원본 부분 분할이 L0와 같다 (예제로 확인)
- M4: 뒤쪽 공통 조각 가중치 0. 문맥 오염 2×2의 iso = L0 조각 그대로
- 데이터: Classification 공용 풀 · 8B, LmArena 합친 정답. 오프라인 L0 · L4 · L8 (`--save-scores`), 문맥 오염 L4 · L8
- 비교: 같은 데이터의 앞쪽 버전 (12절 재실행, 13.4)

### 17.2 결과: 뒤에 붙이기 (2026-10-28, NVIDIA L4)

출력: [`exp6_position/`](mvr-cache/benchmarks/prefix/exp6_position/). 첫 실행에서 L0에도 `[SEP]` 앞을 강제로 자르는 버그가 있어 (L0 M4 ≠ M2, 정렬 0%) 고친 뒤 다시 돌렸다 (`7852b04`). 고친 뒤 L0 M2는 앞쪽 버전과 같고(0.7335 · 0.6970), 문맥 오염 정렬 비율 1.000.

| 데이터 | 위치 | L8 M2 (`auc_selected`) | L8 M3 회복률 `auc_selected` | L8 M3 회복률 `auc_pairs` | L8 조각 cos | L8 ctx × L0 | L8 iso × L |
|---|---|---|---|---|---|---|---|
| LmArena (R 0.734 / 0.826) | 앞 | 0.616 | 0.66 [0.36, 1.39] | 0.65 [0.56, 0.75] | 0.844 | −0.123 | +0.126 |
| LmArena | **뒤** | 0.534 | 0.10 [−0.06, 0.29] | 0.19 [0.15, 0.24] | 0.909 | −0.142 [−0.208, −0.081] | +0.019 |
| Classification (R 0.697 / 0.613) | 앞 | 0.624 | 0.36 [0.24, 0.48] | 0.15 [0.13, 0.17] | 0.873 | −0.043 | +0.014 |
| Classification | **뒤** | 0.644 | 0.47 [0.32, 0.66] | 0.27 [0.24, 0.29] | 0.932 | −0.035 [−0.048, −0.023] | +0.007 |

L4도 방향이 같다 (LmArena 뒤: 회복률 0.20 / 0.15, Classification 뒤: 0.40 / 0.24).

### 17.3 읽은 것

1. **뒤에 붙여도 문맥 오염이 손실의 대부분이다.** ctx × L0가 두 데이터 모두 구간이 0 아래 (LmArena −0.14, Classification −0.035), iso × L은 0 근처.
2. **조각 벡터는 덜 바뀌는데(cos 0.91~0.93 > 앞 0.84~0.87) LmArena의 손실은 더 크다.** cos만으로 손실 크기를 설명할 수 없다. 짧은 프롬프트 뒤에 공통 문장이 오면 문장 벡터 검색(M1)이 더 크게 무너지고 (L8 `auc_selected` M1 0.512), 후보 집합이 L0와 70% 겹치는데도 점수가 낮다.
3. **LmArena의 양성 결과(13.4)는 앞에 붙였을 때만 나온다.** 뒤에 붙이면 회복률 0.10~0.20으로 크게 떨어진다. 학습 가중치 효과가 공통 문단의 위치에 민감하다.
4. **Classification은 뒤에 붙이면 하락이 작고 (L8 0.053 vs 0.073) 회복률이 오른다 (0.47).** 그래도 0.5 기준에는 못 미친다.
5. 14절의 인코딩 분리(마스크 · 잘라내기)는 위치와 무관하게 적용할 수 있다. 뒤에 붙는 경우의 마스크 실험은 하지 않았다.

## 18. M3가 공통 조각에 준 가중치 (2026-10-28)

저장된 M3 가중치 헤드(`prefix_w_variant_L*.pt`)를 test 앞 2,000행에 적용해 측정. 공통 조각과 원본 조각이 모두 있는 행만.

| 데이터 · 위치 | 단계 | 공통 조각 가중치 합 | 균등이었다면 | 원본 / 공통 조각 1개당 가중치 비 (중앙값) |
|---|---|---|---|---|
| Classification · 앞 | L4 / L8 | 0.01 / 0.04 | 0.40 / 0.55 | 112배 / 125배 |
| LmArena · 앞 | L4 / L8 | 0.02 / 0.06 | 0.56 / 0.70 | 111배 / 89배 |
| LmArena · 뒤 | L4 / L8 | 0.18 / 0.03 | 0.56 / 0.70 | 8배 / 199배 |

→ M3는 공통 여부를 모른 채 공통 조각을 거의 0으로 낮춘다. 회복이 부족한 이유는 가중치 학습이 아니라 남은 원본 조각 벡터의 문맥 오염이다 (10 · 14절).

## 19. 실험 7: 문맥 오염에 강한 인코더 (LoRA 불변성 학습, 설계 밖, 2026-10-28, 브랜치 `prefix-exp7-invariant-encoder`)

**왜:** 10~18절에서 회복이 부족한 원인은 문맥 오염이었다 (M3는 공통 조각 가중치를 이미 1~6%로 낮춤). 인코딩을 나누는(14절) 대신 인코더 자체가 공통 문장에 흔들리지 않게 학습할 수 있는지 본다.

**이 실험은 1단계 결과(미달)를 본 뒤 추가한 탐색이다.** 그래서 아래 설계와 판정 기준을 실행 전에 이 커밋으로 고정한다.

### 19.1 설계 (실행 전 고정)

| 항목 | 내용 |
|---|---|
| 데이터 | **LmArena 하나** (13절 분할, 합친 정답 `id_set_merged`). 학습은 **train만**, 에폭 선택은 **val만**, **test는 학습 · 선택에 쓰지 않음** |
| 공통 문장 | 학습: 새로 쓴 [`prefix_pool_train.json`](mvr-cache/benchmarks/prefix/prefix_pool_train.json) (칸 10 × 변형 6, 칸 역할이 평가 풀과 다름, 평가 문장과 겹치는 문장 0개 확인, 문장 cos 최대 0.83 · 평균 최댓값 0.70 < 평가 풀 내부 0.81). 평가: 기존 `prefix_pool.json` (**학습 때 본 적 없음**) |
| 모델 | bge-base-en-v1.5 + LoRA (모든 층 query · value, r 16, alpha 32, dropout 0.1). 원래 가중치와 그 dropout은 고정 · 끔. 학습 파라미터 589,824개 (0.5%) |
| 손실 (라벨 미사용) | 선생 = 고정된 원래 모델에 원본만, 학생 = LoRA 모델에 공통 문장을 붙인 프롬프트. ① 불변성: 원본 조각끼리 (1 − cos), ② 유지: 공통 문장 0개인 행 (25%)에서 같은 값. 합 = ① + ②. 공통 문장은 행마다 1~8개 무작위, 20%는 원본 뒤에 붙임 |
| 최적화 | AdamW lr 1e-4, weight decay 0.01, warmup 5% 후 선형 감소, 배치 32, fp16, gradient clipping 1.0, 최대 5 에폭, 시드 0 |
| 에폭 선택 | 매 에폭 **val**에서 같은 두 손실을 계산 (행마다 고정된 증강: 단계 0·2·4·6·8 순환). 합이 가장 낮은 에폭. 2 에폭 연속 안 줄면 중단. **라벨 · test 미사용** |
| 대조군 | 같은 설정, 공통 문장을 항상 0개 (② 유지 손실만). "더 학습했다는 것만으로" 생긴 변화를 분리 |
| 저장 | 고른 에폭의 LoRA를 원래 가중치에 합쳐 일반 transformers 모델로 저장 (구조 · 크기 · 속도 = bge-base) |

### 19.2 평가와 판정 기준 (실행 전 고정)

같은 캐시 정책 안에서 **인코더만 바꿔** 비교한다: vCache 방식(M1) base vs LoRA vs 대조군, MVR 방식(M2 · M3 · M4) base vs LoRA vs 대조군. base는 13.4의 결과(같은 데이터 · 앵커 · 설정).

- 실행: `prefix_offline.py --model <모델>` (L0 · L4 · L8, `--save-scores`), `prefix_contamination.py --model <모델>` (L4 · L8). 일반화: 뒤에 붙인 데이터(17절)로 문맥 오염 2×2. 지연: `prefix_encoder_latency.py` (batch 1, p50 · p95)
- **주 기준 (성공 조건, 셋 다)**
  1. L8 문맥 오염 손실 (ctx × L0 − R, `auc_selected`)이 base의 **절반 이하** (base −0.123 → −0.062 이상), 그리고 LoRA의 95% 구간이 base 값(−0.123)을 포함하지 않음
  2. 같은 기준을 대조군은 만족하지 못함 (효과가 불변성 손실에서 나왔음을 확인)
  3. R 유지: LoRA 모델의 L0 M2가 base R보다 `auc_selected`, `auc_pairs` 모두 **0.01 넘게 떨어지지 않음**
- 보조 (참고): L4 문맥 오염, 조각 cos (ctx vs iso), M1 · M2 · M3 · M4의 L4 · L8 값과 회복률, 통과 판정, 뒤에 붙인 데이터에서의 오염 손실, 지연 p50 · p95

### 19.3 결과 (2026-10-28, NVIDIA L4)

출력: [`exp7_invariant_encoder/`](mvr-cache/benchmarks/prefix/exp7_invariant_encoder/) (합친 모델 가중치는 용량 때문에 커밋하지 않음, `prefix_invariance_train.py`로 재현). base = 13.4 결과.

**학습:** 5 에폭 모두 val 손실이 줄어 마지막 에폭(4) 선택. val 불변성 손실 0.151 → 0.027, 유지 손실 0.003. **대조군은 설계상 빈틈으로 사실상 학습되지 않았다**: LoRA가 0으로 시작해 학생 = 선생이므로 유지 손실이 처음부터 0이고, 선택 에폭이 −1(학습 전) → 대조군 모델 = base. 평가 값도 base와 같다 (R 차이 −0.0001). 따라서 기준 2는 자명하게 만족하며, "더 학습한 효과"를 분리하는 역할은 하지 못했다.

**주 기준 판정: ✅ 셋 다 만족**

| 기준 | base | LoRA | 대조군 | 판정 |
|---|---|---|---|---|
| 1. L8 문맥 오염 손실 (ctx × L0 − R, `auc_selected`) | −0.123 [−0.205, −0.046] | **−0.020** [−0.065, +0.027] | −0.133 | ✅ 절반(−0.062) 이하, 구간이 −0.123을 포함하지 않음 |
| 2. 대조군은 기준 1 불만족 | | | −0.133 | ✅ (대조군 = base라 자명) |
| 3. R 유지 (L0 M2) | 0.7335 / 0.8256 | 0.7410 / 0.8273 (차이 +0.008 [−0.004, +0.022] / +0.002 [−0.002, +0.006]) | 0.7334 / 0.8257 | ✅ 떨어지지 않음 |

**같은 캐시 정책에서 인코더만 바꾼 비교** (`auc_selected` / `auc_pairs` / top1_acc)

| 정책 | 단계 | base | LoRA |
|---|---|---|---|
| vCache (M1) | L4 | 0.618 / 0.722 / 0.978 | **0.739 / 0.805 / 0.991** |
| vCache (M1) | L8 | 0.617 / 0.664 / 0.948 | **0.709 / 0.782 / 0.990** |
| MVR 균등 (M2) | L4 | 0.613 / 0.721 / 0.977 | **0.737 / 0.806 / 0.991** |
| MVR 균등 (M2) | L8 | 0.616 / 0.667 / 0.950 | **0.707 / 0.782 / 0.990** |
| MVR 학습 가중치 (M3) | L8 | 0.694 / 0.769 / 0.965 | 0.689 / 0.780 / 0.991 |
| MVR 공통 조각 0 (M4) | L8 | 0.691 / 0.776 / 0.968 | 0.701 / 0.782 / 0.991 |
| (R = L0 M2) | L0 | 0.734 / 0.826 / 0.992 | 0.741 / 0.827 / 0.992 |

**문맥 오염 2×2 · 일반화** (R과의 차이, `auc_selected`)

| 데이터 | 인코더 | L8 조각 cos | L8 이웃 겹침 | L8 M2 | L8 ctx × L0 |
|---|---|---|---|---|---|
| 앞에 붙임 (평가 풀) | base | 0.844 | 0.55 | −0.117 | −0.123 |
| 앞에 붙임 (평가 풀) | LoRA | **0.963** | **0.80** | −0.034 [−0.085, +0.017] | −0.020 |
| 뒤에 붙임 (학습의 20%만) | base | 0.909 | 0.70 | −0.199 | −0.142 |
| 뒤에 붙임 | LoRA | **0.973** | **0.85** | −0.071 [−0.134, −0.013] | −0.029 [−0.067, +0.009] |

**지연 (batch 1, L8 test 1,000개):** base p50 13.35 ms · p95 14.22 ms, LoRA 13.29 · 13.74, 대조군 13.39 · 13.92 → 차이 없음 (합친 모델은 구조가 같음).

### 19.4 읽은 것

1. **문맥 오염이 학습으로 크게 줄어든다.** 학습 때 본 적 없는 평가용 공통 문장에서도 L8 오염 손실 −0.123 → −0.020, 조각 cos 0.84 → 0.96. 뒤에 붙인 경우(학습의 20%)에도 −0.142 → −0.029.
2. **vCache와 MVR 모두 좋아진다.** 인코더만 바꿔도 L8 `auc_selected` M1 0.617 → 0.709, M2 0.616 → 0.707, 1등 정답률 0.95 → 0.99. 문장 벡터(후보 검색)도 덜 흔들린다 (이웃 겹침 0.55 → 0.80).
3. **LoRA 인코더에서는 학습 가중치(M3)가 더 이상 필요 없다.** M2(균등)가 M3보다 같거나 높다 (L8 0.707 vs 0.689). 1단계 기준으로 판정하면 하락 0.034로 조건 1은 겨우 넘지만 M3 회복률이 음수라 "통과"가 아니다. 하락 자체가 거의 사라졌기 때문이며, 사전 해석표의 "M2가 떨어지지 않음 → 가중치 필요 없음"에 가까운 상태다.
4. R은 오히려 조금 오른다 (+0.008, 구간이 0 포함). 지연 비용 없음.

### 19.5 한계

- **학습용 공통 문장도 "어시스턴트 지시문" 형식**이다. 문장은 겹치지 않지만(최대 cos 0.83) 형식이 비슷해, 다른 모양의 공통 텍스트(예: 긴 문서 머리말, 코드 템플릿)에서 같은 효과가 있는지는 모른다.
- 데이터 1개 (LmArena, 정답은 결과 본 뒤 합친 묶음), 시드 1개.
- 대조군이 base와 같아져 "학습 자체의 효과"를 분리하지 못했다. 공통 문장 대신 무관한 문장을 붙이되 불변성 손실 없이 학습하는 등의 대조군이 필요하다.
- Classification · 다른 인코더에서의 재현, 2단계 41K 스트림 검증은 하지 않았다.

## 20. 실험 8: 2단계 41K 스트림 — 히트율 (설계 밖, 2026-10-28, 브랜치 `prefix-exp8-stage2-hitrate`)

**왜:** 1단계 지표(AUC)는 진단용이고, 최종 결과는 실제 캐시의 **히트율 · 오류율**이다. 실험 7의 LoRA 인코더가 히트율을 올리는지, 논문의 MVR-Cache(RL 분할기)를 포함한 비교군과 함께 잰다. **이 설계와 판정 기준은 실행 전에 이 커밋으로 고정한다.**

### 20.1 설계 (실행 전 고정)

| 항목 | 내용 |
|---|---|
| 데이터 | LmArena test 40,964행 (`fits_all_levels`). 순서 그대로 (같은 묶음이 연달아 나오는 비율 0.07% ≈ 무작위 0.05%) |
| 단계 | **L8** (공용 풀 공통 문장 8개, 앞에 붙임, 주 비교) · **L0** (원본, 기준) |
| 정답 | 같은 `id_set` = 13.3의 합친 묶음 (`id_set_merged`). `--similarity-evaluator benchmark_id_set` |
| 캐시 판정 | vCache verified 정책, δ = 0.01, `--sleep 0.02`, 시드 1 · 2 · 3 (L0은 시드 1) |
| 집계 | 앞 2,000행 제외 (캐시가 차는 구간) → 38,964행. 히트율 = 재사용 / 요청, 오류율 = 틀린 재사용 / 요청 |

**조건 (같은 캐시 정책 안에서 인코더만 바꿈)**

| 조건 | 캐시 정책 | 인코더 |
|---|---|---|
| A / B | vCache (문장 벡터 1개, 논문 기준선) | base / **LoRA** |
| C / D | MVR, 규칙 분할 (`. , ! ? ; :`, 상한 32) + 균등 가중치, 후보 10개 재순위 | base / **LoRA** |
| E | MVR, 규칙 분할 + 학습 가중치 (M3, 13.4의 L8 헤드) — 원래 제안 | base |
| F / G | **MVR-Cache (논문)**: RL 분할기 + `multivector_top_k` k 10 + `--mix-fullcos --include-full-embedding` | base / **LoRA** |

- RL 분할기: 논문 레시피 그대로 (`RL4COTrainer.py`, `--label_mode id_set --train_sampling_mode anchor_nn --policy_mode separate --punctuation_only --bce_auto_balance`, 배치 8, lr 1e-4, 최대 200 에폭, 5 에폭마다 val, 조기 종료). **LmArena train 3,007행 · 원본 프롬프트(L0)로 학습, val로 선택, test 미사용.** 논문 체크포인트(SearchQueries 10K)는 이 VM에 없어 새로 학습했다. val reward가 가장 높은 체크포인트를 쓴다. 분할기는 base 인코더로 학습했으므로 G(LoRA)에서는 분할기와 인코더가 어긋날 수 있다.
- 인코더 교체: `BGE_MODEL_PATH` 환경변수 (평가 코드 수정 없음). LoRA = 19절 합친 모델
- 논문과 다른 점: 논문은 MVR-Cache에 `--sleep 0.1`을 썼다. 여기서는 실행 시간 때문에 **모든 조건을 0.02로 같게** 했다.

### 20.2 판정 기준 (실행 전 고정)

1. **L8에서 LoRA의 히트율이 base보다 높다**: vCache(B > A), MVR 규칙(D > C), MVR-Cache 논문(G > F) 각각, **시드 3개 모두**에서
2. 그때 LoRA의 오류율이 δ(0.01)를 넘지 않는다 (모든 시드)
3. L0에서 LoRA의 히트율이 base보다 **0.5%p 넘게 낮지 않다**
- 보조: 히트율 회복률 (H_L8,LoRA − H_L8,base) / (H_L0,base − H_L8,base), 정책 간 비교 (A · C · E · F), 요청당 지연

### 20.3 부수 수정

- `eval_sembenchmark_verified.py`: 결과 JSON은 저장되지만 벤치마크 로그 저장에서 numpy float32 직렬화 오류 → `default=float`
- `prefix_invariance_train.py`: 학습 전이 가장 좋을 때 마지막 에폭 가중치가 저장되던 버그 수정 (19.3 대조군이 base와 최대 0.00085 달랐던 원인). 19절 결과는 다시 돌리지 않음

### 20.4 결과 (2026-10-28)

집계: `python -m benchmarks.prefix.prefix_stage2_report --runs ../data/exp8/runs --out ../data/exp8/report` → `benchmarks/prefix/exp8_stage2/`. 앞 2,000행 제외 38,964행. 히트율 = (tp + fp) / n, 오류율 = fp / n. 시드 간 표준편차는 히트율 0.1%p 안팎.

**히트율 (오류율), L8은 시드 3개 평균 · L0은 시드 1**

| 조건 | L0 | L8 | L8 − L0 |
|---|---|---|---|
| A vCache + base | 21.74% (0.07%) | 17.54% (0.18%) | −4.20%p |
| B vCache + **LoRA** | 21.67% (0.05%) | **20.71%** (0.05%) | −0.96%p |
| C MVR 규칙 균등 + base | 20.67% (0.09%) | 18.59% (0.18%) | −2.08%p |
| D MVR 규칙 균등 + **LoRA** | 20.45% (0.09%) | **20.25%** (0.09%) | −0.20%p |
| E MVR 규칙 + 학습 가중치(M3) + base | — | 20.19% (0.18%) | — |
| F MVR-Cache 논문 (RL) + base | 20.24% (0.08%) | 18.54% (0.23%) | −1.70%p |
| G MVR-Cache 논문 (RL) + **LoRA** | 20.04% (0.07%) | 18.68% (0.24%) | −1.36%p |

**판정 (20.2)**

| 기준 | vCache (B vs A) | MVR 규칙 (D vs C) | MVR-Cache 논문 (G vs F) |
|---|---|---|---|
| 1. L8에서 LoRA > base, 시드 3개 모두 | **통과** (+3.17 / +3.14 / +3.20%p) | **통과** (+1.55 / +1.85 / +1.59%p) | **불통과** (+0.25 / **−0.07** / +0.23%p) |
| 2. LoRA 오류율 ≤ δ (모든 시드) | **통과** (최대 0.07%) | **통과** (최대 0.11%) | 통과 (최대 0.26%) |
| 3. L0에서 LoRA가 0.5%p 넘게 낮지 않음 | **통과** (−0.08%p) | **통과** (−0.22%p) | 통과 (−0.20%p) |
| 보조: 히트율 회복률 | **0.75** | **0.80** | 0.08 |


**해석**

- **LoRA 인코더는 히트율에서도 효과가 있다.** vCache는 공통문단 8개로 잃은 히트율 4.20%p 중 3.17%p를 되찾아(회복률 0.75) L8에서 20.71%가 됐고, 규칙 MVR도 0.80을 되찾았다. 오류율은 오히려 낮아졌다 (vCache 0.18% → 0.05%). 원본(L0)에서는 거의 그대로다 (−0.08%p, −0.22%p). 1단계 오프라인 결과(19절)가 실제 캐시에서도 재현됐다.
- **논문 MVR-Cache(RL 분할기)에서는 LoRA 효과가 거의 없다** (+0.14%p, 시드 하나는 음수 → 기준 1 불통과). 미리 적은 한계대로, 분할기를 base 인코더로 학습해 LoRA 인코더와 어긋난 것이 가장 그럴듯한 이유다. 또 이 분할기는 프롬프트당 분할 최대 4개(`max_segments=4`)이고 val reward가 첫 체크(에폭 4, −0.019) 뒤로 좋아지지 않았다 (에폭 9 −0.070, 14 −0.035, 이후 19 · 24 · 29도 개선 없음 → 에폭 29에서 조기 종료. F · G에 쓴 에폭 4가 최종 선택 체크포인트와 같다). LmArena에서 RL 분할기가 잘 학습되지 않았을 수 있다.
- **정책 간 비교 (L8, base):** vCache 17.54% < F 논문 MVR-Cache 18.54% ≈ C 규칙 MVR 18.59% < E 규칙 + 학습 가중치 20.19%. 공통문단에는 MVR 계열이 vCache보다 강하고, 학습 가중치(M3)가 가장 좋다 (1단계 15절의 M3 결과와 같은 방향). 반대로 L0에서는 vCache가 가장 높다 (21.74% vs 20.24 ~ 20.67%).
- **가장 높은 L8 히트율은 vCache + LoRA (20.71%)** 로, 원본 vCache(21.74%)에 1%p 차이까지 왔다. 오류율도 가장 낮다.
- **요청당 시간** (병렬 4 ~ 5개 실행 · `--sleep 0.02` 포함이라 상대 비교만): vCache ≈ 51 ms, 규칙 MVR ≈ 90 ms (L0 67 ms), 논문 MVR-Cache ≈ 130 ms (RL 분할기 추론이 큼). LoRA는 시간 차이가 없다 (19절의 인코더 지연 p50 13.3 ms와 같음).

**한계**

- 데이터셋 하나(LmArena), 공통문단 위치는 앞, 단계는 L0 · L8만. L0은 시드 1개.
- RL 분할기는 논문 체크포인트가 아니라 여기서 새로 학습한 것 (val 기준 에폭 4). 논문 수치와 직접 비교하면 안 된다.
- `--sleep 0.02` (논문 MVR-Cache는 0.1).

## 21. 주장 A: 공통 문단 환경에서 규칙 분할 + 학습 가중치가 vCache · MVR-Cache보다 낫다 (브랜치 `claim-a-rule-weighting`)

정리 문서: [`CLAIM_A.md`](CLAIM_A.md). 이 절은 근거 · 약점 · 정한 것 · 보강 계획을 기록한다.

### 21.1 이 브랜치의 출처

- 실험 7 브랜치(`prefix-exp7-invariant-encoder`, `a8c9d57`) 위에 실험 8을 다시 올렸다. 실험 8 원래 커밋(`ef25c23` 설계, `6f8f966` 결과 등)은 Colab VM에서 push되지 않은 채 VM과 함께 사라졌다.
- 코드 · 실행 스크립트 · 20절 문서는 VM 세션 기록에 남은 명령 · 파일 내용에서 그대로 복원했다. 데이터 준비 · RL 학습은 원래 인라인 명령이라 같은 내용으로 스크립트로 옮겼다 (`scripts/make_stage2_lmarena_data.py`, `scripts/train_rl_splitter_lmarena.sh`).
- 결과 [`exp8_stage2/stage2_runs.csv`](mvr-cache/benchmarks/prefix/exp8_stage2/stage2_runs.csv)는 VM에서 `prefix_stage2_report.py`가 출력한 실행별 표(소수 6자리)에서 복원했다. 실행별 원본 JSON · timing · RL 학습 로그 · RL 체크포인트 · LoRA 모델은 없다.

### 21.2 근거 (실험 8, base 인코더, LmArena L8, 앞 2,000행 제외 38,964요청)

| 시드 | 우리 방식 E | vCache A | 논문 MVR-Cache F | 규칙 균등 C | E − A | E − F | E − C |
|---|---|---|---|---|---|---|---|
| 1 | 20.32% | 17.64% | 18.57% | 18.77% | +2.68%p | +1.75%p | +1.55%p |
| 2 | 20.00% | 17.47% | 18.52% | 18.43% | +2.53%p | +1.48%p | +1.57%p |
| 3 | 20.25% | 17.52% | 18.54% | 18.56% | +2.73%p | +1.70%p | +1.69%p |
| 평균 | **20.19%** | 17.54% | 18.54% | 18.59% | **+2.65%p** | **+1.65%p** | **+1.60%p** |

- E 오류율 최대 0.20% (δ = 1% 이내). 세 비교 모두 시드 3개에서 같은 방향.
- 1단계 오프라인도 같은 방향: L8 `auc_selected` LmArena M1 0.617 · M2 0.616 · **M3 0.694** · M4 0.691 (13.4 · 19.3의 base), Classification 4개 풀 × 라벨 조합 모두 M3 > M4 > M2 (8 · 11절).

### 21.3 약점

1. **사전 기준 미달 · 보조 분석**: 1단계 사전 고정 기준은 미달(8절)이라 설계 규칙상 2단계가 없다. 실험 8은 결과 본 뒤 설계했고, 사전 기준은 LoRA vs base였다. E vs A · F는 보조 분석(정책 간 비교)이다. 데이터(LmArena)와 정답 정의(`id_set_merged`, 13.3)도 결과를 본 뒤 정했다.
2. **E의 L0 없음**: L0에서는 vCache 21.74% > 규칙 균등 20.67% > MVR-Cache 20.24%. M3 헤드는 단계별로 따로 학습했고(E는 L8 헤드), 공통 문단 수를 모르는 섞인 스트림에서 헤드 하나로 되는지 확인하지 않았다.
3. **데이터셋 1개**: LmArena, 공통 문장 풀 1개(어시스턴트 지시문), 앞에만, L0 · L8.
4. **RL 분할기 재학습**: 논문 체크포인트(SearchQueries 10K)가 공개되지 않아 LmArena train으로 논문 레시피 그대로 학습. val reward가 첫 검증(에폭 4) 뒤로 좋아지지 않았고, 프롬프트당 조각 최대 4개.
5. **공통 문단 정보 사용**: 규칙 분할의 상한 32를 `cap_scope original`로 적용하려고 공통 문장 목록(`--prefix-pool`)으로 공통 문단 끝을 찾았다. 실제 서비스에서는 쓸 수 없는 정보다. 가중치(M3)는 이 정보를 쓰지 않는다.
6. **같은 공통 문장으로 학습 · 평가**: M3 헤드는 test와 같은 공통 문장 40개(`prefix_pool.json`)가 붙은 train으로 학습했다. 처음 보는 템플릿에서 되는지는 확인하지 않았다.
7. **`--sleep 0.02`**: 논문 README는 MVR-Cache에 0.1. sleep이 짧으면 백그라운드 관측 갱신이 덜 끝나 탐색(LLM 호출)이 늘고, 갱신이 무거운 MVR 계열이 더 불리할 수 있다.

### 21.4 정한 것

| 항목 | 결정 |
|---|---|
| MVR-Cache 기준선 | 논문 체크포인트가 공개되지 않았으므로 **여기서 학습한 RL 분할기**(논문 레시피 그대로)를 기준선으로 쓴다. 학습 곡선과 조각 수 분포를 함께 보고한다 |
| `--sleep` | **모든 조건 0.1** (`SLEEP=0.1`). 조건마다 기다리는 시간을 같게 |
| δ (`--delta`) | 0.01 (논문 본문과 같음) |
| 상한 32 | **`cap_scope all`** (`CAP_SCOPE=all`): 공통 문장 목록을 쓰지 않고 맨 앞부터 셈 |
| 정답 | LmArena는 `id_set_merged`(13.3)로 사전 고정 |
| M3 일반화 | 처음 보는 템플릿 조건 추가: 헤드를 `prefix_pool_train.json`(19절, 평가 풀과 겹치는 문장 0개)이 붙은 train으로 학습, `prefix_pool.json`으로 평가 |

`scripts/run_stage2_lmarena.sh`는 `SLEEP` · `CAP_SCOPE` 환경변수를 받는다. 기본값은 실험 8에서 쓴 값(0.02, original)이라 20절 결과를 그대로 재현한다.

### 21.5 보강 실험 계획 (실행 전에 이 설계를 커밋으로 고정)

| ID | 내용 | 약점 |
|---|---|---|
| A0 | 실험 8 재현: 기본 설정(0.02, original)으로 LmArena L8 시드 1의 A · C · E · F가 21.2와 ±0.2%p 안에서 일치 | 출처 |
| A1 | 21.4 설정(0.1, all)으로 LmArena L0 · L8, A · C · E · F, 시드 1 · 2 · 3 | 2 · 5 · 7 |
| A2 | 같은 설정으로 Classification(cls_seed42, `fits_all_levels`) L0 · L8 | 3 |
| A3 | 처음 보는 템플릿: 헤드를 학습용 풀로 학습한 E′를 A1 · A2 스트림에 추가 | 6 |
| A4 | 참고: vCache + LoRA 인코더(B)를 같은 표에 함께 보고 | 리뷰어 질문 |

**판정 기준 (주장 A 성립 조건, 데이터마다)**

1. L8에서 E의 히트율이 A(vCache)와 F(MVR-Cache 기준선)보다 시드 3개 모두 높음
2. E 오류율 ≤ δ, 모든 단계 · 시드
3. L0에서 E 히트율이 C(규칙 균등)보다 0.5%p 넘게 낮지 않음 (L0에서 A와의 차이는 그대로 보고)
- 보조: E − C, E′ − E (처음 보는 템플릿에서 잃는 몫), 요청당 시간, B와의 차이

### 21.6 재현

```bash
# 런타임마다 (7.1절과 같음)
git clone -b claim-a-rule-weighting https://github.com/sgwag6799/MVR-Cache.git /root/MVR-Cache
cd /root/MVR-Cache && source colab_setup.sh
# LmArena 데이터 (13절): 분할 지문 train ec6e778ee76f / val 411c8e1b6412 / test 3e5cc1fe8fa8
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkLmArena','train.parquet',repo_type='dataset',local_dir='data/raw_lmarena')"
python scripts/make_lmarena_splits.py --raw data/raw_lmarena/train.parquet --out data/lmarena
(cd mvr-cache && python -m benchmarks.prefix.make_prefix_data --train ../data/lmarena/train.parquet \
  --val ../data/lmarena/val.parquet --test ../data/lmarena/test.parquet --out ../data/lmarena/prefix/data)
python scripts/lmarena_merge_duplicate_sets.py --raw data/raw_lmarena/train.parquet --data data/lmarena/prefix/data
# 실험 8 입력 (stream L0 · L8 각 40,964행, rl train 3,007 / val 1,000)
python scripts/make_stage2_lmarena_data.py
bash scripts/train_rl_splitter_lmarena.sh                    # MVR-Cache 기준선 분할기 → data/exp8/rl/best.ckpt
# 실험 8 그대로 (A · C · E는 batch1, F는 batch2에 있음. B · D · G는 data/exp7/lora/model 필요)
bash scripts/run_stage2_lmarena.sh scripts/stage2_jobs_batch1.txt 5
bash scripts/run_stage2_lmarena.sh scripts/stage2_jobs_batch2.txt 4
# 보강 실험 설정
SLEEP=0.1 CAP_SCOPE=all OUT=data/claimA/runs bash scripts/run_stage2_lmarena.sh <jobs.txt> 4
(cd mvr-cache && python -m benchmarks.prefix.prefix_stage2_report --runs ../data/exp8/runs --out ../data/exp8/report)
```
