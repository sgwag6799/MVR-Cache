# 방언 ↔ 표준어 의미 캐시 실험 (`dialect-cache-experiment` 브랜치)

> **답변은 한국어로 해 주세요.** `main`(원저자 코드)에서 딴 브랜치이며, `gpu-production`의 분할·가중치·로그 실험과는 별개다.

## 0. 한눈에 보기

| | 내용 |
|---|---|
| 무엇을 | 사용자가 **같은 질문을 경상방언으로** 했을 때, 의미 캐시가 캐시에 있는 **표준어 문장**을 같은 요청으로 알아보는가 |
| 왜 | 의미 캐시의 핵심은 "표현은 달라도 뜻이 같은 요청"을 재사용하는 것. 방언은 실제 사용자 발화에서 가장 흔한 다른 표현이고, 원 논문은 영어 데이터만 다룸 |
| 비교 | 문장 인코더 2개: 논문의 영어 전용 `BAAI/bge-base-en-v1.5` vs 다국어 `BAAI/bge-m3` |
| 방식 | **단일 문장 벡터 + 코사인** (Step 1 Vanilla vCache와 같은 방식). 규칙 분할·RL 분할·조각 가중치(IDF/MLP)는 **쓰지 않음** (→ 3절) |
| 본 실험 | 오프라인 검색: 방언 질의 약 32만 개 → 표준어 문장 약 204만 개(캐시) 중 제 짝을 1등으로 찾는가. **Training 질의로 재사용 임계값을 정하고 Validation 질의로 잰다** |
| 보조 실행 | vCache(δ=0.01)를 두 스트림에 돌림: 실제로 여러 번 말해진 문장만(`repeat`) / 인기 있는 질문이 반복되도록 만든(`zipf`) |
| 데이터 | AI Hub **한국어 방언 발화 데이터(경상도)** 라벨 (Training 대화 6,822개 + Validation 843개). **git에 올리지 않고 Google Drive에서 읽음** |

## 1. 무엇에 대한 실험인가

### 1.1 상황
의미 캐시는 이전 요청과 뜻이 같은 새 요청이 오면 LLM을 다시 부르지 않고 저장된 답을 돌려준다. 이때 "뜻이 같다"는 판단은 **두 문장의 벡터가 얼마나 가까운가**로 한다.

| | 문장 |
|---|---|
| 캐시에 있는 요청 (표준어) | 카레를 먹을 때 저는 약간 카레가 매운 거예요 **그러니까** |
| 새로 들어온 요청 (경상방언) | 카레를 먹을 때 저는 약간 카레가 매운 거예요 **긍까** |
| 캐시에 있는 다른 요청 (함정) | 같은 종류의 일상 대화에서 나온, 주제·어휘가 비슷하지만 뜻이 다른 문장 수백만 개 |

좋은 의미 캐시라면 방언 요청의 가장 가까운 캐시 항목이 **같은 뜻의 표준어 문장**이어야 하고, 그 유사도가 **뜻이 다른 문장들보다 확실히 높아야** 임계값 하나로 안전하게 재사용할 수 있다.

### 1.2 구체적으로 확인하는 것
1. **찾는가:** 방언 질의의 1등 캐시 항목이 제 짝(같은 뜻의 표준어 문장)인가 — top-1·top-5 정확도
2. **구분되는가:** 제 짝과의 유사도가 가장 비슷한 오답과의 유사도보다 높은가 — AUC, margin
3. **안전하게 재사용할 수 있는가:** "유사도 ≥ t면 재사용" 규칙 하나로, 오답 재사용을 질의의 1%·5% 이하로 지키면서 몇 %나 재사용하는가 (vCache의 δ와 같은 기준). **t는 Training 질의로 정하고 Validation 질의에서 잰다**
4. **무엇이 어렵게 만드는가:** 방언 어절 비율이 높을수록, 방언형과 표준어형의 글자가 많이 다를수록 얼마나 떨어지는가
5. **인코더 차이:** 위 1~4가 영어 전용 인코더와 다국어 인코더에서 어떻게 다른가
6. **vCache를 그대로 쓰면:** 같은 의미가 반복될 때 vCache가 실제로 몇 %나 재사용하는가, 특히 전에 표준어로 나온 질문을 방언으로 다시 물을 때

### 1.3 가설
1. 영어 인코더는 한국어를 의미가 아니라 하위 단어 조각의 **글자 겹침**으로 비교한다 → 글자 유사도가 낮거나 방언 비율이 높은 질의에서 크게 떨어진다.
2. 다국어 인코더는 그 하락이 작다.
3. 두 인코더 모두 비슷한 일상 문장(함정) 때문에 `auc_hard`가 `auc_random`보다 낮다. 캐시가 커질수록(샘플 4천 → 전체 204만) 함정이 많아져 정확도가 떨어진다.
4. vCache는 같은 의미가 반복돼야 재사용을 배운다(관측 6개 규칙). 반복되는 스트림에서도 방언 형태로 다시 온 요청(`cross_variant`)의 재사용률은 인코더에 크게 좌우된다.

## 2. 데이터

### 2.1 받을 파일 (AI Hub `014.한국어 방언 발화 데이터(경상도)`)

| 받을 것 | 크기(압축) | 파일키 | 용도 |
|---|---|---|---|
| ✅ 1.Training / (new3)라벨링데이터 / `(비식별화완료)경상도_학습데이터_1.zip` | 308MB (풀면 2.9GB) | 572701 | 재사용 임계값을 정하는 쪽 |
| ✅ 2.Validation / (new3)라벨링데이터 / `(비식별화완료)경상도_학습데이터_2.zip` | 32MB (풀면 342MB) | 572713 | 정한 임계값으로 성능을 재는 쪽 |
| ❌ 원천데이터 `(비식별화완료)경상도_1~12.zip` | 각 17~28GB, 합계 약 300GB | — | 음성 WAV — 이 실험은 텍스트만 쓰므로 받지 않음 |

압축을 풀면 대화마다 `.json`(라벨)과 `.txt`(화자 번호가 붙은 전사문)가 하나씩 있다. **`.json`만 쓰고 `.txt`는 무시한다.** 파일 이름 접두어가 `DKSR`과 `DKCI` 두 종류인데 JSON 구조는 같다. Training에는 둘 다(DKSR 4,636 + DKCI 2,186), Validation에는 DKSR(843)만 있다 → 결과를 split·출처(`source`)별로 나눠 볼 수 있게 했다.

**실제 규모** (전체 데이터로 가공해 본 결과)

| | Training | Validation |
|---|---|---|
| 대화 파일 | 6,822 | 843 |
| 발화 | 1,874,263 | 229,497 |
| 방언 질의 (정제 후 방언형 ≠ 표준어형, 표준어 2어절 이상) | 287,073 (DKCI 114,325 + DKSR 172,748) | 30,943 |
| 캐시 (두 split의 서로 다른 표준어 문장 합계) | 2,039,061 | |

- 발화 중 방언형과 표준어형이 실제로 다른 것은 약 15%(샘플 대화 15개에서는 8%)
- 같은 표준어 문장(2어절 이상)이 2번 이상 나온 것 12,136개, **6번 이상 1,134개**, 20번 이상 213개. 가장 많이 반복되는 문장은 "어 저는"(302회), "그래가지고", "아 진짜?", "그래서 저는" 같은 **추임새·말 시작 표현**이다.

### 2.2 Google Drive 배치

압축을 푼 **폴더**를 올려도 되고 **zip**을 올려도 된다(스크립트가 폴더는 하위 폴더까지 `.json`을 찾고, zip은 안의 JSON을 직접 읽는다). 폴더로 올리면 작은 파일 1만 5천 개라 업로드와 첫 가공(Drive에서 파일을 하나씩 읽음)이 느리지만, 가공은 한 번만 하면 된다.

현재 배치 (압축을 풀어 폴더로 올림):
```
MyDrive/dialects/
├─ data/
│   ├─ training/     ← Training 라벨 (.json + .txt, 하위 폴더가 있어도 됨)
│   └─ validation/   ← Validation 라벨
├─ prepared/   ← 셀 2가 만듦 (가공 데이터, 약 260MB)
└─ results/    ← 셀 2가 만듦 (결과)
```
셀 2 맨 위의 `D`, `TRAIN`, `VAL`이 이 경로를 가리킨다. zip으로 올렸다면 `TRAIN`, `VAL`에 zip 경로를 넣으면 된다(macOS가 zip에 넣는 `__MACOSX/`, `._*` 파일은 스크립트가 건너뛴다). 샘플로 돌렸던 예전 파일(`MyDrive/dialect/`)은 남겨 둬도 된다.

### 2.3 git에 올리지 않는 이유
AI Hub 이용약관상 재배포가 제한되고, 화자 정보(나이·성별·출생지 등)가 들어 있으며, 이 저장소는 공개 저장소다. `.gitignore`가 `mvr-cache/data/` 아래의 방언 데이터(`원천데이터/`, `라벨링데이터/`, `*경상도*/` 폴더, `*.zip`, 가공 결과 `dialect/`)와 `*.wav`를 막는다.

### 2.4 가공 (`benchmarks/dialect/prepare_dialect_dataset.py`)
- 입력: `--train`·`--val`에 폴더 또는 **zip 파일**(여러 개 가능)
- 정제: `{laughing}` 같은 태그 제거, `((x))`(불확실 전사) → `x`, `(())` 제거, `~`(장음 표시) 제거, 공백 정리
- **캐시(`index.parquet`)**: Training+Validation의 모든 표준어 문장(같은 문장은 하나로, 띄어쓰기·문장부호 차이 무시)
- **질의(`pairs.parquet`)**: 정제 후 방언형과 표준어형이 실제로 다른 발화, 표준어 2어절 이상. `split`(train/val), `source`(DKSR/DKCI), 방언 비율, 글자 유사도 열 포함
- **vCache 스트림** (`id_set` = 표준어 문장 그룹 → `benchmark_id_set`으로 채점, `kind` 열: `first` 처음 나온 의미 / `exact_repeat` 같은 문장이 전에 나옴 / `cross_variant` 같은 의미가 전에 **다른 형태**로 나옴 = 의미 캐시가 필요한 경우)

| 스트림 | 구성 | 전체 데이터 규모 | 실행 |
|---|---|---|---|
| `stream_repeat` | **실제로 6번 이상 말해진 의미**의 모든 발화를 실제 발화 형태(방언 전사)로 + 한 번만 나온 표준어 문장 30%(긴 꼬리) | 27,046 요청 (cross_variant 741, 방언 형태 971) | ✅ 기본 |
| `stream_zipf` | 질의 쌍을 **Zipf 인기도**(지수 1.1)로 뽑아 요청 5만 개, 요청마다 50% 확률로 방언형/표준어형 + 한 번만 나온 표준어 문장 30%. 반복 빈도는 **인위적** | 50,000 요청 (cross_variant 1,446) | ✅ 기본 |
| `stream_cold` | 표준어 문장 전부 → 방언 질의 전부. 모든 의미가 한 번씩만 | 2,357,077 행 | ❌ vCache로 하루 이상 걸려 전체 데이터에서는 돌리지 않음 (샘플 결과는 7.2절) |

- `manifest.json`: split·출처별 건수, 정제 규칙, 실제 반복 통계와 가장 많이 반복된 문장 15개

## 3. 사용하는 방식 (분할 · 가중치 · 후보 선택 · 점수)

### 3.1 프로젝트 Step 표에서의 위치

| Step (gpu-production 기준) | 분할 | 가중치 | 이 실험에서 |
|---|---|---|---|
| **1 Vanilla vCache** | 없음 (단일 벡터) | — | ✅ **이 방식을 씀** (인코더만 바꿔 비교) |
| 대조군 (후보 10개 재순위) | 없음 | — | ❌ main에 없음 (`gpu-production`에서 추가된 기능) |
| 2 규칙 분할 + 균등 | 규칙(구두점) | 1/N | ❌ main에 규칙 분할기 없음 |
| 3a 규칙 분할 + 휴리스틱 | 규칙 | IDF·길이·중심거리 | ❌ 〃 |
| 3b 규칙 분할 + 학습 가중치 | 규칙 | MLP + BCE | ❌ 〃 |
| 4 MVR-cache 재현 | RL | 1/N | ❌ 아래 3.4 |
| 5 RL 분할 + 적응형 가중치 | RL | MLP | ❌ 〃 |

### 3.2 문장 벡터와 점수
- **문장 하나 = 벡터 하나.** 문장을 자르지 않는다.
- 인코더의 마지막 층 토큰 벡터들을 **마스크 평균**(패딩 제외, [CLS]·[SEP] 포함)한 것이 문장 벡터. vCache 원 코드(`EmbeddingModel.get_embedding`)와 같은 방식이다.
- 두 문장의 유사도 = 문장 벡터의 **코사인**. MaxSim(조각×조각 비교)은 쓰지 않는다.

### 3.3 후보 선택과 재사용 판단

| | 본 실험 (오프라인 검색) | 보조 실행 (vCache) |
|---|---|---|
| 캐시 | 표준어 문장 약 204만 개를 처음부터 다 넣어 둠 | 스트림 순서대로 들어오며, 놓치면 추가(cache-on-miss) |
| 후보 검색 | **전수 비교**(정확한 코사인, 근사 없음) | HNSW 근사 검색으로 **가장 가까운 1개**(`k=1`, `verified.py`) |
| 재사용 판단 | 없음 (순위·유사도만 측정). 임계값 규칙은 사후 계산: Training으로 정하고 Validation에서 측정 | vCache verified 정책(δ=0.01): 항목마다 관측 (s, 정답여부)로 로지스틱 임계값을 학습해 탐색/재사용 결정 |
| 정답 기준 | 1등이 제 짝 문장인가 | 같은 `id_set`(같은 표준어 문장 그룹)인가 |

- 검색 실험의 메모리: 문장 벡터 204만 × 1024를 float32로 들면 8GB라 **벡터는 float16으로 보관**하고, 캐시를 26만 행씩 잘라 float32로 바꿔 질의 1,024개씩 계산한다. 샘플에서 float32로 계산한 결과와 비교하면 top-1·AUC·임계값 히트율은 같고, 동점 근처에서 순위가 하나씩 바뀌는 정도(top-5 0.978 → 0.975)의 차이만 있다.
- vCache는 항목마다 관측이 6개(사전값 2 + 실제 4) 쌓이기 전에는 무조건 탐색한다. 의미마다 문장이 한 번씩만 오면 어떤 항목도 6개를 채울 수 없다(샘플 `cold` 실행에서 확인, 7.2절). 데이터를 키워도 의미 그룹 수가 늘 뿐 그룹당 문장 수는 대부분 그대로라, 반복이 있는 `repeat`(실제)·`zipf`(인위적) 스트림으로 vCache를 시험한다.

### 3.4 분할(규칙·RL)과 조각 가중치(IDF·MLP)를 쓰지 않는 이유
- **main 기준이라 규칙 분할기·조각 가중치 코드가 없다.** 이 기능들은 `gpu-production`에서 추가됐다(`RuleSplitter.py`, `build_segment_weight_stats.py`).
- **RL 분할기는 이 데이터에 맞지 않는다.** 가진 체크포인트는 영어 데이터를 768차원 영어 인코더로 학습한 것이다. bge-m3(1024차원)와는 차원부터 맞지 않고, 한국어 분할 정책은 이 데이터로 RL 학습을 다시 해야 한다.
- **먼저 기본 방식에서 인코더 차이를 확인하는 게 순서다.** 단일 벡터에서도 방언을 못 알아본다면 분할 이전에 인코더 문제다. 분할·가중치 효과는 인코더를 정한 다음 단계로 둔다(8절).

### 3.5 비교 조건

| 조건 | 인코더 | 차원 | 비고 |
|---|---|---|---|
| A | `BAAI/bge-base-en-v1.5` | 768 | 원 논문·vCache 기본값. 영어 전용 |
| B | `BAAI/bge-m3` | 1024 | 다국어(한국어 포함). 약 2.3GB |

HNSW 차원은 첫 벡터로 정해지므로 1024차원도 코드 수정 없이 들어간다. 두 조건 모두 풀링·점수·데이터·순서가 같고 인코더만 다르다.

## 4. 지표

### 4.1 본 실험 (`benchmarks/dialect/eval_dialect_retrieval.py` → `results/retrieval_<모델>.json`)
인코더마다 `all`(전체 질의), `train`, `val`로 나눠 계산한다.

| 지표 | 의미 |
|---|---|
| `top1_acc`, `top5_acc`, `mrr` | 제 짝(표준어 문장)이 1등 / 5등 안 / 역순위 평균 |
| `own_cos_mean`, `best_wrong_cos_mean`, `margin_mean` | 제 짝과의 코사인 / 가장 비슷한 오답과의 코사인 / 그 차이 |
| `auc_hard` | 제 짝의 코사인 vs 가장 비슷한 오답의 코사인 (함정 구분력) |
| `auc_random` | 제 짝 vs 무작위 캐시 문장 (쉬운 기준) |
| `static_threshold` | 같은 질의로 정한 임계값의 오답률 ≤ 1%·5% 최대 히트율 (**낙관적**, 참고용) |
| **`calibrated`** | **Training 질의로 정한 임계값을 Validation 질의에 적용한 히트율·오답률** (실제 운영 추정치. 오답률이 목표를 넘으면 임계값이 일반화되지 않은 것) |
| `by_dialect_ratio`, `by_surface_sim` | 방언 비율(<0.1, 0.1~0.2, ≥0.2) / 글자 유사도(<0.9, 0.9~0.95, ≥0.95) 구간별 top-1 |

질의별 상세(`results/retrieval_<모델>.csv`, 약 32만 행): split, source, 제 짝 코사인, 가장 비슷한 오답의 코사인과 **그 문장**, 순위 → source(DKSR/DKCI)·주제(topic)별로 더 나눠 볼 수 있다.

### 4.2 보조 실행 (`benchmarks/dialect/analyze_dialect_vcache.py` → `results/vcache_<스트림>_summary.csv`)
`kind`(first / exact_repeat / cross_variant) × `variant`(dialect / standard)별로 `hit_rate`(재사용), `error_rate`(틀린 재사용 / 행), `nn_correct_rate`(가장 가까운 캐시 항목이 정답이었던 비율 = 완벽한 재사용 판단으로 도달 가능한 상한). 핵심은 **`cross_variant`** 행(전에 다른 형태로 나온 의미를 다시 물은 요청).

## 5. Colab 실행

`gpu-production` 실험이 돌고 있는 런타임과 **같은 런타임에서 돌리지 말 것**(같은 이름의 패키지를 다른 폴더로 다시 설치하므로 돌던 실험의 코드가 바뀜). **GPU 런타임 필요**(T4 기준으로 맞춤. CPU로는 문장 236만 개 인코딩에 너무 오래 걸림).

**준비:** 2.2절대로 Training·Validation 라벨을 Drive `MyDrive/dialects/data/training/`, `.../validation/`에 올린다.

### 셀 1: 설치 (끝나면 자동 재시작)
```python
from google.colab import drive
drive.mount('/content/drive')
import os
R = "/content/MVR-Cache-dialect"
if os.path.exists(R):
    !git -C {R} pull
else:
    !git clone -b dialect-cache-experiment https://github.com/sgwag6799/MVR-Cache.git {R}
!git -C {R} log --oneline -1
%cd {R}
!pip install -q --ignore-installed packaging
!pip install -q --prefer-binary -e mvr-cache "datasets<4"
!pip install -q --prefer-binary -e mvr-cache/vcache/vcache_core/cache/embedding_store/hnswlib
os.kill(os.getpid(), 9)
```

### 셀 2: 가공 → 검색 실험 → vCache 보조 실행 → 집계 (끊기면 셀 1 → 셀 2 다시, 끝난 단계는 건너뜀)
```python
%cd /content/MVR-Cache-dialect/mvr-cache
import os, glob, json, torch
os.environ.update(HF_ENDPOINT="https://huggingface.co", HF_CACHE_BASE="/content/hf_cache", USE_TF="0")
D = "/content/drive/MyDrive/dialects"
TRAIN, VAL = f"{D}/data/training", f"{D}/data/validation"   # zip으로 올렸다면 zip 경로
PREP, RES = f"{D}/prepared", f"{D}/results"
MODELS = ["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"]
STREAMS = ["repeat", "zipf"]          # "cold"는 전체 데이터에서 236만 행이라 제외
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REBUILD = False                       # True면 가공을 처음부터 다시
q = lambda paths: " ".join(f'"{p}"' for p in paths)

for p in (TRAIN, VAL):
    assert os.path.exists(p), f"{p} 이(가) 없습니다. 위의 D/TRAIN/VAL을 Drive의 실제 경로로 고쳐 주세요: {os.listdir(D) if os.path.isdir(D) else os.listdir('/content/drive/MyDrive')}"
print("Training:", TRAIN, "\nValidation:", VAL)
os.makedirs(RES, exist_ok=True)

# 1) 가공 (Drive 폴더에서 JSON 7,665개를 읽음, 첫 실행은 10~20분 걸릴 수 있음)
if REBUILD or not os.path.exists(f"{PREP}/stream_repeat.parquet"):
    !python benchmarks/dialect/prepare_dialect_dataset.py --train "{TRAIN}" --val "{VAL}" --out-dir "{PREP}"
print(json.dumps({k: v for k, v in json.load(open(f"{PREP}/manifest.json")).items() if k != "cleaning"}, ensure_ascii=False, indent=1))

# 2) 본 실험: 검색 (인코더별로 저장, Training으로 임계값 → Validation으로 측정)
for m in MODELS:
    out = f"{RES}/retrieval_{m.split('/')[-1]}.json"
    if not os.path.exists(out):
        !python benchmarks/dialect/eval_dialect_retrieval.py --data-dir "{PREP}" --models {m} --device {DEVICE} --out "{out}"

# 3) 보조 실행: vCache (스트림 × 인코더, 끝난 것은 건너뜀)
for s in STREAMS:
    outs = []
    for m in MODELS:
        out = f"{RES}/vcache_{s}_{m.split('/')[-1]}.json"
        if not os.path.exists(out):
            !python benchmarks/eval_sembenchmark_verified.py --dataset "{PREP}/stream_{s}.parquet" --delta 0.01 --sleep 0.02 --similarity-evaluator benchmark_id_set --device {DEVICE} --embedding-model {m} --output-json "{out}"
        outs.append(out)
    !python benchmarks/dialect/analyze_dialect_vcache.py --stream "{PREP}/stream_{s}.parquet" --results {q(outs)} --out "{RES}/vcache_{s}_summary.csv"
```

### 결과 파일 (Drive `MyDrive/dialects/`)
- `prepared/` : `index`·`pairs`·`stream_repeat`·`stream_zipf`·`stream_cold.parquet` + `manifest.json`
- `results/retrieval_<모델>.json` : 인코더별 요약(all·train·val·calibrated) / `results/retrieval_BAAI__<모델>.csv` : 질의별 상세
- `results/vcache_<스트림>_<모델>.json` : vCache 실행 결과 (요청별 hit/tp/fp/fn) / `results/vcache_<스트림>_summary.csv` : kind × variant 집계

**예상 시간 (T4, 추정):** 가공 10~20분(Drive 폴더에서 읽기) / 검색 실험 bge-base-en 약 20분, bge-m3 약 1시간(문장 236만 개 인코딩이 대부분) / vCache `repeat`(2.7만 요청) 인코더당 약 20분, `zipf`(5만 요청) 인코더당 30~40분 → 합계 약 3~4시간. 각 단계가 끝날 때마다 Drive에 저장되므로 세션이 끊겨도 이어서 돌릴 수 있다(단, 돌고 있던 단계는 처음부터).

## 6. 변경한 파일 (main 대비)

### 6.1 실행에 필요해서 `gpu-production`에서 가져온 수정 (원저자 코드 그대로는 Colab에서 설치·실행이 안 됨)

| 파일 | 변경 | 이유 |
|---|---|---|
| `mvr-cache/pyproject.toml` | `vllm`을 선택 설치(`mvr-cache[vllm]`)로 옮김. `[tool.poetry] packages = [{ include = "vcache" }]` 추가 | vllm은 평가에 쓰이지 않는데 설치하면 Colab의 torch를 바꿔 버림. 배포 이름과 코드 폴더가 달라 `pip install -e`가 실패하던 문제 |
| `mvr-cache/vcache/inference_engine/strategies/vllm.py` | `from vllm import ...`를 `try/except`로 감싸고, vllm 없이 엔진을 만들면 설치 방법을 알려 주는 에러 | vllm 없이도 `import vcache`가 되게 |
| `mvr-cache/vcache/vcache_core/splitter/embedding_model.py` | 공유 토크나이저를 잠금 래퍼(`_LockedTokenizer`)로 감쌈 | vCache 백그라운드 갱신 스레드와 메인 스레드가 토크나이저를 동시에 쓰면 `Already borrowed` 에러로 후보가 빠지던 문제. 토큰 결과는 동일 |
| `mvr-cache/vcache/vcache_core/cache/embedding_engine/strategies/lang_chain.py` | `HF_ENDPOINT`를 hf-mirror로 강제로 덮어쓰던 두 줄 삭제 | 지정한 Hugging Face 주소가 무시되던 문제 |

### 6.2 이 실험을 위해 바꾼 것

| 파일 | 변경 | 내용 |
|---|---|---|
| `mvr-cache/benchmarks/eval_sembenchmark_verified.py` | 수정 (+7줄) | `--embedding-model` 옵션 추가 → `EmbeddingModel(model_name=...)`. 주지 않으면 기존과 같은 인코더 |
| `mvr-cache/benchmarks/dialect/prepare_dialect_dataset.py` | 새 파일 | 라벨 폴더/zip → 정제 → `index`/`pairs`(split·source 포함)/`stream_repeat`/`stream_zipf`/`stream_cold.parquet` + `manifest.json` (2.4절) |
| `mvr-cache/benchmarks/dialect/eval_dialect_retrieval.py` | 새 파일 | 인코더별 오프라인 검색 평가, Training→Validation 임계값 보정, float16 보관 + 구간별 float32 계산 (4.1절) |
| `mvr-cache/benchmarks/dialect/analyze_dialect_vcache.py` | 새 파일 | vCache 결과를 kind × variant로 나눠 hit·error·`nn_correct_rate` 집계 (4.2절) |
| `.gitignore` | 수정 | 방언 원본·가공 데이터(`원천데이터/`, `라벨링데이터/`, `*경상도*/`, `*.zip`, `dialect/`)와 `*.wav`를 git에서 제외 |
| `DIALECT_EXPERIMENT.md` | 새 파일 | 이 문서 |

판정 규칙(vCache verified 정책), 캐시·HNSW·채점 코드는 손대지 않았다.

## 7. 샘플 결과 (대화 15개, 질의 363개, 캐시 4,426문장 — 전체 데이터 실행 전 참고용)

### 7.1 검색 (Colab T4)

| 지표 | bge-base-en | bge-m3 |
|---|---|---|
| top-1 / top-5 | 0.950 / 0.978 | **0.986 / 0.995** |
| 제 짝 / 가장 비슷한 오답 코사인 | 0.983 / **0.944** | 0.975 / **0.848** |
| margin | 0.040 | **0.128** |
| `auc_hard` / `auc_random` | 0.924 / 0.997 | **0.972** / 0.999 |
| 오답률 1% 이하 최대 히트율 (같은 질의로 정한 임계값) | 68.6% (t=0.982) | **99.4%** (t=0.824) |
| 방언 비율 ≥ 0.2 구간 top-1 | 0.841 | 0.943 |
| 글자 유사도 < 0.9 구간 top-1 | 0.780 | 0.900 |

- 가설 1·2·3 방향. 영어 인코더는 한국어 문장끼리의 유사도가 1 근처로 몰려(오답도 평균 0.944), 오답을 1% 이하로 막으려면 임계값을 매우 높여야 하고 그만큼 재사용을 놓친다. top-1 차이(3.6%p)보다 안전 기준 재사용률 차이(약 30%p)가 훨씬 크다.
- 샘플을 Training 11개·Validation 4개 대화로 나눠 확인한 보정 결과(맥 CPU, bge-base-en): Training으로 정한 "오답 1%" 임계값(0.969)이 Validation에서는 **오답률 2.2%** → 같은 질의로 정한 임계값은 낙관적이라는 점이 실제로 드러남. 전체 데이터에서 `calibrated`로 다시 확인할 것.

### 7.2 vCache (`cold`, Colab T4)

| | 방언 질의 히트 | 가장 가까운 항목이 정답이었던 비율 | 전체 오답 히트 |
|---|---|---|---|
| bge-base-en | 1/363 (0.3%) | 94.5% | 16 |
| bge-m3 | 0/363 (0%) | 98.3% | 5 |

검색은 94~98% 맞혔지만 vCache는 거의 재사용하지 않음(의미마다 한 번씩이라 관측 6개를 못 채움). 오답률은 0.33% / 0.10%로 δ=1% 이내.

## 8. 한계·다음 단계
- **`repeat` 스트림의 반복은 대부분 추임새·말 시작 표현이다**("어 저는", "그래가지고"). 이런 말은 방언형과 표준어형이 거의 같아서 방언 형태 요청은 4%(971개)뿐이다 → 실제 반복으로 vCache를 시험하는 데는 쓰지만, 방언 캐시 효과는 `zipf`와 검색 실험으로 본다.
- `zipf` 스트림의 반복 빈도는 인위적이다(실제 사용자 로그가 아님). 지수(`--zipf-a`)·방언 비율(`--zipf-dialect-prob`)·긴 꼬리 비율(`--zipf-singleton-frac`)을 바꿔 민감도를 볼 수 있다.
- Validation에는 DKSR만 있어 `calibrated`는 DKSR 대화 기준 수치다. DKCI는 Training에서만 `source`별로 볼 수 있다.
- 방언 차이가 대부분 한두 단어라 과제가 쉬운 편. 방언 비율이 높은 질의만 따로 보거나(`by_dialect_ratio`), 다른 지역(전라·제주 등) 데이터로 넓힐 수 있다.
- 다국어 인코더의 풀링은 이 코드의 평균 풀링을 그대로 썼다(bge-m3 권장 방식은 [CLS]). vCache와 같은 방식으로 맞추기 위한 선택이며, 필요하면 비교해 볼 수 있다.
- 분할·가중치로 넓히려면: (1) `gpu-production`의 규칙 분할기·가중치 코드를 이 브랜치에 가져오고, (2) 한국어 문장부호·어절 기준 분할 규칙을 확인하고, (3) RL은 한국어 인코더로 분할 정책을 새로 학습해야 한다.
