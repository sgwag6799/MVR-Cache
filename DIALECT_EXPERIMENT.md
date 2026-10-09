# 방언 ↔ 표준어 의미 캐시 실험 (`dialect-cache-experiment` 브랜치)

> **답변은 한국어로 해 주세요.** `main`(원저자 코드)에서 딴 브랜치이며, `gpu-production`의 분할·가중치·로그 실험과는 별개다.

## 0. 한눈에 보기

| | 내용 |
|---|---|
| 무엇을 | 사용자가 **같은 질문을 경상방언으로** 했을 때, 의미 캐시가 캐시에 있는 **표준어 문장**을 같은 요청으로 알아보는가 |
| 왜 | 의미 캐시의 핵심은 "표현은 달라도 뜻이 같은 요청"을 재사용하는 것. 방언은 실제 사용자 발화에서 가장 흔한 다른 표현이고, 원 논문은 영어 데이터만 다룸 |
| 비교 | 문장 인코더 2개: 논문의 영어 전용 `BAAI/bge-base-en-v1.5` vs 다국어 `BAAI/bge-m3` |
| 방식 | **단일 문장 벡터 + 코사인** (Step 1 Vanilla vCache와 같은 방식). 규칙 분할·RL 분할·조각 가중치(IDF/MLP)는 **쓰지 않음** (→ 3절) |
| 본 실험 | 오프라인 검색: 방언 질의 363개 → 표준어 문장 4,426개 중 제 짝을 1등으로 찾는가 |
| 보조 실행 | 같은 데이터로 vCache(δ=0.01)를 그대로 돌려 실제 재사용률 확인 |
| 데이터 | AI Hub 경상방언 AI 학습데이터(대화 15개). **git에 올리지 않고 Google Drive에서 읽음** |

## 1. 무엇에 대한 실험인가

### 1.1 상황
의미 캐시는 이전 요청과 뜻이 같은 새 요청이 오면 LLM을 다시 부르지 않고 저장된 답을 돌려준다. 이때 "뜻이 같다"는 판단은 **두 문장의 벡터가 얼마나 가까운가**로 한다.

예를 들어 캐시에 표준어 요청이 저장돼 있고, 다른 사용자가 같은 말을 방언으로 했다고 하자.

| | 문장 |
|---|---|
| 캐시에 있는 요청 (표준어) | 카레를 먹을 때 저는 약간 카레가 매운 거예요 **그러니까** |
| 새로 들어온 요청 (경상방언) | 카레를 먹을 때 저는 약간 카레가 매운 거예요 **긍까** |
| 캐시에 있는 다른 요청 (함정) | 같은 대화에서 나온, 주제·어휘가 비슷하지만 뜻이 다른 문장 수천 개 |

좋은 의미 캐시라면 방언 요청의 가장 가까운 캐시 항목이 **같은 뜻의 표준어 문장**이어야 하고, 그 유사도가 **뜻이 다른 문장들보다 확실히 높아야** 한 임계값으로 안전하게 재사용할 수 있다.

### 1.2 구체적으로 확인하는 것
1. **찾는가:** 방언 질의의 1등 캐시 항목이 제 짝(같은 뜻의 표준어 문장)인가 — top-1·top-5 정확도
2. **구분되는가:** 제 짝과의 유사도가 가장 비슷한 오답과의 유사도보다 높은가 — AUC
3. **안전하게 재사용할 수 있는가:** "유사도 ≥ t면 재사용" 규칙 하나로 운영할 때, 오답 재사용을 질의의 1%·5% 이하로 지키면서 몇 %나 재사용할 수 있는가 (vCache의 δ와 같은 기준)
4. **무엇이 어렵게 만드는가:** 방언 어절 비율이 높을수록, 방언형과 표준어형의 글자가 많이 다를수록 얼마나 떨어지는가
5. **인코더 차이:** 위 1~4가 영어 전용 인코더와 다국어 인코더에서 어떻게 다른가
6. **vCache를 그대로 쓰면:** 실제 vCache는 이 상황에서 몇 %나 재사용하는가, 못 한다면 왜인가

### 1.3 가설
1. 영어 인코더는 한국어를 의미가 아니라 하위 단어 조각의 **글자 겹침**으로 비교한다 → 글자 유사도가 낮거나 방언 비율이 높은 질의에서 크게 떨어진다.
2. 다국어 인코더는 그 하락이 작다.
3. 두 인코더 모두 같은 대화의 비슷한 문장(함정) 때문에 `auc_hard`가 `auc_random`보다 크게 낮다.
4. vCache는 의미 그룹당 문장이 2개뿐인 이 데이터에서 거의 재사용하지 못한다(관측 6개 규칙, 3.3절).

## 2. 데이터

- AI Hub **경상방언 AI 학습데이터** (솔트룩스, 2020). 대화 파일마다 발화별 `dialect_form`(방언 전사)·`standard_form`(표준어 변환)과 어절별 방언 여부(`eojeolList.isDialect`)가 있다.
- 현재 받은 분량: 대화 15개, 발화 4,470개 (라벨 JSON 9MB + 음성 WAV 586MB). **이 실험은 텍스트(라벨 JSON)만 쓴다.**
- ⚠️ **git에 올리지 않는다.** AI Hub 이용약관상 재배포가 제한되고, 음성·화자 정보(나이·성별·출생지 등)가 들어 있으며, 이 저장소는 공개 저장소다.
  `.gitignore`가 `mvr-cache/data/원천데이터/`, `mvr-cache/data/라벨링데이터/`, 가공 결과(`mvr-cache/data/dialect/`), `*.wav`를 막는다. **Colab에서는 Google Drive에 직접 올린 라벨 폴더를 읽는다.**

### 가공 (`benchmarks/dialect/prepare_dialect_dataset.py`)
- 정제: `{laughing}` 같은 태그 제거, `((x))`(불확실 전사) → `x`, `(())` 제거, `~`(장음 표시) 제거, 공백 정리
- **질의(`pairs.parquet`)**: 정제 후 방언형과 표준어형이 실제로 다른 발화. 띄어쓰기·문장부호만 다른 것은 제외하고, 표준어가 2어절 이상인 것만
- **캐시(`index.parquet`)**: 모든 발화의 표준어 문장(같은 문장은 하나로). 질의마다 정답 1개 + 같은 대화들에서 나온 비슷한 일상 문장들이 함정 후보
- **vCache 스트림(`stream.parquet`)**: 캐시 문장 전부(섞음) → 방언 질의(섞음). `id_set` = 표준어 문장 그룹 → `benchmark_id_set`으로 채점 (같은 그룹이면 정답)
- 현재 15개 대화 기준: 질의 **363쌍**, 캐시 **4,426문장**, 스트림 4,789행. 발화 4,470개 중 방언형≠표준어형은 8.2%뿐이고, 그마저 대부분 한두 단어 차이(쫌→조금, 그기→거기, 긍까→그러니까). 방언형-표준어형 글자 유사도 평균 0.94, 어절 중 방언 비율 평균 16%

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
| 캐시 | 표준어 문장 4,426개를 처음부터 다 넣어 둠 | 스트림 순서대로 들어오며, 놓치면 추가(cache-on-miss) |
| 후보 검색 | **전수 비교**(정확한 코사인, 근사 없음) | HNSW 근사 검색으로 **가장 가까운 1개**(`k=1`) |
| 재사용 판단 | 없음 (순위·유사도만 측정). 임계값 규칙은 사후 계산 | vCache verified 정책(δ=0.01): 항목마다 관측 (s, 정답여부)로 로지스틱 임계값을 학습해 탐색/재사용 결정 |
| 정답 기준 | 1등이 제 짝 문장인가 | 같은 `id_set`(같은 표준어 문장 그룹)인가 |

vCache는 항목마다 관측이 6개(사전값 2 + 실제 4) 쌓이기 전에는 무조건 탐색한다. 이 데이터는 의미 그룹당 문장이 2개(표준어 1 + 방언 1)뿐이라 어떤 항목도 6개를 채울 수 없어서, 인코더와 무관하게 재사용률이 0에 가깝게 나올 것으로 예상한다. 그래서 본 실험을 검색으로 두고, vCache 실행은 "가장 가까운 항목은 맞았는데(`nn_correct_rate`) 재사용은 못 한" 차이로 이 한계를 보여 주는 용도로 쓴다.

### 3.4 분할(규칙·RL)과 조각 가중치(IDF·MLP)를 쓰지 않는 이유
- **main 기준이라 규칙 분할기·조각 가중치 코드가 없다.** 이 기능들은 `gpu-production`에서 추가됐다(`RuleSplitter.py`, `build_segment_weight_stats.py`).
- **RL 분할기는 이 데이터에 맞지 않는다.** 가진 체크포인트는 영어 데이터(LmArena·Classification·SearchQueries)를 768차원 영어 인코더로 학습한 것이다. bge-m3(1024차원)와는 차원부터 맞지 않고, 한국어 분할 정책을 새로 학습하려면 이 데이터로 RL 학습을 다시 해야 한다.
- **먼저 기본 방식에서 인코더 차이를 확인하는 게 순서다.** 단일 벡터에서도 방언을 못 알아본다면 그건 분할 이전에 인코더 문제다. 분할·가중치 효과는 인코더를 정한 다음 단계로 둔다(7절).

### 3.5 비교 조건

| 조건 | 인코더 | 차원 | 비고 |
|---|---|---|---|
| A | `BAAI/bge-base-en-v1.5` | 768 | 원 논문·vCache 기본값. 영어 전용 |
| B | `BAAI/bge-m3` | 1024 | 다국어(한국어 포함). 약 2.3GB |

HNSW 차원은 첫 벡터로 정해지므로 1024차원도 코드 수정 없이 들어간다. 두 조건 모두 풀링(마스크 평균)·점수(코사인)·데이터·순서가 같고 인코더만 다르다.

## 4. 지표

### 본 실험 (`benchmarks/dialect/eval_dialect_retrieval.py` → `retrieval.json`)

| 지표 | 의미 |
|---|---|
| `top1_acc`, `top5_acc`, `mrr` | 제 짝(표준어 문장)이 1등 / 5등 안 / 역순위 평균 |
| `own_cos_mean`, `best_wrong_cos_mean`, `margin_mean` | 제 짝과의 코사인 / 가장 비슷한 오답과의 코사인 / 그 차이 |
| `auc_hard` | 제 짝의 코사인 vs 가장 비슷한 오답의 코사인 (함정 구분력) |
| `auc_random` | 제 짝 vs 무작위 캐시 문장 (쉬운 기준) |
| `static_threshold` | "코사인 ≥ t면 재사용" 규칙 하나로 운영할 때, 오답률(틀린 재사용 / 질의) ≤ 1%·5%에서의 최대 히트율과 그 t |
| `by_dialect_ratio` | 어절 중 방언 비율 구간(<0.1, 0.1~0.2, ≥0.2)별 top-1 |
| `by_surface_sim` | 방언형-표준어형 글자 유사도 구간(<0.9, 0.9~0.95, ≥0.95)별 top-1 |

질의별 상세(`retrieval_<모델>.csv`): 제 짝 코사인, 가장 비슷한 오답의 코사인과 **그 문장**, 순위 → 어떤 방언 표현에서 틀리는지 직접 볼 수 있다.

### 보조 실행 (`benchmarks/dialect/analyze_dialect_vcache.py`)
표준어 행 / 방언 질의별로 `hit_rate`(재사용), `error_rate`(틀린 재사용 / 행), `nn_correct_rate`(가장 가까운 캐시 항목이 정답이었던 비율 = 완벽한 재사용 판단으로 도달 가능한 상한).

### 예비 결과 (맥 CPU, bge-base-en, 질의 60개만 — 참고용)
top-1 0.90, top-5 0.95, `auc_hard` 0.885 (`auc_random` 0.9997). 방언 비율 ≥ 0.2 구간 top-1 0.67, 글자 유사도 < 0.9 구간 0.50 → 가설 1 방향. 전체 363개와 bge-m3 결과는 Colab에서.

## 5. Colab 실행

`gpu-production` 실험이 돌고 있는 런타임과 **같은 런타임에서 돌리지 말 것**(같은 이름의 패키지를 다른 폴더로 다시 설치하므로 돌던 실험의 코드가 바뀜). 이 실험은 GPU가 없어도 된다(CPU 런타임 가능, bge-m3 인코딩만 조금 느림).

**데이터 준비 (한 번):** 로컬의 `mvr-cache/data/라벨링데이터` 폴더를 Drive의 `MyDrive/dialect/라벨링데이터`로 올린다 (WAV는 필요 없음).

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

### 셀 2: 가공 → 검색 실험 → vCache 보조 실행
```python
%cd /content/MVR-Cache-dialect/mvr-cache
import os, torch
os.environ.update(HF_ENDPOINT="https://huggingface.co", HF_CACHE_BASE="/content/hf_cache", USE_TF="0")
D = "/content/drive/MyDrive/dialect"        # 라벨 폴더를 올린 위치
LABELS, PREP = f"{D}/라벨링데이터", f"{D}/prepared"
MODELS = ["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_VCACHE = True

!python benchmarks/dialect/prepare_dialect_dataset.py --label-dir "{LABELS}" --out-dir "{PREP}"
!python benchmarks/dialect/eval_dialect_retrieval.py --data-dir "{PREP}" --models {" ".join(MODELS)} --device {DEVICE} --out "{D}/retrieval.json"

if RUN_VCACHE:
    outs = []
    for m in MODELS:
        out = f"{D}/vcache_{m.split('/')[-1]}.json"
        if not os.path.exists(out):
            !python benchmarks/eval_sembenchmark_verified.py --dataset "{PREP}/stream.parquet" --delta 0.01 --sleep 0.02 --similarity-evaluator benchmark_id_set --device {DEVICE} --embedding-model {m} --output-json "{out}"
        outs.append(out)
    !python benchmarks/dialect/analyze_dialect_vcache.py --stream "{PREP}/stream.parquet" --results {" ".join(outs)}
```

### 결과 파일 (Drive `MyDrive/dialect/`)
- `prepared/` : 가공 데이터 + `manifest.json`(건수·정제 규칙)
- `retrieval.json` : 인코더별 요약 지표 / `retrieval_<모델>.csv` : 질의별 상세
- `vcache_<모델>.json` : vCache 실행 결과 (요청별 hit/tp/fp/fn 포함)

## 6. 변경한 파일 (main 대비)

브랜치 커밋: `e083ff6` → `ffa3a60` → `7ccb8bf`(`gpu-production`에서 가져온 실행용 수정 3개) → `8317a7b`(이 실험)

### 6.1 실행에 필요해서 `gpu-production`에서 가져온 수정 (원저자 코드 그대로는 Colab에서 설치·실행이 안 됨)

| 파일 | 변경 | 이유 |
|---|---|---|
| `mvr-cache/pyproject.toml` | `vllm`을 기본 의존성에서 빼고 선택 설치(`mvr-cache[vllm]`)로 옮김. `[tool.poetry] packages = [{ include = "vcache" }]` 추가 | vllm은 평가에 쓰이지 않는데 설치하면 Colab의 torch를 바꿔 버림. 배포 이름(`mvr-cache`)과 코드 폴더(`vcache/`)가 달라 `pip install -e`가 실패하던 문제 |
| `mvr-cache/vcache/inference_engine/strategies/vllm.py` | `from vllm import ...`를 `try/except`로 감싸고, vllm 없이 엔진을 만들면 설치 방법을 알려 주는 에러 | vllm 없이도 `import vcache`가 되게 |
| `mvr-cache/vcache/vcache_core/splitter/embedding_model.py` | 공유 토크나이저를 잠금 래퍼(`_LockedTokenizer`)로 감쌈 | vCache 백그라운드 갱신 스레드와 메인 스레드가 토크나이저를 동시에 쓰면 `Already borrowed` 에러로 후보가 빠지던 문제. 결과(토큰)는 동일 |
| `mvr-cache/vcache/vcache_core/cache/embedding_engine/strategies/lang_chain.py` | `HF_ENDPOINT`를 hf-mirror로 강제로 덮어쓰던 두 줄 삭제(`setdefault`만 남김) | 지정한 Hugging Face 주소가 무시되던 문제 |

### 6.2 이 실험을 위해 바꾼 것

| 파일 | 변경 | 내용 |
|---|---|---|
| `mvr-cache/benchmarks/eval_sembenchmark_verified.py` | 수정 (+7줄) | `--embedding-model` 옵션 추가 → `EmbeddingModel(model_name=...)`로 전달. 주지 않으면 기존과 같은 인코더(bge-base-en 또는 `BGE_MODEL_PATH`) |
| `mvr-cache/benchmarks/dialect/prepare_dialect_dataset.py` | 새 파일 | 라벨 JSON → 정제 → `index`/`pairs`/`stream.parquet` + `manifest.json` (2절) |
| `mvr-cache/benchmarks/dialect/eval_dialect_retrieval.py` | 새 파일 | 인코더별 오프라인 검색 평가 (4절 지표, 질의별 CSV) |
| `mvr-cache/benchmarks/dialect/analyze_dialect_vcache.py` | 새 파일 | vCache 결과를 표준어 행 / 방언 질의로 나눠 hit·error·`nn_correct_rate` 집계 |
| `.gitignore` | 수정 | 방언 원본·가공 데이터와 `*.wav`를 git에서 제외 |
| `DIALECT_EXPERIMENT.md` | 새 파일 | 이 문서 |

판정 규칙(vCache verified 정책), 캐시·HNSW·채점 코드는 손대지 않았다.

## 7. 한계·다음 단계
- 데이터가 대화 15개(질의 363쌍)뿐이라 구간별 수치는 표본이 작다. 대화를 더 받으면 같은 셀로 다시 돌리면 된다.
- 방언 차이가 대부분 한두 단어라 과제가 쉬운 편. 방언 비율이 높은 질의만 따로 모아 보거나, 다른 지역(전라·제주 등) 데이터로 넓힐 수 있다.
- 다국어 인코더의 풀링은 이 코드의 평균 풀링을 그대로 썼다(bge-m3 권장 방식은 [CLS]). vCache와 같은 방식으로 맞추기 위한 선택이며, 필요하면 비교해 볼 수 있다.
- vCache의 관측 6개 규칙 때문에 그룹이 작은 데이터에서는 재사용이 불가능하다 → 같은 표준어 문장에 여러 방언 화자가 있는 데이터처럼 같은 질문이 여러 번 반복돼야 vCache 실험이 의미 있다.
- 분할·가중치로 넓히려면: (1) `gpu-production`의 규칙 분할기·가중치 코드를 이 브랜치에 가져오고, (2) 한국어 문장부호·어절 기준 분할 규칙을 확인하고, (3) RL은 한국어 인코더로 분할 정책을 새로 학습해야 한다.
