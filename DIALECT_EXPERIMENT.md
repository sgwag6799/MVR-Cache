# 방언 ↔ 표준어 의미 캐시 실험 (`dialect-cache-experiment` 브랜치)

> **답변은 한국어로 해 주세요.** `main`(원저자 코드)에서 딴 브랜치이며, `gpu-production`의 가중치·분할·로그 실험과는 별개다.

## 1. 질문

의미 캐시는 "같은 뜻의 다른 표현"을 같은 요청으로 알아봐야 한다. 실제 사용자 발화의 대표적인 다른 표현이 **방언**이다.

- 경상방언 문장으로 질의했을 때, 캐시에 있는 **같은 뜻의 표준어 문장**을 찾는가?
- 논문의 영어 전용 인코더(`BAAI/bge-base-en-v1.5`)와 다국어 인코더(`BAAI/bge-m3`) 중 무엇이 나은가?
- 방언 비율이 높을수록(표면 글자가 많이 다를수록) 얼마나 나빠지는가?
- vCache를 그대로 얹으면 실제로 몇 %나 재사용하는가?

## 2. 데이터

- AI Hub **경상방언 AI 학습데이터** (솔트룩스, 2020). 대화 파일마다 발화별 `dialect_form`(방언 전사)·`standard_form`(표준어 변환)과 어절별 방언 여부(`eojeolList.isDialect`)가 있다.
- 현재 받은 분량: 대화 15개, 발화 4,470개 (라벨 JSON 9MB + 음성 WAV 586MB). **이 실험은 텍스트(라벨 JSON)만 쓴다.**
- ⚠️ **git에 올리지 않는다.** AI Hub 이용약관상 재배포가 제한되고, 음성·화자 정보(나이·성별·출생지 등)가 들어 있으며, 이 저장소는 공개 저장소다.
  `.gitignore`가 `mvr-cache/data/원천데이터/`, `mvr-cache/data/라벨링데이터/`, 가공 결과(`mvr-cache/data/dialect/`), `*.wav`를 막는다.
  **Colab에서는 Google Drive에 직접 올린 라벨 폴더를 읽는다.**

### 가공 (`benchmarks/dialect/prepare_dialect_dataset.py`)
- 정제: `{laughing}` 같은 태그 제거, `((x))`(불확실 전사) → `x`, `(())` 제거, `~` 제거, 공백 정리
- **질의(`pairs.parquet`)**: 정제 후 방언형과 표준어형이 실제로 다른 발화(띄어쓰기·문장부호만 다른 것은 제외), 표준어 2어절 이상
- **캐시(`index.parquet`)**: 모든 발화의 표준어 문장(같은 문장은 하나로). 질의의 정답 1개 + 같은 대화들에서 나온 비슷한 일상 문장들이 함정 후보
- **vCache 스트림(`stream.parquet`)**: 캐시 문장 전부(섞음) → 방언 질의(섞음). `id_set` = 표준어 문장 그룹 → `benchmark_id_set`으로 채점
- 현재 15개 대화 기준: 질의 **363쌍**, 캐시 **4,426문장**, 스트림 4,789행. 방언형-표준어형 글자 유사도 평균 0.94(대부분 한두 단어 차이: 쫌→조금, 그기→거기, 긍까→그러니까), 어절 중 방언 비율 평균 16%

## 3. 설계

### 본 실험: 오프라인 검색 (`benchmarks/dialect/eval_dialect_retrieval.py`)
vCache와 같은 문장 벡터(마스크 평균 풀링, `EmbeddingModel`)·코사인으로, 방언 질의마다 캐시 4,426문장 중 가장 가까운 것을 찾는다.

| 지표 | 의미 |
|---|---|
| `top1_acc`, `top5_acc`, `mrr` | 제 짝(표준어 문장)이 1등 / 5등 안 / 역순위 평균 |
| `auc_hard` | 제 짝의 코사인 vs 가장 비슷한 오답의 코사인 (함정 후보 구분력) |
| `auc_random` | 제 짝 vs 무작위 문장 (쉬운 기준) |
| `static_threshold` | "코사인 ≥ t면 재사용" 규칙 하나로 캐시를 운영할 때, 오답률(틀린 재사용 / 질의) ≤ 1%·5%에서 도달 가능한 최대 히트율과 그 t |
| `by_dialect_ratio`, `by_surface_sim` | 방언 비율 / 글자 유사도 구간별 top-1 정확도 |

### 보조 실행: vCache (`eval_sembenchmark_verified.py --embedding-model ...`)
같은 스트림에 vCache(δ=0.01, sleep 0.02)를 그대로 돌리고 `analyze_dialect_vcache.py`로 표준어 행·방언 질의를 나눠 본다.
- **예상: 히트율 ≈ 0.** vCache는 캐시 항목마다 관측 6개(사전값 2 + 실제 4)가 쌓이기 전에는 무조건 탐색하는데(HANDOFF 3절 요인 3), 이 데이터는 의미 그룹당 문장이 2개뿐이라 어느 항목도 6개를 채울 수 없다.
- 그래서 본 실험을 검색으로 두고, vCache 실행은 "가장 가까운 항목은 맞았는데(`nn_correct_rate`) 재사용은 못 한" 차이로 이 한계를 수치로 보여 주는 용도로 쓴다.

### 비교 조건
| | 인코더 | 차원 |
|---|---|---|
| A | `BAAI/bge-base-en-v1.5` (논문, 영어 전용) | 768 |
| B | `BAAI/bge-m3` (다국어) | 1024 |

HNSW 차원은 첫 벡터로 정해지므로 1024차원도 코드 수정 없이 들어간다. RL 분할기(Step 4)는 768차원 영어 인코더로 학습된 체크포인트가 필요해 이 실험에서는 쓰지 않는다.

### 가설
1. 영어 인코더는 한국어를 하위 단어 조각으로만 보고 **글자 겹침에 의존** → 글자 유사도가 낮거나 방언 비율이 높은 질의에서 크게 떨어진다.
2. 다국어 인코더는 그 하락이 작다.
3. 두 인코더 모두 일상 대화 함정 후보 때문에 `auc_hard`가 `auc_random`보다 크게 낮다.

### 예비 결과 (맥 CPU, bge-base-en, 질의 60개만 — 참고용)
top-1 0.90, top-5 0.95, `auc_hard` 0.885 (`auc_random` 0.9997). 방언 비율 ≥ 0.2 구간 top-1 0.67, 글자 유사도 < 0.9 구간 0.50 → 가설 1 방향. 전체 363개·bge-m3 결과는 Colab에서.

## 4. Colab 실행

`gpu-production` 실험이 돌고 있는 런타임과 **같은 런타임에서 돌리지 말 것**(같은 이름의 패키지를 다른 폴더로 다시 설치하므로 돌던 실험의 코드가 바뀜).
이 실험은 GPU가 없어도 된다(CPU 런타임 가능, bge-m3 인코딩만 조금 느림).

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
- `retrieval.json` : 인코더별 요약 지표 / `retrieval_<모델>.csv` : 질의별 제 짝 코사인·가장 비슷한 오답과 그 문장·순위
- `vcache_<모델>.json` : vCache 실행 결과 (요청별 hit/tp/fp/fn 포함)

## 5. 브랜치 구성
- `main` + 실행에 필요한 수정 3개를 `gpu-production`에서 가져옴: 설치 수정(vllm 선택 설치·`[tool.poetry] packages`), 토크나이저 잠금(`Already borrowed`), `HF_ENDPOINT` 덮어쓰기 수정
- `eval_sembenchmark_verified.py`에 `--embedding-model` 추가 (주지 않으면 기존과 동일)
- 새 파일: `benchmarks/dialect/` 아래 가공·검색 평가·vCache 집계 스크립트, 이 문서, `.gitignore` 규칙

## 6. 한계·다음 단계
- 데이터가 대화 15개(질의 363쌍)뿐이라 구간별 수치는 표본이 작다. 대화를 더 받으면 같은 셀로 다시 돌리면 된다.
- 방언 차이가 대부분 한두 단어라 과제가 쉬운 편. 방언 비율이 높은 질의만 따로 모아 보거나, 다른 지역(전라·제주 등) 데이터로 넓힐 수 있다.
- 다국어 인코더의 풀링은 이 코드의 평균 풀링을 그대로 썼다(bge-m3 권장 방식은 [CLS]). vCache와 같은 방식으로 맞추기 위한 선택이며, 필요하면 비교해 볼 수 있다.
- vCache의 관측 6개 규칙 때문에 그룹이 작은 데이터에서는 재사용이 불가능하다 → 같은 질문이 여러 번 반복되는 데이터(같은 표준어 문장에 여러 방언 화자)가 있어야 vCache 실험이 의미 있다.
