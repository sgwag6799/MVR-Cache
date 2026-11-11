# `gpu-production` 브랜치 노트

main 대비 이 브랜치에서 추가한 문서를 이 파일 하나로 합쳤다. 위쪽이 이 브랜치에서 쓴 문서, 아래쪽이 부모 브랜치(`main`)에서 물려받은 문서다.
각 부분의 제목 앞 이름이 원래 파일 이름이고, 절 번호는 원래 문서 그대로다 (코드 주석의 "X 부분 N절"이 이 번호를 가리킨다).

| 부분 | 원래 파일 |
|---|---|
| HANDOFF | `HANDOFF.md` |
| logging-decisions | `.claude/logging-decisions.md` |

---

<!-- 원래 파일: HANDOFF.md -->

# HANDOFF 부분: MVR-Cache 인수인계 (2026-10-08)

> **답변은 한국어로 해 주세요.** 이 문서는 맥북 로컬 세션의 대화 내용을 이어받기 위한 요약입니다.

## 0. 한눈에 보기

- 목표: MVR-cache 논문(PKU-SDS-lab/MVR-Cache, 기준 커밋 `8cfaa0b`) 재현 → 가중치·분할 개선 실험
- 브랜치
  - `main`: 원저자 코드 그대로
  - `cpu-reproduction`: 맥북 CPU 재현용 (환경 설정, tools/, 예전 데이터·분석 스크립트)
  - **`gpu-production`**: 원본 + 가중치·분할·로그 코드만 (Colab GPU로 팀 공유, 지금 작업 브랜치)
- 실험 실행은 **Colab GPU**(T4, Python 3.11 런타임)에서. 맥북 CPU로는 모델 실행 안 함 (`.venv`도 삭제함)
- 결정 대기 목록: logging-decisions 부분 (3·4번 결정·구현 완료, 1·2·6번 대기)
- 현재 단계 (2026-10-08): Colab 설치·300개 시범 실행 완료 → 로그 수정분 재확인 시범 실행 → 가중치 통계 → 본 실행

## 1. gpu-production 브랜치 구성

| 커밋 | 내용 |
|---|---|
| `61a6839` | 규칙 분할기 + 조각 가중치(uniform/length/idf/centroid/mlp), 가중 MaxSim, 평가 옵션, 분석 스크립트, `data/classification.parquet` |
| `e9259fc` | Classification 분할 `train/train3k.parquet`, `val/val.parquet`, `test/test41k.parquet` + 각 `manifest.json` |
| `18e97bf` → `a7d419c` → `b890de1` | 규칙 분할기: 앞 4개만 자르던 문제 수정 → 구두점마다 자르기 → 예외 규칙 1~6 |
| `d5d1d69` | `build_segment_weight_stats.py` 기본값을 "파일 전체"로 (예전 기본 10000~12999행은 train3k에서 0행) |
| `d8bfd55` | 실험 로그 A~F (`--run-log-dir`) |
| `c95245f` | 결정 대기 목록 logging-decisions 부분 |
| `ab30879` | 이 인수인계 문서 |
| `8ab7d35` → `1b49502` | Python 3.13·numpy 2 허용 → 되돌림 (Colab을 3.11로 맞춰서 main과 같은 범위 유지) |
| `d634437` | **설치 수정**: vllm을 선택 설치로(`mvr-cache[vllm]`), `vllm.py` import 예외 처리, `[tool.poetry] packages` 추가 (원저자 코드는 `pip install -e`가 실패하고, vllm이 Colab torch를 교체해 깨뜨림) |
| `be91154` | **토크나이저 잠금**: 백그라운드 갱신 스레드와 메인 스레드가 토크나이저를 동시에 써서 나던 `Already borrowed` 에러(후보가 점수 계산에서 빠짐) 수정 |
| `a87eef2` | `lang_chain.py`가 `HF_ENDPOINT`를 hf-mirror로 강제로 덮어쓰던 것 수정 (`setdefault`만 남김) |
| `d4d169b` | train을 팀원 파일 `train/train.parquet`로 교체 (이전 train3k와 행·내용 동일, 순서만 다름), `manifest.json` 재생성 |
| `21e0fd6` | 로그 추가: `updates.jsonl`, `alpha_tprime`, 진단 시간 분리(`timing_ms.diag`) — logging-decisions 3·4번 |

### 주요 파일
- `mvr-cache/vcache/vcache_core/splitter/RuleSplitter.py`: 규칙 분할기, 가중치 계산, `describe_split`(로그용)
- `mvr-cache/vcache/vcache_core/splitter/punctuation_rules.py`: 자를 위치 규칙 1~6
- `mvr-cache/vcache/vcache_policy/strategies/verified_splitter.py`: 가중 MaxSim, 상세 로그 수집(`detail_log`), `_LoggedAlgorithm`(판정 규칙은 원본과 동일, 값만 기록 + `--seed`)
- `mvr-cache/benchmarks/eval_sembenchmark_verified_splitter.py`: 평가 실행
- `mvr-cache/benchmarks/build_segment_weight_stats.py`: IDF·centroid·MLP 통계 생성 + 학습 로그(E)
- `mvr-cache/benchmarks/run_log.py`: 로그 A~D + `updates.jsonl` 기록 / `analyze_run_log.py`: 사후 집계 F
- `mvr-cache/vcache/vcache_core/splitter/embedding_model.py`: 공유 BGE 모델·토크나이저 (`_LockedTokenizer`로 스레드 간 동시 사용 방지)

## 2. 확정된 결정

### 데이터 분할 (Classification 45k)
- train 3k / val 1k / test 41k. test·val은 팀원 파일(cls_seed42). train은 처음엔 45k − test − val을 **(dataset_name, id)**로 매칭해 복원(`train3k.parquet`, `id`만으로는 과제끼리 겹침)했고, 2026-10-08 팀원이 준 `train/train.parquet`로 교체 — 같은 3,000행·같은 내용, 순서만 섞임. IDF·centroid·중복 표시는 순서 무관, MLP·RL 학습은 입력 순서 차이로 미세하게 달라질 수 있음
- train 안에 val/test와 문장이 같은 행 685개 (manifest `leak_rows`, 제거하지 않음)
- manifest 지문 = `"dataset_name:id"`를 쉼표로 이어 붙인 문자열의 sha1 앞 12자리 (팀원 방식과 동일)
- test41k는 **원래 순서가 아니라 섞인 순서**(seed 42). test 안에 train/val과 문장이 같은 행 1,534개(거의 상품 분류 중복) 포함

### 분할 규칙 (규칙 분할기)
- 구두점(`, . ! ? : ;` + 전각 `， 。 ！ ？ ： ；`)마다 자름. `--splitter-max-segments 0`만 허용(= 분할 없음 대조군)
- 자르지 않는 예외 (`punctuation_rules.py`, 원문 글자 위치로 판단)
  1. 숫자 안 (3.5, 1,000, 10:30, v2.1.3)
  2. 약어·호칭 (e.g., Mr., Ph.D., U.S., a.m., No. 5)
  3. URL·이메일·파일명
  4. 연속 구두점은 한 번만 (..., !!!, ?!)
  5. 이모티콘 (:) ;) :-( :D ^^;)
  6. 빈 조각 없음 (문장 끝 마침표, 맨 앞 목록 번호 "1.", 구두점만 있는 조각)
- 7번(고유명사 속 구두점)은 한계로 둠, 8번(짧은 조각 합치기)은 구현 안 함
- `3 : 40`처럼 띄어 쓴 시간은 잘림 → 그대로 두기로 함

### 점수 공식
- 실제 평가(`multivector_top_k` 등 캐시된 조각 경로)는 **순수 가중 MaxSim**: `s = ((row + col)/2 + 1)/2` (전체 문장 벡터는 MaxSim 안의 한 행, 가중치 = 조각 가중치 평균)
- `--mix-fullcos`(전체 코사인 50% 혼합)는 이 경로에서 적용되지 않음 (원본 코드도 동일)
- MLP 학습(`pair_scores`)도 순수 가중 MaxSim이 기본 (옛 방식은 `--mix-fullcos`), val AUC 최고 에폭을 저장

## 3. 주요 발견 (예전 맥북 실험 기준: 45k 원래 순서, 옛 분할기)

- **유사도 점수 s가 정답을 구분 못 함**: 정답/오답 쌍 s 평균 0.919 vs 0.918, AUC ≈ 0.50. 정규분포도 아님
- 과제별: 상품 분류는 유사도가 작동(이웃 정답률 83%, 히트 32%), 아마존 감성(yes/no)·상식 질문은 우연 수준
- **히트율을 깎는 요소** (main 코드 `verified.py`의 `_Algorithm` 시뮬레이션 + 실제 로그)
  1. 오답 관측 1개면 그 항목은 사실상 죽음 (t̂가 0.8 이상으로 뛰어 재사용 확률 ≈ 2%) — 오답 있는 항목 재사용 5.0% vs 없는 항목 55.9%
  2. 정답만 쌓여도 재사용 확률 ≈ 50% 상한 (`variance_map` 고정값, 관측 48개 이상 표준편차 0.124)
  3. 관측 6개(사전값 2 + 실제 4) 전 무조건 탐색 — 요청의 38.9%
  4. cache-on-miss라 오답마다 새 항목 → 관측 분산, 6개 도달 항목 41%
  5. 1등 이웃 하나만 판정·갱신 (`k=1`, 갱신도 1등 항목만 — 코드 확인함)
  6. 히트 시에는 관측이 쌓이지 않음
  7. 근본 원인: s의 구분력 없음 (s 하나로 정확도 99% 유지 시 최대 히트 2.1%)
- 팀원의 천장 실험 결론(가중치 효과 없음)과 일치
- **상품 분류의 29.7%가 완전 반복 프롬프트**(이전 답과 99.9% 일치)인데 예전 실행에서 히트 38%뿐 → "완전 일치 계층" 기회 (추정 최대 +12%p, 오류 ≈ 0.1%)

## 4. 검토 중인 아이디어 (아직 구현 안 함)

| 아이디어 | 현재 판단 |
|---|---|
| 완전 일치 계층 (같은 프롬프트·s≈1이면 바로 재사용) | 효과가 가장 클 것으로 예상. 보고 시 "완전 반복 / 새 프롬프트" 분리 필요 |
| 비대칭 점수 (row만 / col만 / min) | 상품 분류에서는 "후보 → 질문" 방향이 유리할 수 있음. 로그에 후보별 row/col 점수 추가하면 재실행 없이 비교 가능 |
| 트리(계층) 캐시 | 관측 분산·항목 죽음 문제를 직접 겨냥. 상품 분류 한정이면 분류기에 가까워짐 → 팀 판단 필요 |
| 후보 K개 전부에 (s, c) 기록 | 6개 규칙·분산 완화(히트↑) vs 오답 관측 증가로 항목 죽음(히트↓), 선택 편향으로 오류율 위험. 옵션으로 만들어 비교 제안 |
| 캐시 삽입 규칙 "후보 중 정답 있으면 추가 안 함" | 관측 집중 vs 그 프롬프트가 캐시에 안 남아 완전 반복 손해 → 완전 일치 계층과 함께 쓰면 보완 |
| 숫자·고유명사 슬롯 템플릿 캐시 | Classification에는 효과 없음 확인 (슬롯화해도 추가 일치 ≤ 1.4%, 답이 라벨이라 채울 슬롯 없음) |

## 5. 결정할 사항

logging-decisions 부분 참고. 현황 (2026-10-08):

| # | 항목 | 상태 |
|---|---|---|
| 1 | 후보 방식 | **결정 대기.** ★안 `top_k --candidate-k 20 --use-cached-candidate-segments`로 실행 중 (논문 README는 `multivector_top_k --candidate-k 10`) |
| 2 | "코사인" 정의 | 결정 대기 (★ HNSW 저장 벡터 코사인 유지) |
| 3 | vCache 판정 로그 | **완료** — `updates.jsonl`, `alpha_tprime` (`21e0fd6`) |
| 4 | C 진단 | **완료** — 비율 0.01(실행 명령), 진단 시간 `timing_ms.diag`로 분리 (`21e0fd6`) |
| 5 | 용량 | 이번엔 적용 안 함 (회당 ≈ 150MB) |
| 6 | run.json 칸 | 결정 대기 (★ `min_tokens` 삭제, `max_splits` 표기) |

**문서에 아직 결정으로 기록되지 않은 차이 — 팀 확인 필요**
- `--sleep`: 논문 README 예시는 vCache 기준선 0.02, MVR-cache **0.1**. 우리는 조건 간 공정 비교를 위해 **모든 조건 0.02**
- `--mix-fullcos`: 논문 README 예시엔 있지만, 캐시된 조각 경로에서는 적용되지 않으므로(2절) 빼고 실행
- 정답 판정: test41k에 `ID_Set`이 없어 `--similarity-evaluator string --llm-col response_llama_3_8b` (README는 라벨이 있으면 `benchmark_id_set` 권장)

## 6. Colab 실행 방법 (GPU, Python 3.11 런타임)

셀 두 개를 그대로 복사해서 쓴다. 결과는 Drive(`/content/drive/MyDrive/mvr_runs`)에 저장.
**세션이 끊기면 셀 1 → 셀 2를 다시 실행**하면 된다. 끝난 단계는 건너뛰고, 중간에 끊긴 실행은 지우고 처음부터 다시 돈다.
코드가 바뀌어도 셀 1이 `git pull`을 하므로 다시 clone할 필요 없음 (`pip install -e`라 바로 반영).

### 셀 1: 설치 (런타임에 새로 연결할 때마다 한 번, 끝나면 자동 재시작)

```python
# ===== 셀 1: 설치 — 끝나면 런타임이 자동 재시작됨 ("세션 종료" 알림은 정상) =====
!python --version; nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os
if os.path.exists("/content/MVR-Cache"):
    !git -C /content/MVR-Cache pull
else:
    !git clone -b gpu-production https://github.com/sgwag6799/MVR-Cache.git /content/MVR-Cache
!git -C /content/MVR-Cache log --oneline -1
%cd /content/MVR-Cache
!pip install -q --ignore-installed packaging
!pip install -q --prefer-binary -e mvr-cache "datasets<4" tensorboard
!pip install -q --prefer-binary -e mvr-cache/vcache/vcache_core/cache/embedding_store/hnswlib
os.kill(os.getpid(), 9)
```

### 셀 2: 실험 전체 (재시작 후 실행)

```python
# ===== 셀 2: 실험 전체 — 셀 1의 재시작이 끝난 뒤 실행 =====
# 순서: 설치 확인 → 가중치 통계 → (시범 실행) → [회차마다] Step1 + 대조군·Step2·3a·3b → RL 학습 → Step4·5 → 집계
%cd /content/MVR-Cache/mvr-cache
import os, json, glob, shutil

os.environ.update(HF_ENDPOINT="https://huggingface.co", HF_CACHE_BASE="/content/hf_cache", USE_TF="0")
OUT = "/content/drive/MyDrive/mvr_runs"; os.makedirs(OUT, exist_ok=True)

# ---------------- 설정 ----------------
REPEATS   = (1, 2, 3)
RUN_SMOKE = True    # 새 로그 코드 확인용 300개 시범 실행 (약 1분). 확인 끝나면 False
RUN_RL    = True    # Step 4·5 (RL 분할기 학습 + 평가)
RL_EPOCHS = 200     # 세션이 짧으면 줄일 것 (RL 학습은 이어서 할 수 없음)

COMMON = ("--dataset test/test41k.parquet --delta 0.01 --sleep 0.02 "
          "--similarity-evaluator string --llm-col response_llama_3_8b "
          "--candidate-selection top_k --candidate-k 20 --use-cached-candidate-segments "
          "--include-full-embedding --splitter-device cuda "
          "--diag-frac 0.01 --train-file train/train.parquet")
WEIGHTS = f"{OUT}/weights_train.pt"
W = f"--segment-weight-stats {WEIGHTS}"
CKPT = f"{OUT}/ckpt_cls_train"
CONDS = {
    "control":     "--splitter-mode rule --splitter-max-segments 0",
    "step2":       "--splitter-mode rule --segment-weighting uniform",
    "3a_idf":      f"--splitter-mode rule --segment-weighting idf {W}",
    "3a_length":   "--splitter-mode rule --segment-weighting length",
    "3a_centroid": f"--splitter-mode rule --segment-weighting centroid {W}",
    "3b_mlp":      f"--splitter-mode rule --segment-weighting mlp {W}",
}

# ---------------- 도우미 ----------------
ip = get_ipython()
def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    ip.system(cmd)
    return ip.user_ns.get("_exit_code")

def done(run_dir):
    """끝까지 완료된 평가만 True (run.json은 시작할 때 생기고, 끝나면 finished_at이 추가됨)"""
    p = f"{run_dir}/run.json"
    return os.path.exists(p) and "finished_at" in json.load(open(p))

def evaluate(name, opts, r, extra=""):
    d = f"{OUT}/{name}_run{r}"
    if done(d):
        print(f"skip {name}_run{r} (완료됨)"); return
    shutil.rmtree(d, ignore_errors=True)          # 중간에 끊긴 흔적은 지우고 처음부터
    sh(f"python benchmarks/eval_sembenchmark_verified_splitter.py {COMMON} {opts} {extra} "
       f"--run-log-dir {d} --run-id {name}_run{r} --condition {name} --repeat {r} --seed {r} "
       f"--output-json {OUT}/{name}_run{r}.json")
    if not done(d):
        raise RuntimeError(f"{name}_run{r} 이(가) 끝까지 실행되지 않음 — 위 출력의 에러 확인")

def step1(r):
    out = f"{OUT}/step1_vcache_run{r}.json"   # 베이스라인 결과 JSON은 실행이 끝나야 생김
    if os.path.exists(out):
        print(f"skip step1_vcache_run{r} (완료됨)"); return
    sh(f"python benchmarks/eval_sembenchmark_verified.py --dataset test/test41k.parquet "
       f"--delta 0.01 --sleep 0.02 --similarity-evaluator string --llm-col response_llama_3_8b "
       f"--device cuda --output-json {out}")
    if not os.path.exists(out):
        raise RuntimeError(f"step1_vcache_run{r} 실패 — 위 출력의 에러 확인")

# ---------------- 0) 설치 확인 ----------------
code = sh('''python -c "import torch, hnswlib, vcache; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| multivector hnswlib', hasattr(hnswlib.Index, 'knn_query_with_parent'))"''')
if code not in (0, None):
    raise RuntimeError("설치 확인 실패 — 셀 1부터 다시")

# ---------------- 1) 가중치 통계 (3a·3b·5용) ----------------
if not os.path.exists(WEIGHTS):
    sh(f"python benchmarks/build_segment_weight_stats.py --dataset train/train.parquet "
       f"--response-col response_llama_3_8b --device cuda --out {WEIGHTS}")
    if not os.path.exists(WEIGHTS):
        raise RuntimeError("가중치 통계 저장 실패 — 'saved ->' 가 나올 때까지 기다려야 함")

# ---------------- 2) 시범 실행 (IDF, 300개) ----------------
if RUN_SMOKE:
    d = f"{OUT}/smoke_idf"
    if not done(d):
        shutil.rmtree(d, ignore_errors=True)
        sh(f"python benchmarks/eval_sembenchmark_verified_splitter.py {COMMON} --max-samples 300 "
           f"--splitter-mode rule --segment-weighting idf {W} "
           f"--run-log-dir {d} --run-id smoke_idf --condition smoke_idf --repeat 1 --seed 1")
        if not done(d):
            raise RuntimeError("시범 실행 실패 — 위 출력의 에러 확인")
    sh(f"python benchmarks/analyze_run_log.py {d}")
    reqs = [json.loads(l) for l in open(f"{d}/requests.jsonl")]
    ups = [json.loads(l) for l in open(f"{d}/updates.jsonl")]
    print("updates.jsonl:", len(ups), "줄 | 첫 줄:", ups[:1])
    print("timing_ms.diag 있는 요청:", sum("diag" in r["timing_ms"] for r in reqs), "/", len(reqs))
    print("alpha_tprime 기록된 요청:", sum((r.get("decision") or {}).get("alpha_tprime") is not None for r in reqs))

# ---------------- 3) 본 실행: 회차마다 모든 조건 (1회차부터 전부 채움) ----------------
for r in REPEATS:
    step1(r)
    for name, opts in CONDS.items():
        evaluate(name, opts, r)

# ---------------- 4) RL 분할기 학습 → Step 4·5 ----------------
if RUN_RL:
    if not os.path.exists(f"{CKPT}/DONE"):
        shutil.rmtree(CKPT, ignore_errors=True)   # 끊긴 학습은 이어서 못 하므로 처음부터
        code = sh(f"cd ../rl-training-algorithm && python RL4COTrainer.py --gpu_id 0 "
                  f"--train_parquet ../mvr-cache/train/train.parquet --val_parquet ../mvr-cache/val/val.parquet "
                  f"--parquet_text_column prompt --label_mode string --response_column response_llama_3_8b "
                  f"--train_sampling_mode anchor_nn --nn_warmup_epochs 5 --nn_candidate_topk 10 "
                  f"--batch_size 8 --accumulate_grad_batches 2 --lr 1e-4 --max_epochs {RL_EPOCHS} "
                  f"--check_val_every_n_epoch 5 --policy_mode separate --punctuation_only --bce_auto_balance "
                  f"--precompute_token_embeddings --save_weights_only --seed 0 --checkpoint_dir {CKPT}")
        if code not in (0, None) or not glob.glob(f"{CKPT}/*.ckpt"):
            raise RuntimeError("RL 학습 실패 — 위 출력의 에러 확인")
        open(f"{CKPT}/DONE", "w").write(f"max_epochs={RL_EPOCHS}\n")
    CK = f"--splitter-mode rl --splitter-checkpoint {CKPT} --splitter-max-segments 4"
    for r in REPEATS:
        evaluate("step4_rl", CK, r)
        evaluate("step5_rlmlp", f"{CK} --segment-weighting mlp {W}", r)

# ---------------- 5) 집계: 조건 × 과제 ----------------
dirs = sorted(d for d in glob.glob(f"{OUT}/*_run*") if done(d))
sh(f"python benchmarks/analyze_run_log.py {' '.join(dirs)} --out {OUT}/summary.csv")
print("모든 실험 완료. 요약:", f"{OUT}/summary.csv")
```

### 참고
- `google-colab requires ...` 같은 의존성 충돌 경고는 Colab 기본 패키지 얘기라 무시
- vllm은 설치하지 말 것(평가에 안 쓰이고 Colab torch를 교체함). 필요하면 `mvr-cache[vllm]`
- 가중치 통계에 `--mix-fullcos`를 붙이지 말 것 — 붙이면 MLP가 평가와 다른 옛 점수식으로 학습됨
- 가중치 통계는 학습이 끝난 뒤(`selected epoch ...`) 최종 AUC 비교에 시간이 더 걸린다. `saved -> ...`가 나오기 전에 중단하면 파일이 저장되지 않음
- 이어하기 판단: 평가는 `run.json`의 `finished_at`(시작할 때 `run.json`이 먼저 생기므로 파일 존재만으로는 판단하면 안 됨), Step 1은 결과 JSON, RL 학습은 `DONE` 파일
- Step 4·5의 `--splitter-checkpoint`에 폴더를 주면 그 안의 가장 최근 `epoch=*-step=*.ckpt`를 자동 선택
- 처리량 참고: T4에서 300개 시범 실행 약 32초(초당 9~13개)

## 7. 주의사항

- **로그 코드 실제 검증**: 2026-10-08 Colab T4 300개 시범 실행(균등 조건)에서 A~D·F 기록·집계 동작 확인. `21e0fd6` 로그 추가분과 가중치 칸은 6-4 시범 실행으로 다시 확인할 것
- 그 시범 실행 로그에서 발견해 고친 것: 토크나이저 동시 사용 `Already borrowed`(`be91154`), `HF_ENDPOINT` 강제 미러(`a87eef2`)
- Colab 무료 세션은 최대 약 12시간 → 본 실행 27회(회당 약 1시간 추정)는 여러 세션에 나눠 돌리고, 끊기면 6절 셀 1 → 셀 2를 다시 실행(끝난 실행은 건너뜀). RL 학습(200 epoch)은 `--save_weights_only`라 이어서 학습 불가 → 시간이 부족하면 epoch를 줄이거나 긴 세션 사용
- 기계가 다르면(맥북·맥미니·T4) 비동기 갱신 속도가 달라 절대값 비교 금지, 같은 기계 결과끼리만 비교
- `results/`, `*.pt`, 체크포인트는 `.gitignore` 대상 → 통계 파일 공유는 Google Drive 또는 `git add -f`
- `requests.jsonl`은 실행 1회 ≈ 150MB (27회 ≈ 4GB, Drive 여유 확인), `updates.jsonl`은 수 MB
- 예전 결과 수치(위 3절)는 옛 분할기·45k 원래 순서 기준이라 새 test41k에서는 달라질 수 있음
- 이 저장소(sgwag6799/MVR-Cache)는 공개 저장소

---

<!-- 원래 파일: .claude/logging-decisions.md -->

# logging-decisions 부분: 실험 로그 — 결정할 사항

브랜치: `gpu-production` (커밋 `d8bfd55` 기준으로 작성, 2026-10-08 결정 반영)
관련 코드: `mvr-cache/benchmarks/run_log.py`, `mvr-cache/benchmarks/analyze_run_log.py`,
`mvr-cache/vcache/vcache_policy/strategies/verified_splitter.py`

각 항목: 현재 동작 → 문제/주의 → 선택지 (추천에 ★, 결정된 항목은 [x])

**결정 현황 (2026-10-08)**
- 결정·구현 완료: 3번(`updates.jsonl`, `alpha_tprime`), 4번(진단 시간 분리)
- 이번에 적용 안 함: 5번(용량 줄이기) — 회당 약 150MB 그대로
- 아직 결정 대기: 1번(후보 방식), 2번(코사인 정의), 6번(run.json 칸). 현재 Colab 실행 명령은 1번 ★안(`top_k --candidate-k 20 --use-cached-candidate-segments`)으로 돌리고 있음

---

## 1. 후보 실행 방식 (K=20)

- 현재: `--candidate-selection`, `--candidate-k`는 실행 명령에서 지정. 기본값은 바꾸지 않음.
- 문서 정의 "HNSW 문장 코사인 Top-20"과 맞추려면 `top_k --candidate-k 20 --use-cached-candidate-segments`.
- `multivector_top_k`로 돌리면 후보가 조각 벡터로 뽑히고 개수도 20개를 넘을 수 있음 → "코사인 1등"이 재순위 전 1등이 아니게 됨.
- `--use-cached-candidate-segments`를 빼면 균등 SMaxSim, 조각 매칭(B6)이 기록되지 않고 C 진단이 매우 느려짐.

선택지
- [ ] ★ 모든 조건을 `top_k --candidate-k 20 --use-cached-candidate-segments`로 고정
- [ ] 기존 `multivector_top_k --candidate-k 10` 유지 (문서의 "후보" 정의를 바꿔 기록)

## 2. "코사인" 정의

- 현재: HNSW에 저장된 문장 벡터([CLS]·[SEP] 포함 토큰 평균)의 코사인 = HNSW가 후보를 고르는 점수와 동일.
- SMaxSim 안의 "전체 문장 행"은 [CLS]를 뺀 평균이라 값이 미세하게 다름.

선택지
- [ ] ★ 현재대로 (HNSW 점수 = 로그의 cos)
- [ ] SMaxSim의 전체 문장 행 코사인으로 변경

## 3. vCache 판정 로그 (B7)

- 현재: t̂·γ는 판정할 때마다 그 항목의 관측 전체로 다시 적합됨. 백그라운드는 관측 추가만 함.
  - `t_hat_before/gamma_before` = 그 항목의 직전 판정 때 값, `t_hat/gamma` = 이번 판정에서 적합한 값
  - `t_prime_before/t_prime` = 신뢰구간 보정 임계값, `alpha` = σ(γ(s − t̂)) (보정 전 t̂ 기준)
  - `n_obs`는 가짜 사전값 2개 포함
- 이번 요청의 (s, c)가 추가된 직후의 상태는 기록되지 않음 (그 항목의 다음 판정에서 보임).

선택지
- [x] 관측 추가 시점 로그 `updates.jsonl` 추가 (요청 번호, 항목 id, s, c, 추가 전후 관측 수)
- [x] α를 t′ 기준으로도 함께 기록 (`alpha_tprime`)
- [ ] 현재대로

구현 (2026-10-08)
- `updates.jsonl`: 백그라운드가 관측 (s, c)를 항목에 추가할 때마다 한 줄. 필드 `order`(관측을 만든 요청 번호 = requests.jsonl의 order),
  `applied_at_order`(실제로 반영될 때 처리 중이던 요청 번호 → 비동기 지연 확인용), `entry_id`, `s`(저장값과 같은 소수 3자리),
  `c`, `n_obs_before`/`n_obs_after`(사전값 2개 포함), `inserted_entry_id`(오답이라 새로 캐시에 넣은 항목, 없으면 null)
- `decision.alpha_tprime` = σ(γ(s − t′)). `alpha`(t̂ 기준)는 그대로 둠. 판정 규칙은 바뀌지 않음

## 4. C 샘플 진단

- 현재: 샘플마다 캐시 전체(끝으로 갈수록 ~3.5만 개)를 가중 SMaxSim으로 하나씩 전수 비교.
  - 추정: 샘플당 2~5초 → `--diag-frac 0.02`(≈820개)면 실행당 30분~1시간 추가 (Colab 실측 아님)
- 진단 시간이 그 요청의 `timing_ms.total`에 섞임 → 시간 분석 시 진단 샘플 제외 필요.

선택지
- 샘플 비율: [x] ★ 0.01 (Colab 실행 명령에서 `--diag-frac 0.01`, 스크립트 기본값은 0.02 그대로)  [ ] 0.02  [ ] 0.05
- [x] ★ 진단 시간을 `timing_ms.diag`로 따로 기록하고 total에서 분리
- [ ] 전수 비교를 GPU 배치 계산으로 속도 개선

구현 (2026-10-08)
- 진단 시간을 따로 재서 `timing_ms.total`(과 결과 JSON의 지연 시간 목록)에서 빼고, 진단한 요청에만 `timing_ms.diag`로 기록
- 진단 중의 단계별 시간과, 백그라운드 스레드(캐시 추가 때 조각 계산)의 시간은 그 요청의 단계별 `timing_ms`에 넣지 않음
  (이전에는 백그라운드 작업 시간이 그때 처리 중이던 요청의 단계별 시간에 섞일 수 있었음)

## 5. requests.jsonl 용량

- 추정: 실행 1회 ≈ 140~160MB (조각 텍스트 ≈ 프롬프트 전체 24MB, 후보 20개 점수 ≈ 75MB, 나머지 ≈ 40MB)
- 조건 6개 × 3회 반복 ≈ 3GB.

선택지 (복수 선택 가능)
- [ ] ★ 조각 텍스트는 조건과 무관하므로 별도 파일에 한 번만 저장, requests에는 조각 수·토큰 수만
- [ ] ★ gzip 압축 저장 (`requests.jsonl.gz`)
- [ ] 현재대로

결정 (2026-10-08): 이번에는 적용하지 않음 (회당 약 150MB, 27회 약 4GB — Drive 여유 공간 확인 필요). `updates.jsonl`은 회당 수 MB 수준 추가

## 6. run.json 분할기 설정 칸

- `min_tokens`: 8번 규칙(짧은 조각 합치기)을 구현하지 않아 항상 null.
- `max_splits`: 규칙 분할기는 구두점마다 자르므로 null(= 제한 없음), 대조군만 0.

선택지
- [ ] ★ `min_tokens` 칸 삭제, `max_splits`는 `"no limit"` / `0` 으로 표기
- [ ] 현재대로 (null 의미를 문서에만 명시)

## 7. 아직 안 한 것 (참고)

- ~~실제 평가 실행으로 로그 전체를 검증하지 않음~~ → 2026-10-08 Colab T4에서 `--max-samples 300` 시범 실행으로 기록·집계(A~D, F) 동작 확인.
  단, 이때는 균등(가중치 없음) 조건이었고 위 3·4번 구현 전 코드라 `updates.jsonl`·`alpha_tprime`·`timing_ms.diag`와 가중치 관련 칸은 다시 확인 필요
- 가중치 통계(IDF·centroid·MLP)는 train(train/train.parquet)으로 아직 만들지 않음 (GPU에서 실행 예정)
