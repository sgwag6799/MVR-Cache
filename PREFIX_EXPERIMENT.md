# 주장 A 실험 기록 (`claim-a-rule-weighting` 브랜치)

> 이 브랜치에서 쓰는 것만 남긴 기록이다. 한 장 요약은 [`CLAIM_A.md`](CLAIM_A.md), 설계 원문은 [`PREFIX_PLAN.md`](PREFIX_PLAN.md)에 있다. 실험 1~7 전체 기록은 `prefix-exp7-invariant-encoder` 브랜치의 같은 파일에 있다.

## 1. 주장

프롬프트 앞에 답과 무관한 공통 문단이 붙는 환경에서는, **규칙 분할 + 학습 가중치(M3)**를 쓰는 MVR이 단일 벡터 vCache와 논문 MVR-Cache(RL 분할기)보다 **오류율 한도(δ = 0.01)를 지키면서 더 높은 히트율**을 낸다. 모든 비교는 base 인코더(`BAAI/bge-base-en-v1.5`)로 한다.

## 2. 설정

### 2.1 데이터: SemBenchmarkLmArena

| 항목 | 내용 |
|---|---|
| 원본 | `vCache/SemBenchmarkLmArena` 63,796행, 같은 뜻의 질문 묶음 `ID_Set` 3,500개 |
| 분할 | `scripts/make_lmarena_splits.py`: 묶음 단위로 `random.Random(42)` 순서대로 train ≥ 3,000 / val ≥ 1,000 / test ≥ 41,000행. 같은 묶음은 한 분할에만 들어간다. 지문: train `ec6e778ee76f` / val `411c8e1b6412` / test `3e5cc1fe8fa8` |
| 사용 행 | 모든 단계에서 512토큰 안에 드는 행(`fits_all_levels`): train 3,007 / val 1,000 / test 40,964 |
| 정답 | **`id_set_merged`**: 같은 묶음이면 정답. 원본에는 글자까지 같은 질문 186개가 서로 다른 `ID_Set`에 들어 있다. 이런 쌍은 점수 1.0인데 오답으로 세어지는 라벨 오류라, 같은 질문을 공유하는 묶음들을 하나로 합쳤다 (union-find, 원본 전체 기준, 3,500 → 3,303묶음, `scripts/lmarena_merge_duplicate_sets.py`) |
| 스트림 순서 | test 원래 순서 그대로 (같은 묶음이 연달아 나오는 비율 0.07%, 무작위로 섞었을 때 0.05%) |

### 2.2 공통 문단

- 공통 문장 풀 `prefix_pool.json`은 칸 8개(역할 · 말투 · 형식 등) × 변형 5개 = 40문장이다. 모두 답과 무관한 영어 어시스턴트 지시문이고, 15~18토큰, 구두점은 끝 마침표 하나다.
- 프롬프트마다 칸별로 문장 하나를 (시드 42, `dataset_name:id`)로 정한 난수로 고른다. 단계 L이면 앞 L칸을 쓴다.
- 붙이는 방식: `공통 문장들 + " " + 원본 프롬프트`. 원본 프롬프트와 정답은 그대로다.
- 단계: **L0**(원본)과 **L8**(공통 문장 8개). 만드는 스크립트는 `make_prefix_data.py`다.

### 2.3 분할과 점수

- **자르는 위치**: 토큰 `. , ! ? ; :`. `[CLS]`, `[SEP]`, `[SEP]` 바로 앞 문장 끝 구두점은 빼며, 팀 `rule_pointers`와 같은 규칙이다 (`prefix_split.py`).
- **상한 32**: 자르는 위치는 최대 32개만 쓴다. `--rule-cap-scope`가 그 32개를 어디서부터 셀지 정한다.
  - `original`: 공통 문장 목록으로 공통 문단 끝을 찾고, 원본 부분에서만 센다. 원본 부분 분할이 L0와 같아진다.
  - `all`: 목록 없이 맨 앞부터 센다.
- **조각 벡터**: 구간 안 토큰 벡터의 평균. `[CLS]`는 빼고, 마지막 조각에 `[SEP]`를 포함한다.
- **문장 벡터**: `[CLS]` · `[SEP]`을 포함한 모든 토큰의 평균. vCache 판정과 후보 검색에 쓴다.
- **SMaxSim**: 질문 조각마다 후보 조각 중 가장 비슷한 cos를 가중 평균하고, 반대 방향도 계산해 합친 뒤 (x + 1) / 2로 바꾼다.

### 2.4 조건

| 조건 | 방식 | 후보 · 점수 |
|---|---|---|
| A | vCache | 문장 벡터 코사인, 최근접 1개 |
| C | 규칙 분할 + 균등 가중치 (M2) | 문장 벡터로 후보 10개 → SMaxSim 재순위 |
| **E** | **규칙 분할 + 학습 가중치 (M3, 우리 방식)** | C와 같고, 조각마다 학습한 가중치를 씀 |
| F | 논문 MVR-Cache | RL 분할기, 조각 벡터 인덱스로 후보 10개(`multivector_top_k`), `--include-full-embedding --mix-fullcos` |
| I | 논문 MVR-Cache, `--mix-fullcos` 없음 | F에서 `--mix-fullcos`만 뺌 |
| B · D · G | A · C · F에 LoRA 인코더 | 참고용 (주장 B에서 다룸) |

- **M3 가중치 헤드**: Linear(768→1), 0 초기화, 프롬프트 안 조각끼리 softmax. **어느 조각이 공통 문단인지는 받지 않는다.**
  - 학습 쌍: train 각 행 × train 안 최근접 10. BCE(양성 비율 보정), Adam 1e-3, 배치 2,048, 20 에폭, val `auc_pairs`가 가장 높은 에폭.
  - 헤드 파일은 LmArena L8 train으로 학습한 `exp2_lmarena/offline_merged/prefix_w_variant_L8.pt`이고, E는 이 헤드 하나를 모든 단계에 쓴다.
- **RL 분할기**: 논문 체크포인트(SearchQueries 10K)가 공개되지 않아, 논문 레시피 그대로 LmArena train(L0)으로 학습한 것을 MVR-Cache 기준선으로 쓴다 (`scripts/train_rl_splitter_lmarena.sh`).
  - 레시피: `RL4COTrainer.py --label_mode id_set --train_sampling_mode anchor_nn --policy_mode separate --punctuation_only --bce_auto_balance`, 배치 8, lr 1e-4, 최대 200 에폭, 5 에폭마다 val, 조기 종료.
  - val reward가 가장 높은 체크포인트를 쓴다.
- **논문 MVR-Cache의 두 옵션**
  - `--include-full-embedding`: 조각 벡터들에 문장 전체 벡터 1개(`[CLS]`를 뺀 모든 토큰 평균)를 조각처럼 한 줄 더 붙인다.
  - `--mix-fullcos`: 최종 점수를 0.5 × SMaxSim + 0.5 × 문장 전체 코사인으로 섞는다 (둘 다 [0, 1]로 바꾼 뒤).
  - 논문 README 예시는 둘 다 쓴다. 논문 본문 수치는 `--mix-fullcos` 없이 잰 것으로 보여(논문 확인 필요) 조건 I를 둔다.

### 2.5 캐시 판정과 집계

- **판정**: vCache verified 정책, **δ = 0.01**(`--delta`, 오류율 허용 한도, 논문 본문과 같음), `--similarity-evaluator benchmark_id_set`.
- **`--sleep`**: 요청마다 기다리는 시간(초)이다. 판정 후 관측 기록(점수, 맞았는지) 갱신이 백그라운드 스레드에서 돌기 때문에, 다음 요청 전에 갱신이 끝나도록 기다린다. 너무 짧으면 관측이 덜 쌓인 상태로 판정해서 탐색(LLM 호출)이 늘어난다. 논문 README 값은 vCache 0.02, MVR-Cache 0.1이다.
- **시드**: L8은 시드 1 · 2 · 3, L0은 시드 1.
- **집계**: 앞 2,000요청은 캐시가 차는 구간이라 빼고 38,964요청으로 센다. 히트율 = 재사용 / 요청, 오류율 = 틀린 재사용 / 요청 (`prefix_stage2_report.py`).

## 3. 결과: 실험 8 (2026-10-28, NVIDIA L4, `--sleep 0.02`, `cap_scope original`)

### 3.1 히트율 (오류율)

L8은 시드 3개 평균, L0은 시드 1. 실행별 값은 [`exp8_stage2/stage2_runs.csv`](mvr-cache/benchmarks/prefix/exp8_stage2/stage2_runs.csv)에 있다.

| 조건 | L0 | L8 | L8 − L0 |
|---|---|---|---|
| A vCache | 21.74% (0.07%) | 17.54% (0.18%) | −4.20%p |
| C 규칙 분할 + 균등 | 20.67% (0.09%) | 18.59% (0.18%) | −2.08%p |
| **E 규칙 분할 + 학습 가중치** | 미실행 | **20.19%** (0.18%) | – |
| F 논문 MVR-Cache | 20.24% (0.08%) | 18.54% (0.23%) | −1.70%p |
| B vCache + LoRA (참고) | 21.67% (0.05%) | 20.71% (0.05%) | −0.96%p |
| D 규칙 균등 + LoRA (참고) | 20.45% (0.09%) | 20.25% (0.09%) | −0.20%p |
| G 논문 MVR-Cache + LoRA (참고) | 20.04% (0.07%) | 18.68% (0.24%) | −1.36%p |

### 3.2 주장 A 비교 (L8, 시드별)

| 시드 | E | A | F | C | E − A | E − F | E − C |
|---|---|---|---|---|---|---|---|
| 1 | 20.32% | 17.64% | 18.57% | 18.77% | +2.68%p | +1.75%p | +1.55%p |
| 2 | 20.00% | 17.47% | 18.52% | 18.43% | +2.53%p | +1.48%p | +1.57%p |
| 3 | 20.25% | 17.52% | 18.54% | 18.56% | +2.73%p | +1.70%p | +1.69%p |
| 평균 | **20.19%** | 17.54% | 18.54% | 18.59% | **+2.65%p** | **+1.65%p** | **+1.60%p** |

- E 오류율은 최대 0.20%로 δ 이내다. 세 비교 모두 시드 3개에서 같은 방향이다.
- 요청당 시간(병렬 4~5개, sleep 포함이라 상대 비교만): vCache ≈ 51 ms, 규칙 MVR ≈ 90 ms (L0 67 ms), 논문 MVR-Cache ≈ 130 ms.
- RL 분할기 학습: val reward가 첫 검증(에폭 4, −0.019) 뒤로 좋아지지 않았다 (에폭 9 −0.070, 14 −0.035, 19 · 24 · 29 개선 없음 → 에폭 29에서 조기 종료). 에폭 4 체크포인트를 썼고, 프롬프트당 조각은 최대 4개다.

### 3.3 오프라인 지표 (같은 방향)

L8 `auc_selected`(앵커 6,000개마다 1등 후보의 점수–정답 AUC), base 인코더.

| 데이터 | vCache (M1) | 균등 (M2) | 학습 가중치 (M3) | 공통 조각 0 (M4, 상한 비교용) |
|---|---|---|---|---|
| LmArena (`id_set_merged`) | 0.617 | 0.616 | **0.694** | 0.691 |
| Classification (팀 분할 cls_seed42, 8B 라벨) | 0.6234 | 0.6241 | **0.6501** | 0.6413 |

## 4. 약점

1. **사전 기준 미달 · 보조 분석**: 원래 설계의 1단계(Classification 오프라인) 통과 기준은 미달이라, 설계 규칙상 2단계 스트림은 없었다. 실험 8은 그 뒤에 설계했고 사전 기준은 LoRA vs base였다. E vs A · F는 보조 분석이다. 데이터(LmArena)와 정답 정의(`id_set_merged`)도 결과를 본 뒤 정했다.
2. **E의 L0 없음**: L0에서는 vCache가 가장 높다 (21.74%). 헤드 하나로 공통 문단 수가 섞인 스트림을 처리할 수 있는지도 확인하지 않았다.
3. **데이터셋 1개**: LmArena, 공통 문장 풀 1개, 앞에만, L0 · L8.
4. **RL 분할기 재학습**: 논문 체크포인트가 없어 새로 학습했고, 학습이 잘 되지 않았을 수 있다 (3.2).
5. **공통 문단 정보 사용**: `cap_scope original`은 공통 문장 목록으로 공통 문단 끝을 찾는다. 실제 서비스에서는 쓸 수 없는 정보다. 가중치(M3)는 이 정보를 쓰지 않는다.
6. **같은 공통 문장으로 학습 · 평가**: M3 헤드는 test와 같은 공통 문장 40개가 붙은 train으로 학습했다.
7. **`--sleep 0.02`**: 논문 MVR-Cache는 0.1이다. 갱신이 무거운 MVR 계열이 더 불리했을 수 있다.

## 5. 정한 것

| 항목 | 결정 |
|---|---|
| MVR-Cache 기준선 | 여기서 학습한 RL 분할기. 학습 곡선과 조각 수 분포를 함께 보고한다 |
| `--sleep` | 모든 조건 0.1 |
| δ | 0.01 |
| 상한 32 | `cap_scope all` (공통 문장 목록을 쓰지 않음) |
| 정답 | LmArena는 `id_set_merged`로 고정 |
| M3 일반화 | 학습용 풀 `prefix_pool_train.json`(평가 풀과 겹치는 문장 0개)으로 헤드를 학습한 조건 E′ 추가 |
| `--mix-fullcos` | 논문 MVR-Cache를 있음(F) · 없음(I) 둘 다 잰다 |

## 6. 보강 실험 계획과 판정 기준 (실행 전에 커밋으로 고정)

| ID | 작업 목록 | 설정 | 실행 | 약점 |
|---|---|---|---|---|
| A0 + 채우기 | `stage2_jobs_claimA_fill.txt` | 실험 8과 같음 (`SLEEP=0.02`, `CAP_SCOPE=original`) | 재현 확인(A · C · E · F, L8 시드 1이 3.2와 ±0.2%p) + **E의 L0** + **I**. 9개 | 2, I |
| A1 | `stage2_jobs_claimA_sleep01.txt` | **`SLEEP=0.1`**, `CAP_SCOPE=original` | A · C · E · F · I × (L8 시드 1 · 2 · 3, L0 시드 1). 20개 | 7 |
| A1b | `stage2_jobs_claimA_capall.txt` | `SLEEP=0.1`, **`CAP_SCOPE=all`** | C · E × (L8 시드 1 · 2 · 3, L0 시드 1). 8개. A1과 비교해 공통 문장 목록의 영향만 분리 | 5 |
| A2 | (데이터 준비 후 작성) | A1 · A1b와 같음 | Classification L0 · L8 | 3 |
| A3 | (헤드 학습 후 작성) | A1과 같음 | 처음 보는 템플릿 E′ | 6 |
| A4 | – | – | 참고: vCache + LoRA 인코더(B) 함께 보고 | 리뷰어 질문 |

- 예상 시간(L4, 병렬 4~5개): A0 + 채우기 약 1.5시간, A1 약 8시간, A1b 약 3시간.
- 설정마다 결과 폴더를 나누고, `scripts/autosave_stage2.sh`가 실행 중에 집계표를 push한다.

**판정 기준 (데이터마다)**

1. L8에서 E 히트율 > A, F (시드 3개 모두)
2. E 오류율 ≤ δ (모든 단계 · 시드)
3. L0에서 E 히트율이 C보다 0.5%p 넘게 낮지 않음 (A와의 차이는 그대로 보고)
- 보조: E − C, E − I, E′ − E, 요청당 시간, B와의 차이

## 7. 파일

| 파일 | 역할 |
|---|---|
| `scripts/make_lmarena_splits.py` · `scripts/lmarena_merge_duplicate_sets.py` | LmArena 분할 · 합친 정답 `id_set_merged` |
| `mvr-cache/benchmarks/prefix/prefix_pool.json` · `make_prefix_data.py` | 공통 문장 풀 · 단계별 데이터 |
| `mvr-cache/vcache/vcache_core/splitter/prefix_split.py` | 분할 규칙 (자르는 위치, 상한, 공통 조각 표시) |
| `mvr-cache/vcache/vcache_core/splitter/PrefixRuleSplitter.py` | 스트림용 규칙 분할기 + M3 헤드 |
| `mvr-cache/benchmarks/prefix/prefix_offline.py` | 오프라인 지표 · M3 헤드 학습 |
| `mvr-cache/benchmarks/prefix/exp2_lmarena/offline_merged/prefix_w_variant_L8.pt` | E의 M3 헤드 |
| `scripts/make_stage2_lmarena_data.py` | 스트림(L0 · L8 test) · RL 학습용 파일 |
| `scripts/train_rl_splitter_lmarena.sh` | MVR-Cache 기준선 RL 분할기 학습 |
| `scripts/run_stage2_lmarena.sh` · `scripts/stage2_jobs_*.txt` | 스트림 실행 (조건 A~G · I, 환경변수 `SLEEP` · `CAP_SCOPE` · `HEAD` · `OUT`) |
| `scripts/run_claimA_stage2.sh` · `scripts/autosave_stage2.sh` | 보강 배치를 순서대로 실행 · 결과 자동 push |
| `mvr-cache/benchmarks/prefix/prefix_stage2_report.py` | 히트율 · 오류율 집계 |
| `mvr-cache/benchmarks/prefix/exp8_stage2/` | 실험 8 결과표 |
| `eval_sembenchmark_verified.py` · `eval_sembenchmark_verified_splitter.py` (수정) | `--seed`, `--filter-true-column`, `--splitter-mode prefix_rule`, `--rule-cap-scope`, `--prefix-pool`, `--segment-head` |

## 8. 재현

```bash
# 런타임마다
git clone -b claim-a-rule-weighting https://github.com/sgwag6799/MVR-Cache.git /root/MVR-Cache
cd /root/MVR-Cache && source colab_setup.sh
# LmArena 데이터 (GPU 불필요, 수 분)
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkLmArena','train.parquet',repo_type='dataset',local_dir='data/raw_lmarena')"
python scripts/make_lmarena_splits.py --raw data/raw_lmarena/train.parquet --out data/lmarena   # 지문 확인
(cd mvr-cache && python -m benchmarks.prefix.make_prefix_data --train ../data/lmarena/train.parquet \
  --val ../data/lmarena/val.parquet --test ../data/lmarena/test.parquet --out ../data/lmarena/prefix/data)
python scripts/lmarena_merge_duplicate_sets.py --raw data/raw_lmarena/train.parquet --data data/lmarena/prefix/data
python scripts/make_stage2_lmarena_data.py               # stream L0 · L8 각 40,964행, rl train 3,007 / val 1,000
bash scripts/train_rl_splitter_lmarena.sh                # → data/exp8/rl/best.ckpt
# 보강 실험: A0 + 채우기 → A1 → A1b 순서대로 (VSCode 터미널에서, push 인증 필요)
mkdir -p data/claimA
nohup bash scripts/run_claimA_stage2.sh 5 > data/claimA/run.log 2>&1 < /dev/null &
#   중간에 끊기면 남은 배치부터: nohup bash scripts/run_claimA_stage2.sh 5 sleep01 > data/claimA/run.log 2>&1 < /dev/null &
#   집계 결과: mvr-cache/benchmarks/prefix/claimA_stage2/{fill,sleep01,capall}/stage2_summary.csv
```

## 9. 이 브랜치의 출처

- 실험 7 브랜치(`prefix-exp7-invariant-encoder`, `a8c9d57`) 위에 실험 8을 다시 올렸다. 실험 8 원래 커밋은 Colab VM에서 push되지 않았다.
- 실험 8의 코드 · 설정 · 결과는 VM 세션 기록에서 복원했다.
  - `stage2_runs.csv`는 VM에서 출력된 실행별 표(소수 6자리)를 옮긴 것이다.
  - 실행별 원본 JSON, RL 학습 로그 · 체크포인트, LoRA 모델은 없다.
- 실험 8 중 고친 것: `eval_sembenchmark_verified.py`의 결과 저장 시 numpy float32 직렬화 오류 (`default=float`), `prefix_invariance_train.py`의 학습 전 상태 저장 버그.
