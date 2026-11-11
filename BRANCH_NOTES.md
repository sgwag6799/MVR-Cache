# `cpu-reproduction` 브랜치 노트

main(논문 공개본) 위에 커밋 4개를 올린 브랜치다. 논문 코드를 **맥(Apple Silicon) CPU에서 vllm · CUDA 없이** 돌리고, 규칙 분할기 · 조각별 가중치 · (s, c) 기록을 추가했다. 같은 작업을 Colab GPU용으로 다시 정리한 것이 `gpu-production` 브랜치다 (이 브랜치 위에 쌓인 것은 아님).

| 커밋 | 날짜 | 내용 |
|---|---|---|
| `5fc4d92` | 2026-10-02 | vllm · CUDA 없이 macOS CPU에서 실행 |
| `2c87426` | 2026-10-02 | 규칙 분할기, 조각별 가중치, (s, c) 기록 |
| `38ffea8` | 2026-10-02 | 실험에 쓴 데이터셋과 train/val/test 분할 추가 |
| `5d936e0` | 2026-10-04 | 가중치 코드에 한국어 주석, MLP를 평가 점수 그대로 학습 |

## 1. 맥 CPU 실행

- `mvr-cache/pyproject.toml`: vllm 의존성 제거, scipy 상한 1.17 미만, `[tool.poetry] packages = vcache` (`pip install -e` 가능).
- `vcache/inference_engine/__init__.py` · `strategies/__init__.py` · `strategies/vllm.py`: vllm import를 try/except로 감쌌다. 벤치마크 엔진만 쓰면 vllm이 없어도 된다.
- `rl-training-algorithm/RL4COTrainer.py`: CUDA가 없으면 정밀도를 `32-true`로 쓴다 (`16-mixed`는 CPU에서 bf16이 되어 포인터 정책의 float32 버퍼가 깨진다).
- `requirements-lock.txt`: 실험에 쓴 패키지 버전 고정.
- `tools/setup_mac.sh`: Homebrew Python 3.11 + 가상환경(`.venv`) 설치, 로컬 multi-vector hnswlib 빌드, import 확인. 다른 맥으로 폴더를 복사한 뒤(`.venv` 제외) `bash tools/setup_mac.sh`.

## 2. 규칙 분할기와 조각별 가중치

**`vcache/vcache_core/splitter/RuleSplitter.py`**

- `RulePunctuationSplitter`: RL 체크포인트 없이 구두점에서 자르는 분할기. 자르는 문자는 RL 정책과 같은 `PUNCT_CHARS`(영문 · 전각 구두점 12개)이고, `[CLS]`와 `[SEP]` 바로 앞 구두점은 제외한다. 앞에서부터 `--splitter-max-segments`개 위치에서 자른다.
- 조각 가중치 (`--segment-weighting`):

| 방식 | 가중치 |
|---|---|
| `uniform` | 1 (main과 같음) |
| `length` | 조각 토큰 수 |
| `idf` | 조각 토큰 IDF 평균 |
| `centroid` | 1 − cos(조각, 참고 데이터 조각 평균) |
| `mlp` | softplus(MLP(조각 벡터)) + 1e-3 |

- 문장 전체 행(`--include-full-embedding`)의 가중치는 조각 가중치 평균이다.
- `SegmentRowWeighter`: RL 분할기의 조각에 centroid · mlp 가중치를 붙인다 (Step 5).

**`vcache/vcache_policy/strategies/verified_splitter.py`**: MaxSim 계산 네 곳을 `_score_tensors`로 바꿔, 분할기에 `segment_weights`가 있으면 가중 MaxSim을 쓴다. 없으면 main과 같다. 결정마다 s · 항목 id · 행동 · 관측 수 · t̂ · γ를 `last_decision`에 남긴다.

**`benchmarks/build_segment_weight_stats.py`**: 가중치에 쓰는 통계 파일(`.pt`)을 만든다. IDF 표, 조각 평균(centroid), MLP. MLP는 단일 벡터 코사인 최근접 k개 쌍을 정답(응답 일치 또는 ID_Set 일치)으로 BCE 학습하고, 앵커를 80/20으로 나눠 검증한다. 학습 점수는 평가와 같은 순수 가중 MaxSim이다 (`--mix-fullcos`를 주면 예전 방식인 전체 코사인 50% 혼합). `--epochs 0`이면 IDF · centroid만 저장한다.

## 3. 평가 스크립트 추가 옵션 (`benchmarks/eval_sembenchmark_verified_splitter.py`)

| 옵션 | 내용 |
|---|---|
| `--splitter-mode rl\|rule` | `rule`이면 규칙 분할기, `--splitter-checkpoint`는 `rl`에만 필요 |
| `--segment-weighting` | 위 표의 다섯 방식 (rule 모드) |
| `--segment-weight-stats` | `build_segment_weight_stats.py`가 만든 통계 파일 (idf · centroid · mlp에 필요) |
| `--log-scores` | 요청마다 s, c(tp 또는 fn), 행동, t̂, γ, 관측 수를 JSONL로 기록 |
| `--dump-observations` | 종료 후 항목별 (s, c) 관측 전체를 저장 (사전 관측 2개 제외) |

## 4. 분석 스크립트 (`mvr-cache/benchmarks/`)

- `eval_splitter_pair_separation.py`: 캐시 정책 없이 점수만 비교. 단일 벡터 후보 쌍에서 vCache 코사인, MVR 점수(0.5 MaxSim + 0.5 코사인), MaxSim을 학습 · 규칙 · 무작위 분할별로 AUC와 1등 정답률로 비교한다.
- `analyze_logged_scores.py`: `--log-scores` 기록으로 s 분포, 보정 곡선, 항목별 통계, 히트가 난 위치와 정밀도.
- `analyze_observation_normality.py`: `--dump-observations` 관측이 로지스틱 모델 전제(정답 · 오답별 s가 같은 분산의 정규분포)를 만족하는지 확인.

## 5. 데이터 (`mvr-cache/data/`)

- `classification.parquet`, `searchqueries.parquet`, `lmarena.parquet`: 원본.
- `classification_first10k_seed{1,2,3}.parquet`: 시드 1 · 2 · 3 평가 스트림 (1만 행).
- `classification_rl/`, `classification_rl_10k/`: RL 분할기 train/val/test.
- `lmarena_rl/`: `rl-training-algorithm/scripts/prepare_lmarena_splits.py`로 만든 ID_Set 단위 분할 (train/val/test + `eval_stream`). 묶음당 최대 2개, 50%는 한 개만 남겨 음성 쌍이 생기게 한다.

## 6. 실행 (`tools/`)

```bash
bash tools/setup_mac.sh                                   # 1회
bash tools/run_3a.sh                                      # Classification: uniform · idf, 3회씩 (~12h)
CONDITIONS="uniform idf length centroid" bash tools/run_3a.sh
bash tools/run_searchqueries.sh                           # SearchQueries: nosplit · uniform, 3회씩 (~17h)
python tools/make_share_bundle.py --out ~/Desktop/mvr-cache-results   # 결과를 작은 묶음으로
```

- 공통 설정: δ 0.01, `--sleep 0.02`, 규칙 분할 상한 4, `multivector_top_k` 후보 10, 문자열 정답 판정, CPU. `caffeinate`로 잠자기를 막는다.
- 정책이 백그라운드 스레드로 문턱을 갱신하므로 기계가 다르면 히트율이 조금 달라진다. **같은 기계에서 돈 `uniform`과만 비교한다.**
- `make_share_bundle.py`: 실행별 요약 · 인자와 500요청 간격 히트율 곡선, 가중치 통계 `.pt`를 `summary.csv`와 함께 묶는다.

## 7. 알려진 한계

- `run_3a.sh`의 가중치 통계는 처음 3,000행(`--start 0 --n 3000`)으로 만드는데, 이 행들은 평가 스트림에도 들어 있다 (스크립트 주석에 명시).
- 규칙 분할기는 앞에서부터 `max_segments`개 구두점에서만 자른다 (`RuleSplitter.py:237`). 모든 구두점에서 자르는 규칙은 `gpu-production`의 `punctuation_rules.py`에서 바뀌었다.
- `--mix-fullcos`는 multivector 후보 경로에서는 효과가 없다 (`verified_splitter.py` 주석). `run_3a.sh` · `run_searchqueries.sh`는 이 옵션을 넘기지만 점수에 반영되지 않는다. 같은 이유로 `eval_splitter_pair_separation.py`의 "mvr" 점수(코사인 50% 혼합)는 실제 평가 점수와 다르다.
- 가중치는 출력 텐서 내용의 해시로 찾는다. 캐시에 없으면 length · idf는 균등 가중치로 돌아간다.
