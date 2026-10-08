# MVR-Cache 인수인계 (2026-10-08)

> **답변은 한국어로 해 주세요.** 이 문서는 맥북 로컬 세션의 대화 내용을 이어받기 위한 요약입니다.

## 0. 한눈에 보기

- 목표: MVR-cache 논문(PKU-SDS-lab/MVR-Cache, 기준 커밋 `8cfaa0b`) 재현 → 가중치·분할 개선 실험
- 브랜치
  - `main`: 원저자 코드 그대로
  - `cpu-reproduction`: 맥북 CPU 재현용 (환경 설정, tools/, 예전 데이터·분석 스크립트)
  - **`gpu-production`**: 원본 + 가중치·분할·로그 코드만 (Colab GPU로 팀 공유, 지금 작업 브랜치)
- 실험 실행은 **Colab GPU**에서. 맥북 CPU로는 모델 실행 안 함 (`.venv`도 삭제함)
- 결정 대기 목록: [`.claude/logging-decisions.md`](.claude/logging-decisions.md)

## 1. gpu-production 브랜치 구성

| 커밋 | 내용 |
|---|---|
| `61a6839` | 규칙 분할기 + 조각 가중치(uniform/length/idf/centroid/mlp), 가중 MaxSim, 평가 옵션, 분석 스크립트, `data/classification.parquet` |
| `e9259fc` | Classification 분할 `train/train3k.parquet`, `val/val.parquet`, `test/test41k.parquet` + 각 `manifest.json` |
| `18e97bf` → `a7d419c` → `b890de1` | 규칙 분할기: 앞 4개만 자르던 문제 수정 → 구두점마다 자르기 → 예외 규칙 1~6 |
| `d5d1d69` | `build_segment_weight_stats.py` 기본값을 "파일 전체"로 (예전 기본 10000~12999행은 train3k에서 0행) |
| `d8bfd55` | 실험 로그 A~F (`--run-log-dir`) |
| `c95245f` | 결정 대기 목록 `.claude/logging-decisions.md` |

### 주요 파일
- `mvr-cache/vcache/vcache_core/splitter/RuleSplitter.py`: 규칙 분할기, 가중치 계산, `describe_split`(로그용)
- `mvr-cache/vcache/vcache_core/splitter/punctuation_rules.py`: 자를 위치 규칙 1~6
- `mvr-cache/vcache/vcache_policy/strategies/verified_splitter.py`: 가중 MaxSim, 상세 로그 수집(`detail_log`), `_LoggedAlgorithm`(판정 규칙은 원본과 동일, 값만 기록 + `--seed`)
- `mvr-cache/benchmarks/eval_sembenchmark_verified_splitter.py`: 평가 실행
- `mvr-cache/benchmarks/build_segment_weight_stats.py`: IDF·centroid·MLP 통계 생성 + 학습 로그(E)
- `mvr-cache/benchmarks/run_log.py`: 로그 A~D 기록 / `analyze_run_log.py`: 사후 집계 F

## 2. 확정된 결정

### 데이터 분할 (Classification 45k)
- train 3k / val 1k / test 41k. test·val은 팀원 파일(cls_seed42), train3k는 45k − test − val을 **(dataset_name, id)**로 매칭해 복원 (`id`만으로는 과제끼리 겹침)
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

[`.claude/logging-decisions.md`](.claude/logging-decisions.md) 참고. 요약:
1. 후보 방식: ★ `top_k --candidate-k 20 --use-cached-candidate-segments`로 고정
2. "코사인" = HNSW 저장 벡터 코사인 유지
3. vCache 판정 로그: `updates.jsonl` 추가 여부, t′ 기준 α 추가 여부
4. C 진단: ★ 비율 0.01, ★ 진단 시간 분리 기록
5. 용량: ★ 조각 텍스트 별도 저장, ★ gzip
6. run.json: ★ `min_tokens` 삭제, `max_splits`는 `"no limit"` 표기

## 6. 다음 할 일 (Colab GPU)

1. 저장소 클론, `gpu-production` 체크아웃, 원본 README대로 환경 + **커스텀 hnswlib**(멀티벡터) 설치
2. 가중치 통계 생성
   ```bash
   cd mvr-cache
   python benchmarks/build_segment_weight_stats.py \
     --dataset train/train3k.parquet --response-col response_llama_3_8b \
     --device cuda --out results/classification_train3k_weight_stats.pt
   ```
3. 로그 검증용 소규모 실행 (`--max-samples 300`)
   ```bash
   python benchmarks/eval_sembenchmark_verified_splitter.py \
     --dataset test/test41k.parquet --llm-col response_llama_3_8b --similarity-evaluator string \
     --delta 0.01 --sleep 0.02 \
     --candidate-selection top_k --candidate-k 20 --use-cached-candidate-segments \
     --splitter-mode rule --segment-weighting idf \
     --segment-weight-stats results/classification_train3k_weight_stats.pt \
     --include-full-embedding --splitter-device cuda \
     --run-log-dir runs/smoke --run-id smoke --condition "③ IDF" --seed 1 --repeat 1 \
     --train-file train/train3k.parquet --diag-frac 0.01 --max-samples 300
   python benchmarks/analyze_run_log.py runs/smoke
   ```
4. 본 실행 (조건별 × 반복) → `analyze_run_log.py`로 조건 × 과제 집계
5. RL 분할기 학습 (필요 시)
   ```bash
   cd rl-training-algorithm
   python RL4COTrainer.py --train_parquet ../mvr-cache/train/train3k.parquet \
     --val_parquet ../mvr-cache/val/val.parquet --parquet_text_column prompt \
     --label_mode string --response_column response_llama_3_8b ... (README 예시 참고)
   ```

## 7. 주의사항

- **로그 코드는 실제 평가로 아직 한 번도 안 돌려 봄** (기록·집계 로직만 가짜 데이터로 확인) → 6-3 소규모 실행으로 먼저 확인
- `results/`, `*.pt`, 체크포인트는 `.gitignore` 대상 → 통계 파일 공유는 Google Drive 또는 `git add -f`
- `requests.jsonl`은 실행 1회 ≈ 150MB 예상
- 예전 결과 수치(위 3절)는 옛 분할기·45k 원래 순서 기준이라 새 test41k에서는 달라질 수 있음
- 이 저장소(sgwag6799/MVR-Cache)는 공개 저장소
