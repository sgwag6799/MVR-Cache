# 실험 로그 — 결정할 사항

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
