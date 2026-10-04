# 실험 로그 — 결정할 사항

브랜치: `gpu-production` (커밋 `d8bfd55` 기준)
관련 코드: `mvr-cache/benchmarks/run_log.py`, `mvr-cache/benchmarks/analyze_run_log.py`,
`mvr-cache/vcache/vcache_policy/strategies/verified_splitter.py`

각 항목: 현재 동작 → 문제/주의 → 선택지 (추천에 ★)

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
- [ ] 관측 추가 시점 로그 `updates.jsonl` 추가 (요청 번호, 항목 id, s, c, 추가 전후 관측 수)
- [ ] α를 t′ 기준으로도 함께 기록 (`alpha_tprime`)
- [ ] 현재대로

## 4. C 샘플 진단

- 현재: 샘플마다 캐시 전체(끝으로 갈수록 ~3.5만 개)를 가중 SMaxSim으로 하나씩 전수 비교.
  - 추정: 샘플당 2~5초 → `--diag-frac 0.02`(≈820개)면 실행당 30분~1시간 추가 (Colab 실측 아님)
- 진단 시간이 그 요청의 `timing_ms.total`에 섞임 → 시간 분석 시 진단 샘플 제외 필요.

선택지
- 샘플 비율: [ ] ★ 0.01  [ ] 0.02  [ ] 0.05
- [ ] ★ 진단 시간을 `timing_ms.diag`로 따로 기록하고 total에서 분리
- [ ] 전수 비교를 GPU 배치 계산으로 속도 개선

## 5. requests.jsonl 용량

- 추정: 실행 1회 ≈ 140~160MB (조각 텍스트 ≈ 프롬프트 전체 24MB, 후보 20개 점수 ≈ 75MB, 나머지 ≈ 40MB)
- 조건 6개 × 3회 반복 ≈ 3GB.

선택지 (복수 선택 가능)
- [ ] ★ 조각 텍스트는 조건과 무관하므로 별도 파일에 한 번만 저장, requests에는 조각 수·토큰 수만
- [ ] ★ gzip 압축 저장 (`requests.jsonl.gz`)
- [ ] 현재대로

## 6. run.json 분할기 설정 칸

- `min_tokens`: 8번 규칙(짧은 조각 합치기)을 구현하지 않아 항상 null.
- `max_splits`: 규칙 분할기는 구두점마다 자르므로 null(= 제한 없음), 대조군만 0.

선택지
- [ ] ★ `min_tokens` 칸 삭제, `max_splits`는 `"no limit"` / `0` 으로 표기
- [ ] 현재대로 (null 의미를 문서에만 명시)

## 7. 아직 안 한 것 (참고)

- 실제 평가 실행으로 로그 전체를 검증하지 않음 (기록·집계 로직만 가짜 데이터로 확인)
  → Colab에서 `--max-samples 300` 정도로 먼저 돌려 확인 필요
- 가중치 통계(IDF·centroid·MLP)는 train3k로 아직 만들지 않음 (GPU에서 실행 예정)
