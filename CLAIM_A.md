# 주장 A: 공통 문단 환경에서 규칙 분할 + 학습 가중치가 vCache · MVR-Cache보다 낫다

> 프롬프트 앞에 답과 무관한 공통 문단이 붙는 환경에서, **규칙 분할(구두점 위치, 상한 32) + 학습 가중치(M3)** 를 쓰는 MVR이 단일 벡터 vCache와 논문 MVR-Cache(RL 분할기)보다 **오류율 한도(δ = 0.01)를 지키면서 더 높은 히트율**을 낸다.

- 인코더: 모든 비교는 base 인코더 `BAAI/bge-base-en-v1.5` 기준
- 데이터셋 (공통 문단 붙이는 방식): [`PREFIX_EXPERIMENT.md`](PREFIX_EXPERIMENT.md), 설계 원문 [`PREFIX_PLAN.md`](PREFIX_PLAN.md)
- 결과 파일: [`mvr-cache/benchmarks/prefix/exp8_stage2/`](mvr-cache/benchmarks/prefix/exp8_stage2/)

## 근거: 실험 8 (LmArena 41K 스트림, 공통 문장 8개 L8)

앞 2,000요청 제외 38,964요청, δ = 0.01, 시드 1 · 2 · 3 평균. 괄호 안은 오류율.

| 방식 | 조건 | L0 (원본) | L8 | L8 − L0 |
|---|---|---|---|---|
| vCache | A | 21.74% (0.07%) | 17.54% (0.18%) | −4.20%p |
| 논문 MVR-Cache (재학습 RL 분할기) | F | 20.24% (0.08%) | 18.54% (0.23%) | −1.70%p |
| 규칙 분할 + 균등 (M2) | C | 20.67% (0.09%) | 18.59% (0.18%) | −2.08%p |
| **규칙 분할 + 학습 가중치 (M3)** | **E** | 미실행 | **20.19%** (0.18%) | – |

L8에서 E − A = **+2.65%p**, E − F = **+1.65%p**, E − C = **+1.60%p**, 세 비교 모두 시드 3개에서 같은 방향 ([`claim_a_L8_comparison.csv`](mvr-cache/benchmarks/prefix/exp8_stage2/claim_a_L8_comparison.csv)).

## 약점과 정한 것

| 약점 | 정한 것 · 보강 |
|---|---|
| 1단계 사전 기준 미달, E vs A · F는 실험 8의 보조 분석, 데이터 · 정답 정의는 결과 본 뒤 선택 | 아래 판정 기준을 실행 전에 커밋으로 고정하고 다시 확인 (A1 · A2) |
| E의 L0 없음 | A1에서 L0 실행 |
| 데이터셋 1개 | A2에서 Classification 추가 |
| 논문 체크포인트 미공개 → RL 분할기 재학습 | 재학습 분할기를 MVR-Cache 기준선으로 쓰고 학습 곡선 · 조각 수 분포 보고 |
| 상한 32 적용에 공통 문장 목록 사용 (`cap_scope original`) | `CAP_SCOPE=all`로 목록 없이 실행 |
| M3 헤드가 test와 같은 공통 문장으로 학습 | A3: 학습용 풀(`prefix_pool_train.json`)로 헤드 학습 후 평가 풀로 평가 |
| `--sleep 0.02` (논문 MVR-Cache 0.1) | `SLEEP=0.1`, 모든 조건 같게. δ는 0.01 그대로 |

## 보강 실험과 판정 기준

논문 MVR-Cache 옵션: `--include-full-embedding`은 조각 벡터에 문장 전체 벡터 1개를 조각처럼 한 줄 더 붙이고, `--mix-fullcos`는 최종 점수를 0.5 × SMaxSim + 0.5 × 문장 전체 코사인으로 섞는다. 실험 8의 F는 README 예시대로 둘 다 썼다. **조건 J = F에서 두 옵션을 모두 뺀 것**(조각 벡터의 SMaxSim만, RL 분할기 · `multivector_top_k` k 10은 같음)을 추가한다. L8에서는 문장 전체 벡터의 약 3/4이 공통 문단이라, 두 옵션이 F에 불리하게 작용했는지 본다.

| ID | 작업 목록 | 설정 | 내용 |
|---|---|---|---|
| A0 + 채우기 | `stage2_jobs_claimA_fill.txt` | 실험 8과 같음 (sleep 0.02) | 재현 확인(A · C · E · F, L8 시드 1) + **E의 L0** + **J** |
| A1 | `stage2_jobs_claimA_sleep01.txt` | **sleep 0.1** | A · C · E · F · J, L8 시드 1 · 2 · 3 + L0 시드 1 |
| A1b | `stage2_jobs_claimA_capall.txt` | sleep 0.1, **cap all** | C · E (공통 문장 목록 없이) |
| A2 | – | A1과 같음 | Classification |
| A3 | – | A1과 같음 | 처음 보는 템플릿 E′ |
| A4 | – | – | 참고: vCache + LoRA 인코더(B) |

- E는 모든 단계에서 L8 헤드 하나를 쓴다 (공통 문단 수를 모르는 실제 서비스 조건).
- 실행: `nohup bash scripts/run_claimA_stage2.sh 5 > data/claimA/run.log 2>&1 < /dev/null &` (배치를 순서대로, 결과는 `mvr-cache/benchmarks/prefix/claimA_stage2/`로 자동 push). 예상 시간: A0 + 채우기 약 1.5시간, A1 약 8시간, A1b 약 3시간.

데이터마다:
1. L8에서 E 히트율 > A, F (시드 3개 모두)
2. E 오류율 ≤ δ (모든 단계 · 시드)
3. L0에서 E 히트율이 C보다 0.5%p 넘게 낮지 않음

## 파일

| 파일 | 역할 |
|---|---|
| `scripts/make_lmarena_splits.py` · `scripts/lmarena_merge_duplicate_sets.py` | LmArena 분할(지문 train `ec6e778ee76f` / val `411c8e1b6412` / test `3e5cc1fe8fa8`) · 합친 정답 `id_set_merged` |
| `mvr-cache/benchmarks/prefix/make_prefix_data.py` · `prefix_pool.json` | 공통 문단 데이터 L0 · L2 · L4 · L8 |
| `scripts/make_stage2_lmarena_data.py` | 스트림(L0 · L8 test) · RL 분할기 train/val 파일 |
| `scripts/train_rl_splitter_lmarena.sh` | MVR-Cache 기준선 RL 분할기 학습 (논문 레시피) |
| `mvr-cache/benchmarks/prefix/exp2_lmarena/offline_merged/prefix_w_variant_L8.pt` | E의 M3 가중치 헤드 (LmArena L8 train으로 학습) |
| `scripts/run_stage2_lmarena.sh` · `scripts/stage2_jobs_*.txt` | 스트림 실행 (조건 A~G · J, `SLEEP` · `CAP_SCOPE` · `HEAD` · `OUT` 환경변수) |
| `scripts/run_claimA_stage2.sh` · `scripts/autosave_stage2.sh` | 보강 배치를 순서대로 실행 · 결과 자동 push |
| `mvr-cache/benchmarks/prefix/prefix_stage2_report.py` | 히트율 · 오류율 집계 |
| `mvr-cache/vcache/vcache_core/splitter/PrefixRuleSplitter.py` | 규칙 분할기 + M3 헤드 |

데이터 파일(`data/`)은 git에서 제외되어 있다. 위 스크립트로 Hugging Face 원본(`vCache/SemBenchmarkLmArena`)에서 같은 파일을 다시 만든다. 데이터 만드는 방법은 [`PREFIX_EXPERIMENT.md`](PREFIX_EXPERIMENT.md), 실험 실행은 `scripts/run_claimA_stage2.sh`.
