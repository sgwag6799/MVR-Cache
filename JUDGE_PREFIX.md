# 실험 3: 공통 문단이 붙은 Classification에서 판정자 확인

> 질문: 후보 답 합의 판정자(learned_all_add35)는 프롬프트 앞에 공통 문단(템플릿)이 붙어도 원본에서처럼 같은 오류율로 히트율을 올리는가?

- 브랜치 `judge-prefix-classification` (기반: `claim-a-classification`)
- 데이터 · 공통 문단 규칙: [`PREFIX_EXPERIMENT.md`](PREFIX_EXPERIMENT.md), 기준선 · 스트림 설정: [`CLAIM_A.md`](CLAIM_A.md)의 Classification 실행
- 판정자: 팀 판정자 코드(`code_changes.patch`)의 합의 함수와 팀 실험 기록 2~5의 정책(learned_all_add35)을 이 저장소 정책 코드에 옮김

## 1. 왜 기대할 만한가 / 무엇이 걸리나

| 공통 문단이 바꾸는 것 | 판정자에 미치는 영향 |
|---|---|
| ① 조각 벡터 오염 → ③ 점수 s가 흐려짐 | Classification 판정자는 원래 s를 거의 쓰지 않음 (test에서 s AUC 0.490 vs 합의 0.859) |
| ② 후보 10개가 바뀜 (L8에서 L0 후보와 32% 겹침) | 오염 분석에서 L8 후보 자체는 나빠지지 않음 (후보만 바꾼 경우 R 대비 +0.014) |
| 후보들의 저장된 답 | 공통 문단은 정답을 바꾸지 않음 → 합의 신호는 그대로 |
| 1등 정답률 하락 (균등 L0 0.812 → L8 0.744) | 판정자가 걸러야 할 틀린 1등이 늘어 정밀도가 떨어질 수 있음 |

## 2. 설계 (실행 전 고정)

| 항목 | 내용 |
|---|---|
| 데이터 | Classification 공통 문단 L8, **공용 풀**(`prefix_pool.json`) 필수, **과제별 풀**(`prefix_pool_task.json`)은 시간이 되면. 참고로 L0 |
| 기준선 C | 규칙 분할(구두점, 상한 32, `cap_scope original`) + 균등 SMaxSim, 문장 벡터로 후보 10개, δ 0.01, sleep 0.02, 정답 = 8B 응답 문자열 일치 (실험 2의 C와 같은 설정) |
| 판정자 학습 스트림 | 같은 단계 · 같은 풀의 **train + val** (`fits_all_levels`만, 2,791 + 935 = 3,726행). C로 흘려 특징만 모음 (판정 변경 없음). **test는 쓰지 않음** |
| 판정자 | 로지스틱 P(1등 답이 맞음 \| s, a_excl, 후보 수), 특징 표준화, 5-fold OOF. a_excl = 1등을 뺀 후보 중 1등과 같은 답의 비율, 후보 5개 이상일 때만 적용 |
| 정책 | **learned_all_add35**: vCache가 재사용하지 않기로 한 모든 요청(cold + explore)에서 P ≥ p*이면 재사용 |
| p* | 학습 스트림의 기준선 미스 중 OOF P가 높은 순으로 받아들이며 "받아들인 오답 / 전체 요청 ≤ 0.35%"를 지키는 가장 낮은 P. 수집 시드를 여러 개 쓰면 시드별 p*의 중앙값 (팀 Jmed 규칙) |
| test 조건 | C vs **CJ (C + 판정자)**, L8 시드 1 · 2 · 3 |
| 선택 조건 | L0 C · CJ 시드 1 (같은 설정에서 원본 이득과 비교), E · EJ (학습 가중치 + 판정자, 공용 풀만) |
| 집계 | 앞 2,000행 제외, 전체 · 하위 과제별 히트율 · 오류율 (`prefix_stage2_report.py`), 판정 이유별 건수 (판정 기록 jsonl) |

### 판정 기준

1. L8(공용 풀)에서 CJ 히트율 > C, 시드 3개 모두
2. CJ 오류율 ≤ δ (모든 시드). 판정자가 더한 오답(CJ 오답 − C 오답, 요청 대비)을 예산 0.35%와 함께 보고
3. 보조: 같은 설정의 L8 이득(CJ − C)이 L0 이득의 절반 이상이면 "템플릿에 강건"

### 결과 해석 (실행 전 고정)

| 결과 | 해석 |
|---|---|
| L8 이득이 L0과 비슷 | 판정자는 템플릿에 강건하다 → 실서비스 템플릿 환경에 그대로 적용 가능 |
| 이득이 줄지만 남음 | 판정 단계(④)와 인코딩 · 점수 단계(① · ③)를 함께 고쳐야 한다 → E + 판정자, LoRA + 판정자 |
| 거의 없음 | 공통 문단이 후보 구성까지 흔들어 합의 신호가 약해진다 → ①을 먼저 고쳐야 한다 |
| 오류율이 δ를 넘거나 추가 오답이 예산을 크게 넘음 | 학습 스트림과 test의 차이로 p*가 어긋남 (SearchQueries와 같은 문제) → on-policy 재보정 필요 |

## 3. 실행 (Colab, repo 루트)

```bash
git clone -b judge-prefix-classification https://github.com/sgwag6799/MVR-Cache.git /root/MVR-Cache
cd /root/MVR-Cache && source colab_setup.sh
# 데이터 (data/는 git에 올리지 않는다)
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkClassification','train.parquet',repo_type='dataset',local_dir='data/raw')"
python scripts/make_cls_splits.py
cd mvr-cache
python -m benchmarks.prefix.make_prefix_data --train ../data/cls_seed42/train.parquet --val ../data/cls_seed42/val.parquet \
  --test ../data/cls_seed42_nodedup/test41k.parquet --out ../data/prefix/data --levels 0 8 --modes variant
python -m benchmarks.prefix.make_prefix_data --train ../data/cls_seed42/train.parquet --val ../data/cls_seed42/val.parquet \
  --test ../data/cls_seed42_nodedup/test41k.parquet --out ../data/prefix_task/data --levels 0 8 --modes variant \
  --pool benchmarks/prefix/prefix_pool_task.json          # 과제별 풀 (선택)
cd ..
python scripts/judge/make_judge_cls_data.py              # → data/judge_cls/{shared,task}/L{0,8}_{trainval,test}.parquet

# 핵심 (판정자 학습 약 10분 + test 6회, L4 기준 약 6시간)
POOL=shared bash scripts/judge/run_judge_prefix_cls.sh scripts/judge/jobs_judge_core.txt 3
# 선택
POOL=shared bash scripts/judge/run_judge_prefix_cls.sh scripts/judge/jobs_judge_l0.txt 2
POOL=task   bash scripts/judge/run_judge_prefix_cls.sh scripts/judge/jobs_judge_core.txt 3
POOL=shared bash scripts/judge/run_judge_prefix_cls.sh scripts/judge/jobs_judge_extra.txt 3
# 판정자 수집 시드를 늘리려면 (p* 중앙값): COLLECT_SEEDS="42 43 44 45"

# 집계
cd mvr-cache && python -m benchmarks.prefix.prefix_stage2_report --runs ../data/judge_cls/runs/shared --out ../data/judge_cls/report/shared
```

- 실험 2(`claim-a-classification` 핵심 배치)의 `C_L8_s*.json`은 같은 데이터 · 설정이라 `data/judge_cls/runs/shared/`에 복사해 두면 C 실행을 건너뛴다.
- 결과 파일: `data/judge_cls/runs/<풀>/<이름>.json` (히트율 · 요청별 tp/fp), `<이름>.jsonl` (요청별 판정 기록: s, a_excl, 후보 수, P, 판정 이유), 판정자 `data/judge_cls/judge/<풀>_L<단계>_all_b0.0035.json`

## 4. 파일

| 파일 | 역할 | 출처 |
|---|---|---|
| `mvr-cache/vcache/vcache_core/splitter/consensus.py` | 후보 답 합의(a_excl) · 판정자 확률 계산 | 팀 판정자 코드 그대로 |
| `scripts/judge/run_eval.py` | 평가 스크립트를 감싸 요청별 판정 기록(DECISION_LOG) 저장 | 팀 판정자 코드 그대로 |
| `scripts/judge/analyze_coldstart.py` | 판정 기록으로 콜드 천장 계산 (진단용) | 팀 판정자 코드 그대로 |
| `mvr-cache/vcache/vcache_policy/strategies/verified_splitter.py` | 후보 점수를 매길 때 합의 특징 수집, vCache 판정 뒤 판정자로 덮어쓰기 (`[판정자]` 표시) | 팀 코드의 판정 흐름을 이 저장소 정책(규칙 분할 · 학습 가중치)에 옮김 |
| `mvr-cache/benchmarks/eval_sembenchmark_verified_splitter.py` | `--collect-consensus` · `--consensus-model` · `--consensus-scope` · `--consensus-answer-threshold` | 새 옵션. 안 주면 원래 판정과 같음 |
| `scripts/judge/judge_fit.py` | 판정자 학습 · p* 고정 (add 예산 규칙, 여러 시드면 중앙값) | 새로 작성 (팀 기록 3~5의 규칙대로) |
| `scripts/judge/make_judge_cls_data.py` | 판정자 학습 스트림(train + val) · test 파일, 과제별 풀 문장 목록(`prefix_pool_task_flat.json`) | 새로 작성 |
| `scripts/judge/run_judge_prefix_cls.sh` · `jobs_judge_{core,l0,extra}.txt` | 판정자 고정 → test 실행 (조건 C · CJ · E · EJ) | 새로 작성 |
| `mvr-cache/benchmarks/prefix/prefix_stage2_report.py` | 조건 이름 CJ · EJ도 집계하도록 정규식만 넓힘 | 수정 |

## 5. 팀 판정자와 다른 점 (보고 시 함께 적기)

- 팀 실험의 분할은 앞 구두점 4곳(seg4) 또는 8곳(seg8)이고, 여기는 공통문단 실험과 같은 구두점 전체(상한 32, `cap_scope original`)다. 신빈 박 실험 5에서 seg4 · seg8 차이는 판정자 위에서도 유의하지 않았다.
- 팀 패치 안의 `fit_consensus_oof.py`는 문법 오류로 실행되지 않고, 최종 정책(add35 · Jmed)의 코드는 패치에 없어 `judge_fit.py`를 기록대로 새로 작성했다. 팀원의 최종 코드를 받으면 같은 판정 기록으로 p*를 비교해 맞춘다.
- 판정자는 단계 · 풀마다 그 단계 · 풀의 train + val로 따로 학습한다 (원본 판정자를 L8에 그대로 쓰면 분포가 달라 p*가 어긋날 수 있음).
- 판정자가 재사용으로 바꾼 요청은 LLM을 부르지 않아 관측 · 캐시 추가(⑤)가 없다. 그래서 vCache 자체 재사용이 일부 줄어든다 (팀 기록 4 · 5에서 32~69% 상쇄). 판정 기록의 이유별 건수로 함께 본다.
