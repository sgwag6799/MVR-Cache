# `judge-cap-overconfident` 브랜치 노트

main 대비 이 브랜치에서 추가한 문서를 이 파일 하나로 합쳤다. 위쪽이 이 브랜치에서 쓴 문서, 아래쪽이 부모 브랜치(`judge-prefix-classification`)에서 물려받은 문서다.
각 부분의 제목 앞 이름이 원래 파일 이름이고, 절 번호는 원래 문서 그대로다 (코드 주석의 "X 부분 N절"이 이 번호를 가리킨다).

| 부분 | 원래 파일 |
|---|---|
| HANDOFF_E | `HANDOFF_E.md` |
| JUDGE_CAP | `JUDGE_CAP.md` |
| JUDGE_PREFIX | `JUDGE_PREFIX.md` |
| CLAIM_A | `CLAIM_A.md` |
| exp8_stage2 README | `mvr-cache/benchmarks/prefix/exp8_stage2/README.md` |
| PREFIX_EXPERIMENT | `PREFIX_EXPERIMENT.md` |
| PREFIX_PLAN | `PREFIX_PLAN.md` |

---

<!-- 원래 파일: HANDOFF_E.md -->

# HANDOFF_E 부분: 판정자 실험 E — 진행 상황 · 이어서 할 일

> 세션을 이어갈 때 이 파일을 먼저 읽는다. 설계는 JUDGE_CAP 부분(실행 전 고정), 실행은 `notebooks/mvr_colab_judge_cap_E.ipynb`.

## 1. 지금까지 정해진 것

| 항목 | 내용 |
|---|---|
| 실험 | 판정자 실험 E: 과신하는 판정자에서 실행 중 한도(JC1L)가 오답을 막는가. 팀원이 맡긴 실험 |
| 기본 조건 (팀원 지시) | Jlow_s1 (p* = p_low, 한도 없음) · JlowC35_s1 (+ JC1L ε 0.35%) · JlowC10_s1 (+ JC1L ε 0.10%). Classification test41k, 공식 39K 집계, 시드 1, 기반 U |
| 재사용 기록 | U_s1 · Jmed_s1 · JC1L_s1 (팀원 기록, Drive "캡스톤디자인(1)" `mvr_eval`). **같은 GPU(L4)일 때만** 재사용, 다르면 이 GPU에서 다시 돌림 (+3실행) |
| 선택 조건 (제안, 팀원 확인 전) | Jover_s1 · JoverC35_s1: 판정자 확률을 부풀림 (logit + Δ). 이유: 한도의 구간 사전값이 1 − 구간 중앙이라 p*만 낮추면 라벨 없이도 낮은 구간이 닫혀, 라벨로 추정을 고치는 부분이 시험되지 않는다 |
| 코드 | 팀원과 같은 저장소 `PKU-SDS-lab/MVR-Cache@8cfaa0b` + 패치 3개 (`pair_judge` 07c4751c4be0 · `online_budget` b07856972ddc · `probe_label_only` 92ca6d07db0a). 패치 두 개는 업로드본으로 지문 일치 확인 |
| 바꾸는 인자 | 기존 기록 `_meta.json`의 인자에서 `--consensus-pstar` · `--consensus-online-budget` · (선택) `--consensus-model`만 |
| 한도 기록 | `policy_reason`에 `online_budget_blocked` · `online_budget_bin_closed` · `online_budget_probe`가 남는다 (기록 코드 추가 불필요). 구간별 상태는 `ONLINE_BUDGET_STATE` → `_budget.json` |
| label only 효과 | 짝(Jlow vs JlowC)끼리 캐시 경로가 같다 → 한도가 LLM으로 돌린 요청 중 오답 수 = 한도가 막은 오답 |

## 2. 확인한 자료

- `online_budget.patch` · `probe_label_only.patch`: 한도 동작 (구간 0.02, 사전 관측 20건, 사후 평균 + zσ, probe 10%, 허용 = 요청 수 × ε)
- SearchQueries 노트북 `mvr_colab_sq_judge.ipynb`: U 인자(seg8, `multivector_top_k`, `--mix-fullcos --include-full-embedding`), 판정 기록 필드(`i` · `hit` · `nn_correct` · `reason` · `has_nn` · `candidate_count` · `a_excl` · `s` · `policy_reason`), p는 기록에 없음 → `learned.json`으로 다시 계산
- `JC1_s1_meta.json`: **SearchQueries** 기록 (Tesla T4, ε 0.30%, label only 없음). 인자가 `calibrated` · z 1 · probe 0.1 · prior 20 · 폭 0.02 · `--consensus-seed 1`임을 확인. 실험 E의 GPU 비교 대상은 아니다
- 판정자 실험 A 노트북 일부: 작업 폴더 탐색, `REQUIRE_L4`("B · C · D · E 기준선과 맞춘다")

## 3. 아직 안 한 것 · 할 일

1. **SSH로 실행 (권장)**: 브라우저 노트북은 Drive 인증 오류(`credential propagation was unsuccessful`)가 나서 SSH 판으로 돌린다
   - 맥 터미널: `colab drivemount -s l4` (Drive 붙이기, 처음 한 번 인증) → `ssh colab-l4`
   - VM: `cd /content && git clone -b judge-cap-overconfident https://github.com/sgwag6799/MVR-Cache.git judge_cap_run && cd judge_cap_run && tmux new -s e`
   - `python3 -u scripts/judge_cap/run_E.py 2>&1 | tee -a /content/judge_e_run.log`
   - 설정은 환경변수: `P_LOW_OVERRIDE=0.5` · `RUN_OVERCONF=1` · `MD_OVERRIDE=...` · `DATA_OVERRIDE=...` · `BASE_META_U/JMED/JC1L=...`
2. (대안) Colab L4에서 노트북 "모두 실행"
   - 공유 드라이브 "캡스톤디자인(1)"을 내 드라이브에 바로가기로 추가해 둔다
   - 노트북이 멈추면 출력 메시지대로 `BASE_META_OVERRIDE` · `DATA_OVERRIDE` · `MD_OVERRIDE`에 경로를 넣는다
2. **팀원 확인**
   - p_low: 지시는 "예: 0.5". 노트북은 사전 점검 규칙(0.5 → 0.3 → 0.1 중 근사 추가 오답 ≥ 0.70%인 첫 값)으로 정한다. 팀원과 값을 정했으면 `P_LOW_OVERRIDE`
   - 선택 조건(Jover)을 돌릴지. 돌린다면 **사전 기록을 처음 쓰기 전에** `RUN_OVERCONF = True` (결과를 본 뒤 추가하지 않는다)
3. **결과 정리**: 노트북 §6 출력(판정 기준 ①~④)과 Drive `judge_e_cap/results_E.csv` · `record_E.json`을 가져와 보고한다

## 4. 이어서 진행할 때 Claude에게

- 이 파일 · JUDGE_CAP 부분을 읽고, 노트북 §6 · §7 출력이나 `results_E.csv` · `PREREG.json` · `record_E.json`을 올리면 결과 해석(설계 6절 표)부터 이어간다
- 노트북 · SSH 판을 고칠 때는 `scripts/judge_cap/gen_notebook_E.py`를 고치고 둘 다 다시 만든다 (`python scripts/judge_cap/gen_notebook_E.py notebooks/mvr_colab_judge_cap_E.ipynb && python scripts/judge_cap/gen_run_E.py notebooks/mvr_colab_judge_cap_E.ipynb scripts/judge_cap/run_E.py`)
- 데이터 · 결과 파일은 GitHub에 올리지 않는다

## 5. 확인하지 못한 것 (실행하면 알게 됨)

- Classification U_s1 · Jmed_s1 · JC1L_s1 기록의 실제 GPU · 파일 위치 · 이름 (노트북 §1이 찾아 출력)
- Jmed p* 값과 `learned.json` 위치 (§1이 Jmed_s1 인자에서 읽음)
- test 파일 이름 (§3이 JC1L_s1의 `--dataset` 이름으로 `mvr_splits`에서 찾음)
- 노트북은 문법 검사와 가짜 데이터로 §6 · §7 집계만 확인했다. Colab에서 실제로 돌려 본 적은 없다

---

<!-- 원래 파일: JUDGE_CAP.md -->

# JUDGE_CAP 부분: 판정자 실험 E: 한도 스트레스 — 과신하는 판정자에서 한도가 오류를 막는가

> 질문: 판정자를 일부러 과신하게 만들었을 때, 실행 중 한도(JC1L)가 판정자 추가 오답을 허용 한도 ε 안으로 막는가?

- 브랜치 `judge-cap-overconfident` (기반: `judge-prefix-classification`)
- 조건 이름(U · Jmed · JC1 · JC1L)과 용어(p · p* · 전환 · 추가 오답 · probe · 실행 중 한도 · δ · ε)는 팀원 문서 "판정자 변경한 부분" **0절**을 그대로 따른다
- 한도 동작은 `online_budget.patch`(b07856972ddc) · `probe_label_only.patch`(92ca6d07db0a) 코드 기준이다

## 0. 실행 조건

| 항목 | 내용 |
|---|---|
| 기반 | **U (Jmed 모델 그대로)로 확정** (2026-11-03). 실험 A를 기다리지 않는다. 이 실험의 질문(한도가 실제로 오답을 막는가)은 기반에 크게 달려 있지 않고, 비교에 쓸 기존 기록도 같은 기반이다 |
| 같은 GPU | 이 실험의 모든 조건과, 재사용할 **U_s1 · Jmed_s1 · JC1L_s1** 기록을 같은 GPU에서 돌린 것으로 맞춘다. 다른 GPU에서 돌리면 이 세 기록도 그 GPU에서 다시 돌린다 (3실행 추가). 이유: GPU 종류가 다르면 임베딩 계산의 소수점 끝자리가 달라져, 임계값 근처 요청의 판정이 바뀌고 그 뒤 캐시 경로가 갈라진다. 그러면 같은 시드라도 기존 기록과 요청 단위로 짝지어 비교할 수 없다 |
| 코드 | **팀원 노트북 방식 그대로**: 저장소 `8cfaa0b` + 패치 3개 (`pair_judge.patch` 07c4751c4be0 · `online_budget.patch` b07856972ddc · `probe_label_only.patch` 92ca6d07db0a). 작업 폴더는 공유 드라이브 "캡스톤디자인(1)"의 `mvr_eval` · `mvr_patches` · `mvr_splits`. 실험 E 노트북(`notebooks/mvr_colab_judge_cap_E.ipynb`)이 기존 기록(Jmed_s1 · JC1L_s1)의 `_meta.json` 인자를 그대로 읽어 **`--consensus-pstar` · `--consensus-model` · `--consensus-online-budget`만 바꾼다**. 한도 규칙 · 집계 코드는 고치지 않는다. 이 브랜치의 판정자 코드(공통문단 CJ · EJ용, 고정 p*만 있음)는 쓰지 않는다 |
| GPU | **L4** (팀원 노트북이 `REQUIRE_L4`로 강제. 판정자 실험 B · C · D · E 기준선과 맞춤). 실행마다 노트북의 `_meta.json`에 GPU · 커밋 · 패치 지문이 남는다 |

### JC1L 인자 (패치 기준, JC1L_s1의 `.cmd`로 확인)

```
U 인자 + --consensus-policy learned_all --consensus-model <learned.json> --consensus-pstar <p*> --consensus-top-k 10
        --consensus-online-budget 0.0035 --consensus-online-mode calibrated --consensus-online-z 1
        --consensus-online-label-only --consensus-seed 1
        (기본값 그대로: --consensus-online-bin-width 0.02 --consensus-online-prior 20 --consensus-online-probe 0.1)
```

- `--consensus-online-mode`의 기본값은 `cap`(비용 = 1 − p)이다. JC1 · JC1L은 `calibrated`여야 한다.
- `--consensus-seed`는 probe 추첨 난수(`consensus_seed + 7919`)를 정한다. 짝끼리 같아야 한다.
- 확인한 기록: SearchQueries `JC1_s1_meta.json`(Tesla T4, ε 0.30%)의 인자가 위와 같고 label only만 없다. 실험 E의 GPU 비교에는 Classification 기록을 쓴다.

## 1. 목적

- 하드 캡이 한 번도 막지 않았다.
  - JC1 Classification: 누적 비용 81 / 허용 144
  - JC1L Classification: 99 / 144
- 지금 근거는 사례 하나뿐이다: SearchQueries J(추가 오답 4.45%)를 JC1이 0.23~0.27%로 막았다 (한도 0.30%).
- 고정 p*의 과신은 반복된 실패다 (0절 3-3). 실패 방식은 두 가지다.
  - **전환 건수가 늘어남**: SearchQueries J(한도의 약 13배), 공통문단 CJ(1.5배)
  - **정밀도가 판정자 생각보다 낮음**: LmArena 사전 점검(1.3~1.6배, 준비 데이터 97~99% → test 초반 약 93%)
- 판정자를 일부러 과신하게 만들어, 한도가 이걸 막는지 직접 보인다.

## 2. 한도가 실제로 하는 일 (코드 기준)

| 단계 | 동작 |
|---|---|
| 구간 | p ≥ p*인 요청을 p*부터 0.02 폭으로 나눈다. Jmed p*(0.9539)면 3개, p* 0.5면 25개 |
| 구간 오답률 추정 | 사전값 = **1 − 구간 중앙**(판정자 확률을 그대로 믿음), 사전 관측 20건. 라벨이 쌓이면 갱신. 비용 = 사후 평균 + 1σ |
| 구간 열기 | 추정 오답률이 낮은 구간부터 "도착률 × 추정 오답률" 합이 ε에 찰 때까지 연다. 닫힌 구간 요청은 `online_budget_bin_closed` |
| 누적 한도 | 통과시킨 요청의 비용 합(최신 추정치로 다시 계산)이 "지금까지 요청 수 × ε"를 넘으면 `online_budget_blocked` |
| probe | 위 두 검사를 통과한 요청의 10%를 LLM으로 보낸다 (`online_budget_probe`) |
| 라벨 | 닫힘 · 막힘 · probe 요청 모두 LLM 응답으로 1등 후보가 맞았는지 알고, 구간 추정에 쓴다 |
| label only | 이 요청들은 vCache 관측 · 캐시 추가를 하지 않는다. **그래서 캐시 상태 경로가 한도 없는 판정자와 같다** |
| 기록 | 판정 기록의 `policy_reason`에 위 세 이유가 남고, 실행 끝에 구간별 상태(`online_budget`: arrivals · accepted · probed · label_n · label_err · est_err · admitted_final, spent · allowed)가 결과 JSON에 남는다 |

- 허용은 요청 수에 비례해 **실행 중 계속 늘어난다** (0.35%면 끝에서 약 144, 0.10%면 약 41). 한도는 "한 번 닿으면 끝"이 아니라 **전환 속도를 제한하는 장치**다.
- 그래서 "한도 때문에 전환하지 못한 요청"은 판정 기록에 따로 남는다. 기록 코드를 추가할 필요가 없다.

## 3. 설계 (실행 전 고정)

### 과신을 만드는 두 방법

| | 낮은 p* (p_low, 원래 지시) | 부풀린 확률 (Δ, 추가 제안) |
|---|---|---|
| 방법 | `--consensus-pstar p_low` | `learned.json` intercept + Δ → logit p' = logit p + Δ. p*는 Jmed 그대로 |
| 판정자 확률 | 그대로 (정직함) | 실제보다 높음 (과신) |
| 한도 사전값(1 − 구간 중앙) | **맞음**: 낮은 구간은 처음부터 오답률이 높다고 알고 있어, 라벨 없이도 닫힌다 | **틀림**: 위험한 요청이 안전한 구간에 들어온다. 한도가 probe · 막힘 라벨로 고쳐야 한다 |
| 닮은 실제 실패 | 전환 건수 증가 (SearchQueries · 공통문단) | 정밀도 하락 (LmArena) |
| 코드 변경 | 없음 | 없음 (모델 파일만 바꿈, `make_overconf_model.py`) |

- 낮은 p*만으로는 한도의 사전값이 이미 위험을 알기 때문에, **라벨로 추정을 고치는 부분(probe · 재계산)은 시험되지 않는다.**
- 부풀린 확률은 그 부분을 직접 시험한다.
- **기본은 원래 지시(낮은 p*) 3실행이다.** 부풀린 확률 2실행은 팀원 확인 후 선택으로 돌린다.

### 조건

| 실행 이름 | 판정자 모델 | p* | 실행 중 한도 ε | 실행 |
|---|---|---|---|---|
| U_s1 | 없음 | – | – | 재사용 (같은 GPU일 때) |
| Jmed_s1 | Jmed | Jmed p* | 없음 | 재사용 (같은 GPU일 때) |
| JC1L_s1 | Jmed | Jmed p* | 0.35% | 재사용 (같은 GPU일 때) |
| **Jlow_s1** | Jmed | **p_low** | 없음 | 새로 (원래 지시) |
| **JlowC35_s1** | Jmed | **p_low** | **0.35%** | 새로 (원래 지시) |
| **JlowC10_s1** | Jmed | **p_low** | **0.10%** | 새로 (원래 지시) |
| Jover_s1 | Jmed + **Δ** | Jmed p* | 없음 | 선택 (팀원 확인 후) |
| JoverC35_s1 | Jmed + **Δ** | Jmed p* | **0.35%** | 선택 (팀원 확인 후) |

- 데이터: Classification test41k, 공식 39K 집계 (앞 2,000행 제외). 시드 1. δ 0.01. 기반 U: 규칙 분할 seg8 + 균등 가중.
- 판정자는 다시 학습하지 않는다. 짝(Jlow vs JlowC, Jover vs JoverC)끼리는 **같은 모델 · 같은 p***이고, 차이는 한도뿐이다.
- 한도 조건은 모두 JC1L 인자 그대로다 (`calibrated`, z 1, probe 10%, label only).
- 원래 지시 3개를 먼저 돌린다. 선택 2개는 팀원이 동의하면 그다음에 돌린다.

### 과신 세기 고정 규칙 (실행 전에 정함)

- Jmed_s1 판정 기록과 Jmed `learned.json`으로 `scripts/judge_cap/precheck_plow.py`를 돌린다.
  - 판정 기록에는 p가 남지 않으므로 모델로 다시 계산한다.
  - 대상 요청은 팀원 노트북의 `eligible()`과 같다: reason cold · explore, has_nn, 후보 5개 이상, a_excl 있음.
  - 조건을 넘는 대상 요청을 모두 전환한다고 가정하고 추가 오답을 근사한다.
- **p_low**: 0.5 → 0.3 → 0.1 순으로 보고, 근사 추가 오답이 **ε의 2배(0.70%) 이상**인 첫 값을 쓴다.
- **Δ**: +1 → +2 → +3 → +4 순으로 보고, 같은 기준(0.70% 이상)을 넘는 첫 값을 쓴다. 점검 출력의 "판정자가 믿는 기대 오답"이 실제 전환 중 오답보다 충분히 작은지(사전값이 틀리는 조건인지) 함께 적는다.
- 2배로 잡는 이유는 두 가지다.
  - 근사는 잡아먹기를 반영하지 않아 실제보다 작다.
  - 한도 조건에서는 통과 요청의 10%가 probe로 빠진다.
- 정한 값과 점검 출력을 아래 칸에 적고 커밋한 뒤 실행한다.

| 항목 | 값 |
|---|---|
| p_low · 근사 추가 오답 | (실행 전 기록) |
| Δ · 근사 추가 오답 · 판정자가 믿는 기대 오답 | (실행 전 기록) |
| 점검에 쓴 기록 | Jmed_s1 판정 기록, `learned.json`, GPU · 커밋 (실행 전 기록) |

## 4. 지표

| 지표 | 정의 | 출처 |
|---|---|---|
| **추가 오답** (주 지표) | 전환 중 틀린 건수 ÷ 전체 요청 수 (0절 정의). 전환 = `hit`이고 `policy_reason`이 `policy_`로 시작 | 판정 기록 |
| **막은 오답** | 한도가 LLM으로 돌린 요청(`online_budget_blocked` · `bin_closed` · `probe`) 중 1등 후보가 틀린 건수. 이유별로 나눈다 | 판정 기록 |
| 히트율 · 오류율 | 공식 39K 집계. 오류율의 one-sided 95% Clopper–Pearson 상한을 함께 적는다. 명목 상한은 δ + ε (0.35%면 1.35%, 0.10%면 1.10%) | 판정 기록 |
| 한도 장부 | 끝의 spent / allowed, 구간별 도착 · 통과 · probe · 라벨 · 추정 오답률 | 결과 JSON `online_budget` |
| 실질 통과 임계값 | `admitted_final`이 참인 가장 낮은 구간 | 결과 JSON `online_budget` |
| 추정 vs 실제 | 구간별 추정 오답률(est_err)과 라벨 실측 오답률(label_err / label_n) | 결과 JSON `online_budget` |
| 캐시 경로 같음 확인 | 짝끼리 자체 재사용 수 · 최종 캐시 크기가 같은지 | 판정 기록 |
| 같은 오류율 히트율 (참고) | U δ 곡선(δ 0.01 · 0.0125 · 0.015 = 17.14 / 18.90 / 22.42%, 시드 1)을 각 조건의 실제 오류율에서 보간한 값과의 차이 (0절 3-7) | 새 실행 없음 |

- label only라서 짝끼리 캐시 상태 경로가 같다. 그래서 한도 조건에서 LLM으로 돌린 요청은 **한도 없는 짝에서 전환됐을 바로 그 요청들**이다.
- 따라서 "막은 오답"이 한도가 줄인 추가 오답을 그대로 보여준다. probe 몫과 한도 몫도 이유별로 정확히 나뉜다.

## 5. 판정 기준 (실행 전 기록)

1. **전제 (과신이 실제로 생김)**: 한도 없는 조건(Jlow_s1 · Jover_s1)의 추가 오답 > 0.35%. 성립하지 않는 짝은 "과신 유도 실패"로 기록하고 ②·③을 판정하지 않는다.
2. **주 기준 (한도가 오답을 막음)**: 판정자 추가 오답 ≤ ε × 1.2. 1.2는 추정 오차 여유다.
   - JlowC35_s1 · JoverC35_s1: 0.42% 이하
   - JlowC10_s1: 0.12% 이하
3. **보조 (막은 것이 probe가 아니라 한도 때문임)**: `online_budget_blocked` + `bin_closed`로 막은 오답이 0보다 크고, probe로 막은 오답보다 많다.
4. **유효성**: 짝끼리 자체 재사용 수 · 최종 캐시 크기가 같다 (label only 동작 확인). 다르면 원인을 먼저 기록한다.

## 6. 결과 해석 (실행 전 고정)

| 결과 | 해석 |
|---|---|
| Jlow 짝에서 ② 충족, 막은 오답의 대부분이 `bin_closed` | 한도가 판정자 확률을 근거로 낮은 구간을 닫았다. **전환 건수 증가형 과신**은 막는다. 라벨로 추정을 고치는 부분은 이 짝으로는 보이지 않는다 |
| Jover 짝에서 ② 충족, 추정 오답률이 실행 중 올라가며 `blocked`가 늘어남 | 사전값이 틀려도 probe · 막힘 라벨로 추정을 고쳐 막는다. **정밀도 하락형 과신**도 막는다. 가장 강한 근거 |
| Jover 짝에서 ② 초과 | 사전 관측 20건 · probe 10%로는 틀린 사전값을 제때 고치지 못한다. 초기 구간에서 이미 쓴 오답이 남는다 (Surplus의 장부 계수 g 같은 장치가 없는 한계) |
| 0.35%는 충족, 0.10%는 초과 | 허용이 작으면 추정 오차가 상대적으로 커진다. 작은 ε에서의 동작을 따로 보고한다 |
| ② 충족이지만 실질 통과 임계값이 Jmed p*보다 훨씬 높고 히트율이 JC1L_s1보다 크게 낮음 | 한도가 보수적이다. 오답은 막지만 히트율을 필요 이상으로 잃는다 |
| ① 불성립 | 과신 유도 실패. 그 짝은 판정 불가 |

## 7. 무료 집계

- 지금까지 모든 실행에서 "오류율 95% 상한 > 명목 상한"인 횟수를 센다. 새 실행은 필요 없다.
- 명목 상한은 조건에 따라 다르다 (0절 3-7).
  - 기반만 쓴 조건(U · vCache): δ
  - 판정자 조건: δ + ε. 참고로 δ 기준 횟수도 함께 적는다.
- 입력은 실행별 `name, n, wrong, delta` CSV이고, 도구는 `scripts/judge_cap/cp_upper_count.py`다.

## 8. 실행 순서

**SSH 판 `scripts/judge_cap/run_E.py`(노트북과 같은 코드, Drive는 `colab drivemount`)를 Colab L4 VM의 tmux 안에서 실행한다.** 브라우저에서는 노트북 `notebooks/mvr_colab_judge_cap_E.ipynb`를 "모두 실행"해도 같다. 끝난 단계는 Drive에 남아 건너뛰므로, 끊기면 다시 "모두 실행"하면 이어서 돈다.

1. **GPU 확인 (§0 · §1)**: L4가 아니면 멈춘다. U_s1 · Jmed_s1 · JC1L_s1 `_meta.json`을 Drive에서 찾아 GPU가 다르면 세 기록을 다시 돌릴 목록에 넣는다.
2. **JC1L 인자 확인 (§1)**: JC1L_s1 인자가 `calibrated` · z 1 · probe 0.1 · prior 20 · 폭 0.02 · label only · ε 0.35%인지, Jmed와 같은 모델 · p*인지 확인한다. 다르면 멈춘다.
3. **저장소 · 패치 · 데이터 (§2 · §3)**: 패치 지문, test 행 수(= 기존 판정 기록 행 수)를 확인한다.
4. **사전 기록 (§4)**: `precheck_plow.py`로 p_low(선택이면 Δ도)를 정해 Drive `judge_e_cap/PREREG.json`에 쓴다. 있으면 읽기만 한다. 팀원과 값을 정했으면 `P_LOW_OVERRIDE`.
5. **실행 (§5)**: JC1L_s1 · Jmed_s1 인자에서 p* · ε · (선택) 모델만 바꿔 순서대로 돌린다. 한도 상태는 `ONLINE_BUDGET_STATE`로 `_budget.json`에 남는다.
   1. Jlow_s1
   2. JlowC35_s1
   3. JlowC10_s1
   4. (선택) Jover_s1
   5. (선택) JoverC35_s1
6. **집계 (§6 · §7)**: 4절 표와 5절 기준을 출력하고 `results_E.csv` · `free_agg_cp_upper.csv` · `record_E.json`을 Drive에 남긴다.

## 9. 비용

- 기본: 원래 지시 3실행 약 3.3시간 (L4).
- 선택: 부풀린 확률 2실행 약 2.2시간.
- 기존 기록과 GPU가 다르면 3실행(약 3.3시간)이 추가된다.

## 10. 한계 (실행 전 공개)

- 기반이 vCache가 아니라 U라서, "한도가 오답을 막는다"는 결과는 **U 위 판정자**에 대한 것이다. 한도 규칙 자체는 기반과 독립적이라 실험 A의 기반(분할 0)에서도 같은 동작을 기대하지만, *추정*이다.
- 시드 1개다.
- 구간 비용(사후 평균 + 1σ)은 정규근사라 유한 표본 보장이 아니다. 논문에서는 **형식적 보장이 없는 경험적 한도**로 서술한다.
  - 구조는 Conformal Risk Control · Learn-then-Test와 비슷하고, 동시기 연구 Surplus(arXiv 2611.02498v1)와는 거의 같다.
  - *"기존 위험 통제 기법의 적용" · "보장"이라는 표현은 쓰지 않는다* (2026-11-03 정정).
  - 여력이 있으면 LTT를 기준선으로 함께 잰다.
- 참고: Surplus는 장부 계수 g로 "쓴 오답이 예산을 앞지르면 구간을 닫는" 장치를 두고, 분포 이동 스트림(ClassificationSorted)에서 이를 시험했다. 우리 한도에는 이 장치가 없다.
- 두 과신 모두 인위적으로 만든 것이다. p_low는 실제 실패 중 전환 건수 증가형과, Δ는 정밀도 하락형과 닮았지만 원인이 같지는 않다. 실제 사례에서의 동작은 판정자 실험 G(공통문단 L8 + 실행 중 한도)에서 본다.
- Δ는 모든 요청의 logit을 똑같이 올린다. 실제 과신은 구간마다 다르게 어긋날 수 있다.
- 명목 상한은 δ + ε다. 한도가 막는 것은 판정자 추가 오답(ε)이고, vCache 자체 오답(δ)은 별도다.

## 11. 파일

| 파일 | 역할 |
|---|---|
| JUDGE_CAP 부분 | 이 설계 문서 |
| `scripts/judge_cap/precheck_plow.py` | 실행 전 과신 세기 고정 (p_low · Δ). Jmed_s1 판정 기록 + `learned.json`으로 p를 다시 계산해 추가 오답 근사 |
| `scripts/judge_cap/make_overconf_model.py` | Jmed `learned.json`의 intercept에 Δ를 더한 과신 모델 |
| `scripts/judge_cap/cp_upper_count.py` | 무료 집계: 실행별 오류율 95% CP 상한 > 명목 상한 횟수 |
| `notebooks/mvr_colab_judge_cap_E.ipynb` | 실험 E 실행 노트북 (Colab L4, "모두 실행", 끊기면 이어서) |
| `scripts/judge_cap/run_E.py` · `gen_run_E.py` | SSH 판 실행 스크립트 (노트북 코드 셀에서 생성) |
| HANDOFF_E 부분 | 진행 상황 · 다음에 할 일 (세션을 이어갈 때 먼저 읽는다) |
| `scripts/judge_cap/run_meta.sh` | 팀원 노트북 밖에서 돌릴 때만 GPU · 커밋 기록 (노트북은 `_meta.json`에 자동 기록) |

### 팀원에게 받아야 하는 것

- 노트북이 Drive "캡스톤디자인(1)"에서 U_s1 · Jmed_s1 · JC1L_s1의 `_meta.json` · `.jsonl`, Jmed `learned.json`, 패치, test 파일을 자동으로 찾는다. **공유 드라이브 접근 권한**만 있으면 된다
- 자동 탐색이 실패하면 노트북 §0의 `BASE_META_OVERRIDE` · `DATA_OVERRIDE`에 경로를 넣는다
- 팀원 확인: p_low 값(정했으면 `P_LOW_OVERRIDE`), 선택 조건 Jover · JoverC35를 돌릴지(`RUN_OVERCONF`)

---

<!-- 원래 파일: JUDGE_PREFIX.md -->

# JUDGE_PREFIX 부분: 실험 3: 공통 문단이 붙은 Classification에서 판정자 확인

> 질문: 후보 답 합의 판정자(learned_all_add35)는 프롬프트 앞에 공통 문단(템플릿)이 붙어도 원본에서처럼 같은 오류율로 히트율을 올리는가?

- 브랜치 `judge-prefix-classification` (기반: `claim-a-classification`)
- 데이터 · 공통 문단 규칙: PREFIX_EXPERIMENT 부분, 기준선 · 스트림 설정: CLAIM_A 부분의 Classification 실행
- 판정자: 팀 판정자 코드(`code_changes.patch`)의 합의 함수와 팀 실험 기록 2~5의 정책(learned_all_add35)을 이 저장소 정책 코드에 옮김
- 결과: 이 부분에는 실행 결과가 기록되지 않았다. 실험 3 결과 요약(공통문단 L8에서 CJ 히트율 25.24% · 오류율 1.12%, p* 0.880)은 `judge-prefix-online-cap` 브랜치 BRANCH_NOTES.md의 JUDGE_G 부분 1절에 인용돼 있다

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

---

<!-- 원래 파일: CLAIM_A.md -->

# CLAIM_A 부분: 주장 A: 공통 문단 환경에서 규칙 분할 + 학습 가중치가 vCache · MVR-Cache보다 낫다

> 프롬프트 앞에 답과 무관한 공통 문단이 붙는 환경에서, **규칙 분할(구두점 위치, 상한 32) + 학습 가중치(M3)** 를 쓰는 MVR이 단일 벡터 vCache와 논문 MVR-Cache(RL 분할기)보다 **오류율 한도(δ = 0.01)를 지키면서 더 높은 히트율**을 낸다.

- 인코더: 모든 비교는 base 인코더 `BAAI/bge-base-en-v1.5` 기준
- 데이터셋 (공통 문단 붙이는 방식): PREFIX_EXPERIMENT 부분, 설계 원문 PREFIX_PLAN 부분
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

## Classification 실행 (브랜치 `claim-a-classification`)

LmArena와 같은 방식으로 Classification 41K 스트림에서 잰다.

| 항목 | 설정 |
|---|---|
| 데이터 · 정답 | Classification test 41,000행 중 `fits_all_levels` 38,304행 (앞 2,000행 제외 집계), 8B 응답 문자열 일치 |
| 조건 | A vCache · C 규칙 분할 + 균등 · **E 규칙 분할 + 학습 가중치** · J 논문 MVR-Cache (RL 분할기 + `multivector_top_k` k 10, `--mix-fullcos` · `--include-full-embedding` 없음) |
| E의 헤드 | `stage1_out/prefix_w_variant_L8.pt` (Classification L8 train, 8B 라벨, 20 에폭). 모든 단계에 같은 헤드 |
| RL 분할기 | Classification train(L0)으로 논문 레시피 그대로 학습, 정답만 8B 응답 문자열 일치 (`scripts/train_rl_splitter_cls.sh`) |
| 판정 | δ = 0.01, `--sleep 0.02` (LmArena 실험 8과 같음) |
| 보고 | 전체 히트율 · 오류율과 함께 **하위 과제별**(ecommerce · amazon · commonsense) 히트율 · 오류율. 전체 흔들림이 대부분 ecommerce에서 나오므로 효과가 한 과제에 몰려 있는지 확인 |
| **핵심 배치 (우선)** | `stage2_jobs_cls_core.txt` 12개: ① C · E × L8 시드 1 · 2 · 3 → ② A × L8 시드 1 · 2 · 3 → ③ A · C · E × L0 시드 1. RL 분할기(J)가 필요 없어 바로 실행 가능. 결과는 main 배치 폴더에 저장되어 main을 돌리면 건너뜀 |
| 핵심 배치 판정 기준 (실행 전 고정) | ① L8에서 E − C > 0이 시드 3개 모두 ② 모든 실행의 오류율 ≤ δ ③ L0에서 E가 C보다 0.5%p 넘게 낮지 않음. 하위 과제별 · 누수 1,534행 제외 집계를 함께 보고. 1단계 사전 기준 미달 뒤 추가한 확인이라 보고 시 그렇게 표시 |
| 배치 | ① main: A · C · E · J × (L8 시드 1 · 2 · 3, L0 시드 1) 16개 → ② sleep01: `--sleep 0.1`로 E · J × L8 시드 1 · 2 · 3, A · C × L8 시드 1, 8개 (sleep 민감도 확인) → ③ capall: C · E × (L8 시드 1 · 2 · 3, L0 시드 1)을 `cap_scope all`로 8개 |

```bash
git clone -b claim-a-classification https://github.com/sgwag6799/MVR-Cache.git /root/MVR-Cache
cd /root/MVR-Cache && source colab_setup.sh
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkClassification','train.parquet',repo_type='dataset',local_dir='data/raw')"
python scripts/make_cls_splits.py
(cd mvr-cache && python -m benchmarks.prefix.make_prefix_data --train ../data/cls_seed42/train.parquet \
  --val ../data/cls_seed42/val.parquet --test ../data/cls_seed42_nodedup/test41k.parquet --out ../data/prefix/data)
python scripts/make_stage2_cls_data.py
# 핵심 배치만 먼저 (L4 1회 약 1시간, 병렬 3이면 약 4~5시간):
SLEEP=0.02 CAP_SCOPE=original OUT=data/cls_stage2/main bash scripts/run_stage2_cls.sh scripts/stage2_jobs_cls_core.txt 3
bash scripts/train_rl_splitter_cls.sh            # → data/cls_stage2/rl/best.ckpt (조건 J에 필요)
nohup bash scripts/run_claimA_cls.sh 5 > data/cls_stage2/run.log 2>&1 < /dev/null &
#   결과: mvr-cache/benchmarks/prefix/claimA_cls/{main,sleep01,capall}/ (실행 중 자동 push)
#     stage2_summary.csv (전체) · stage2_summary_by_task.csv (하위 과제별: ecommerce · amazon · commonsense)
```

## 파일

| 파일 | 역할 |
|---|---|
| `scripts/make_lmarena_splits.py` · `scripts/lmarena_merge_duplicate_sets.py` | LmArena 분할(지문 train `ec6e778ee76f` / val `411c8e1b6412` / test `3e5cc1fe8fa8`) · 합친 정답 `id_set_merged` |
| `mvr-cache/benchmarks/prefix/make_prefix_data.py` · `prefix_pool.json` | 공통 문단 데이터 L0 · L2 · L4 · L8 |
| `scripts/make_stage2_lmarena_data.py` | 스트림(L0 · L8 test) · RL 분할기 train/val 파일 |
| `scripts/train_rl_splitter_lmarena.sh` | MVR-Cache 기준선 RL 분할기 학습 (논문 레시피) |
| `mvr-cache/benchmarks/prefix/exp2_lmarena/offline_merged/prefix_w_variant_L8.pt` | E의 M3 가중치 헤드 (LmArena L8 train으로 학습) |
| `mvr-cache/benchmarks/prefix/stage1_out/prefix_w_variant_L{0,2,4,8}.pt` | Classification M3 가중치 헤드 (단계별, 8B 라벨). E는 L8 헤드를 씀 |
| `scripts/make_cls_splits.py` · `scripts/make_stage2_cls_data.py` · `scripts/train_rl_splitter_cls.sh` · `scripts/run_stage2_cls.sh` · `scripts/run_claimA_cls.sh` | Classification 분할 · 스트림 데이터 · RL 분할기 · 실행 |
| `scripts/run_stage2_lmarena.sh` · `scripts/stage2_jobs_*.txt` | 스트림 실행 (조건 A~G · J, `SLEEP` · `CAP_SCOPE` · `HEAD` · `OUT` 환경변수) |
| `scripts/run_claimA_stage2.sh` · `scripts/autosave_stage2.sh` | 보강 배치를 순서대로 실행 · 결과 자동 push |
| `mvr-cache/benchmarks/prefix/prefix_stage2_report.py` | 히트율 · 오류율 집계 |
| `mvr-cache/vcache/vcache_core/splitter/PrefixRuleSplitter.py` | 규칙 분할기 + M3 헤드 |

데이터 파일(`data/`)은 git에서 제외되어 있다. 위 스크립트로 Hugging Face 원본(`vCache/SemBenchmarkLmArena`)에서 같은 파일을 다시 만든다. 데이터 만드는 방법은 PREFIX_EXPERIMENT 부분, 실험 실행은 `scripts/run_claimA_stage2.sh`.

---

<!-- 원래 파일: mvr-cache/benchmarks/prefix/exp8_stage2/README.md -->

# exp8_stage2 README 부분: 실험 8 결과 (2단계 41K 스트림, LmArena)

설계 · 결과 · 판정: CLAIM_A 부분.

| 파일 | 내용 |
|---|---|
| `stage2_runs.csv` | 실행별 히트율 · 오류율 · 정밀도 (조건 A~G × 단계 × 시드, 앞 2,000행 제외 38,964요청) |
| `stage2_summary.csv` | 조건 × 단계 평균 · 표준편차 · 오류율 최대 |
| `claim_a_L8_comparison.csv` | 주장 A: L8에서 E(규칙 분할 + 학습 가중치)와 A · F · C의 시드별 차이 |

- `stage2_runs.csv`는 Colab VM에서 `prefix_stage2_report.py`가 출력한 표(소수 6자리)에서 복원했다. 원본 실행 JSON · timing 파일 · RL 학습 로그는 VM과 함께 없어졌다.
- `stage2_summary.csv` · `claim_a_L8_comparison.csv`는 `stage2_runs.csv`에서 계산했다.
- 조건: A/B vCache base/LoRA, C/D 규칙 분할 균등 base/LoRA, E 규칙 분할 + 학습 가중치(M3) base, F/G 논문 MVR-Cache(재학습 RL 분할기) base/LoRA.

---

<!-- 원래 파일: PREFIX_EXPERIMENT.md -->

# PREFIX_EXPERIMENT 부분: 공통 문단 데이터셋 (LmArena · Classification)

LmArena · Classification 프롬프트 앞에 **답과 무관한 공통 문장**을 붙여서, "긴 공통 템플릿 + 짧은 결정 부분" 상황을 만든 데이터셋이다. 실험 설계 · 결과는 CLAIM_A 부분에 있다.

## 1. 원본

| 항목 | 내용 |
|---|---|
| 데이터 | `vCache/SemBenchmarkLmArena` (Hugging Face) 63,796행 |
| 정답 | 같은 뜻의 질문 묶음이면 정답. 원본 묶음 번호(`ID_Set`)에서, 글자까지 같은 질문이 서로 다른 묶음에 들어간 경우(186개)를 하나로 합친 `id_set_merged`를 쓴다 |
| 분할 | 묶음 단위로 나눔 (같은 묶음은 한 분할에만): train 3,007 / val 1,000 / test 40,964행 |

### Classification

| 항목 | 내용 |
|---|---|
| 데이터 | `vCache/SemBenchmarkClassification` (Hugging Face). 하위 과제 3개: ecommerce · amazon_instant_video · commonsense_qa |
| 정답 | 두 질문의 Llama 3 8B 응답(`response_llama_3_8b`)이 같으면 정답 (공백 · 끝 마침표 · 따옴표 제거, 소문자로 맞춘 뒤 비교) |
| 분할 | 팀 분할 cls_seed42 (`scripts/make_cls_splits.py`, `df.sample(frac=1, random_state=42)`): train 3,000 / val 1,000 / test 41,000. 지문 `35365cc8e27f` / `a472edba40b2` / `f215310197be` |
| 사용 행 | 모든 단계에서 512토큰 안에 드는 행: train 2,791 / val 935 / test 38,304 |

## 2. 공통 문단을 붙이는 방식

### 공통 문장 풀 (`mvr-cache/benchmarks/prefix/prefix_pool.json`)

칸 8개 × 변형 5개 = 40문장. 모두 영어 한 문장, 15~20토큰, 구두점은 끝 마침표 하나다 (그래서 문장 하나가 조각 하나가 된다).

| 칸 | 변형 0번 문장 (예) |
|---|---|
| role | You are a helpful assistant that answers questions for users of an online service. |
| tone | Please keep your answer short and polite and avoid using any unnecessary words in the reply. |
| format | Return only the final answer without adding any additional text or special formatting to it. |
| safety | Do not share harmful content and politely refuse any requests that could cause real damage. |
| privacy | Do not ask for or reveal personal information such as names or home addresses. |
| language | Reply in English even if the user writes the request in a different language. |
| accuracy | Base the answer only on the given input and do not guess any missing facts. |
| session | This conversation is part of an automated evaluation run that is performed on the service. |

### 붙이는 규칙

1. 프롬프트마다 칸별로 변형 5개 중 하나를 고른다. 고르는 난수는 (시드 42, `dataset_name:id`)로 정해서, 같은 행은 항상 같은 문장을 받는다.
2. 단계 **L**이면 앞에서부터 L칸의 문장을 쓴다. 따라서 L2의 문장은 L8의 앞 두 문장과 같다.
3. `공통 문장들(공백으로 연결) + " " + 원본 프롬프트`로 붙인다. 원본 프롬프트와 정답은 바꾸지 않는다.
4. 가장 긴 단계(L8)에서도 512토큰 안에 드는 행만 모든 단계에서 똑같이 쓴다 (`fits_all_levels`).

### 단계

| 단계 | 내용 | 평균 길이 (test) |
|---|---|---|
| **L0** | 원본 그대로 | LmArena 47토큰 · Classification 133토큰 |
| L2 · L4 | 공통 문장 2 · 4개 | – |
| **L8** | 공통 문장 8개 (공통 문단 약 129토큰) | LmArena 176토큰 · Classification 262토큰 |
| 고정 L4 | 모든 프롬프트에 칸마다 0번 문장 (참고용) | – |

### 예시 (L8, 칸마다 0번 문장이 뽑힌 경우)

```
You are a helpful assistant that answers questions for users of an online service. Please keep your
answer short and polite and avoid using any unnecessary words in the reply. Return only the final answer
without adding any additional text or special formatting to it. Do not share harmful content and
politely refuse any requests that could cause real damage. Do not ask for or reveal personal information
such as names or home addresses. Reply in English even if the user writes the request in a different
language. Base the answer only on the given input and do not guess any missing facts. This conversation
is part of an automated evaluation run that is performed on the service. How many sisters does each of
Alice's brothers have?
```

## 3. 만드는 방법

GPU는 필요 없다 (토크나이저만 사용, 수 분).

```bash
git clone -b claim-a-rule-weighting https://github.com/sgwag6799/MVR-Cache.git && cd MVR-Cache
source colab_setup.sh
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkLmArena','train.parquet',repo_type='dataset',local_dir='data/raw_lmarena')"
python scripts/make_lmarena_splits.py --raw data/raw_lmarena/train.parquet --out data/lmarena
(cd mvr-cache && python -m benchmarks.prefix.make_prefix_data --train ../data/lmarena/train.parquet \
  --val ../data/lmarena/val.parquet --test ../data/lmarena/test.parquet --out ../data/lmarena/prefix/data)
python scripts/lmarena_merge_duplicate_sets.py --raw data/raw_lmarena/train.parquet --data data/lmarena/prefix/data
python scripts/make_stage2_lmarena_data.py    # 캐시 스트림용 L0 · L8 test 파일 (data/exp8/stream/)
```

Classification:

```bash
python -c "from huggingface_hub import hf_hub_download as d; d('vCache/SemBenchmarkClassification','train.parquet',repo_type='dataset',local_dir='data/raw')"
python scripts/make_cls_splits.py                 # 팀 분할 재현 + 지문 검사
(cd mvr-cache && python -m benchmarks.prefix.make_prefix_data --train ../data/cls_seed42/train.parquet \
  --val ../data/cls_seed42/val.parquet --test ../data/cls_seed42_nodedup/test41k.parquet --out ../data/prefix/data)
python scripts/make_stage2_cls_data.py            # 캐시 스트림용 L0 · L8 test 파일 (data/cls_stage2/stream/)
```

LmArena 분할이 같은지는 `make_lmarena_splits.py`가 출력하는 지문으로 확인한다: train `ec6e778ee76f` / val `411c8e1b6412` / test `3e5cc1fe8fa8`. Classification은 `make_cls_splits.py`가 지문을 검사한다.

## 4. 결과 파일

| 파일 | 내용 |
|---|---|
| `data/lmarena/prefix/data/variant/L{0,2,4,8}/{train,val,test}.parquet` | 단계별 데이터 |
| `data/lmarena/prefix/data/fixed/L4/{train,val,test}.parquet` | 고정 방식 (참고용) |
| `data/exp8/stream/L{0,8}_test.parquet` | 캐시 스트림 입력 (`id_set` = `id_set_merged`) |
| `data/prefix/data/variant/L{0,2,4,8}/{train,val,test}.parquet` | Classification 단계별 데이터 |
| `data/cls_stage2/stream/L{0,8}_test.parquet` | Classification 캐시 스트림 입력 |

원본 열은 그대로이고, 다음 열이 추가된다: `orig_prompt`(원본 프롬프트), `prefix_text`(붙인 공통 문단), `prefix_choice`(칸별로 고른 변형 번호), `level`, `n_tokens`, `truncated`, `fits_all_levels`, `id_set_merged`(LmArena).

---

<!-- 원래 파일: PREFIX_PLAN.md -->

# PREFIX_PLAN 부분: 공통 문단 테스트: 조각 가중치가 필요한 조건 찾기 (실험 계획)

> 🎯 **한 줄 요약**: Classification 프롬프트 앞에 답과 무관한 공통 문단을 붙여 "긴 공통 템플릿 + 짧은 결정 부분" 상황을 만들고, 공통 문단이 길어질수록 vCache · 균등 MVR · 학습 가중치의 구분력과 히트율이 어떻게 변하는지 측정한다.
>
> 기대하는 바는 vCache와 MVR-Cache에서 수치 하락, 학습 가중치에서 수치 상승.

- 이 문서는 결과를 보기 전에 고정한 **설계 원문**이다.
- 코드로 옮긴 방법, 데이터로 확인한 설계 충돌, 실행 명령은 PREFIX_EXPERIMENT 부분에 있다.

## 목적

- 기존 데이터셋 3개에서는 조각 가중치의 효과가 작았다 (오라클 천장). 이 실험은 **가중치가 효과를 내는 조건이 존재하는지**를 통제된 환경에서 확인한다.
- 실제 LLM 서비스는 질문 앞에 시스템 프롬프트 · 안내문 · 예시를 붙인다. 이 공통 부분이 길수록 문장 벡터와 균등 평균은 공통 부분에 지배된다.
- 가설: 공통 문단이 길어질수록 vCache와 균등 MVR의 구분력이 떨어지고, 학습 가중치는 공통 조각의 비중을 낮춰 구분력을 되찾는다.

## 보고 위치

- **스트레스 테스트(통제 실험)**로 보고한다. "가중치가 효과 있다"의 주 근거가 아니라 "가중치는 어떤 조건에서 필요한가"의 근거다.
- 기존 데이터셋 3개 결과(효과 작음)와 함께 제시한다.
- 데이터 구성 규칙은 결과를 보기 전에 이 문서대로 고정한다.

## 데이터 구성

### 원본

- Classification (train3k / val1k / test41k, seed 42), 하위 과제 ecommerce · amazon · commonsense
- 라벨: 원본 그대로 (`response_llama_3_8b` 문자열 일치). 공통 문단은 답에 영향을 주지 않는 문장으로만 구성한다는 가정을 둔다.

### 공통 문단

- 영어 한 문장 단위, 문장당 15~20토큰, 문장 안에 구두점은 마지막 마침표 하나만
- 답과 무관한 문장만 사용: 역할 소개, 말투, 형식, 안전 안내 등 (예: "You are a helpful assistant for an online store." · "Please keep your answer short and polite.")
- 문장 칸(slot) 8개, 칸마다 바꿔 쓴 문장 5개 → 공통 문장 풀 40개 (`prefix_pool.json`, 실험 전 고정)
- **변형 방식**: 프롬프트마다 칸별로 5개 중 하나를 고정 시드로 뽑음. 공통 부분이 요청마다 조금씩 달라지는 실제 상황(사용자 정보, 대화 이력)을 흉내 내고, "공통 부분을 문자열로 잘라내면 된다"는 대안을 막는다.
- 고정 방식(모든 프롬프트에 같은 문장)은 1단계에서만 참고로 측정

### 길이 단계

| 단계 | 붙이는 공통 문장 수 | 추가 토큰 (약) |
| --- | --- | --- |
| L0 | 0 (원본) | 0 |
| L2 | 2 | 30~40 |
| L4 | 4 | 60~80 |
| L8 | 8 | 120~160 |

### 길이 제한

- bge-base-en-v1.5 최대 입력 512토큰. 넘으면 뒤쪽 원본 프롬프트가 잘린다.
- 단계마다 (공통 문단 + 원본) 토큰 수를 전수 계산해 **잘림 0건**을 확인한다. 잘림이 생기는 단계는 실험에서 제외한다.

## 비교 조건

| 조건 | 점수 계산 | 역할 |
| --- | --- | --- |
| R | 원본(L0)의 균등 MVR | 공통 문단이 없을 때의 기준 |
| M1 | vCache (문장 벡터 1개, CLS 포함 토큰 평균) | 단일 벡터 |
| M2 | MVR 균등 가중치 (논문 식 7) | 멀티벡터 기준 |
| M3 | MVR 학습 가중치 (Linear 768 → 1 + 조각끼리 softmax, 0 초기화) | 제안 |
| M4 | MVR, 공통 문단 조각 가중치 0 (어느 조각이 공통인지 아는 규칙) | 가중치로 얻을 수 있는 상한 |

- M3 학습: 오라클 천장 실험과 같은 절차 (BCE, 클래스 비율 보정, Adam 0.001, 배치 2,048, 20 에폭, val AUC 최고 시점). **단계마다** 그 단계의 train3k로 따로 학습
- M3은 어느 조각이 공통 문단인지 정보를 받지 않는다. 조각 벡터만 보고 배운다.

## 고정 설정

| 항목 | 값 |
| --- | --- |
| 분할 | 모든 구두점 위치 분할 (상한 32). 공통 문장은 항상 각각 한 조각이 되고, 원본 부분의 분할은 단계와 무관하게 같다 |
| 후보 검색 | 문장 벡터 1개로 HNSW 후보 10개 |
| 점수 | SMaxSim, (x+1)/2 변환, 전체 벡터 행 미사용, 전체 코사인 혼합 없음 |
| 판정 | vCache, δ 0.01, cache-on-miss, sleep 0.02 (판정자 미사용) |
| 임베딩 · 라벨 | bge-base-en-v1.5 · Llama 3 8B |
| 환경 | Colab L4, MVR-Cache 저장소 + 팀 패치 (커밋 해시 기록) |

## 진행 단계

### 0단계: 데이터 만들기 · 점검

1. `prefix_pool.json` 작성, 단계별 데이터 생성 (시드 고정)
2. 점검: 단계별 토큰 수 분포와 잘림 0건, 조각 수 분포, 원본 부분 분할이 L0와 같은지

### 1단계: 오프라인 진단 (쌍 단위)

- 평가 쌍: test41k 앵커 6,000 × 문장 벡터 최근접 10 = 6만 쌍 (단계마다 다시 검색)
- 단계(L0 · L2 · L4 · L8) × 조건(M1 · M2 · M3 · M4)마다 측정
    - 검색 천장@10: 후보 10개 안에 정답 후보가 있는 비율
    - 1등 정답률: 점수 1등 후보가 정답인 비율
    - 선택된 이웃 (s, c)의 AUC
    - TPR@FPR1% (팀 쌍 단위 대리 지표)
    - 하위 과제별 같은 지표
- 고정 방식 공통 문단(L4)도 같은 지표로 1회 측정

### 2단계: 41K 스트림 (1단계 통과 시)

- 단계: L0 + 1단계에서 통과한 단계 중 가장 긴 것 1개
- 조건: M1 · M2 · M3, 시드 3개 (같은 시드끼리 짝지어 비교)
- 공식 집계 39,000행 (앞 2,000행 제외)
- 지표
    - 히트율, 오류율(≤ δ 확인), 검색 천장, 과제별 히트율
    - 지연시간: 임베딩 · HNSW 검색 · SMaxSim 재순위 · 가중치 계산 · vCache 판정 (요청당 ms)
    - 저장 벡터 수, 메모리

## 통과 기준 (1단계 → 2단계)

- 어떤 단계 L(≥ L4)에서 M2의 AUC가 L0 대비 **0.03 이상 하락**
- 그 단계에서 M3의 회복률 = (M3 − M2) / (R − M2)가 **0.5 이상** (AUC 기준)
- 둘 다 만족하는 단계가 없으면 2단계를 진행하지 않고 1단계 결과를 최종 결과로 기록

## 결과 해석 (사전 고정)

| 결과 | 해석 |
| --- | --- |
| M2가 단계가 올라가도 떨어지지 않음 | 문맥 임베딩이 공통 문단 영향을 흡수함. 이 조건에서도 가중치는 필요 없음 (음성 결과) |
| M2 하락, M3 회복, M3 ≈ M4 | 학습 가중치가 공통 조각을 스스로 낮춤. "긴 공통 템플릿 환경에서 가중치가 필요하다"는 조건부 양성 결과 |
| M2 하락, M3 일부 회복, M3 < M4 | 가중치로 얻을 수 있는 이득 중 일부만 학습됨. 모델 구조 · 학습 데이터 한계 |
| M2 하락, M4도 회복 못 함, 검색 천장 하락 | 병목이 후보 검색 단계(문장 벡터 1개)에 있음. 재순위 가중치로는 해결 불가 |
| 1단계 회복이 2단계 히트율로 이어지지 않음 | 판정 단계(cold 규칙, 경로 의존)가 이득을 막음 |

## 한계 (보고 시 함께 명시)

- 공통 문단은 답에 영향을 주지 않는다고 가정하고 원본 라벨을 그대로 쓴다. 실제 LLM은 시스템 프롬프트에 따라 답이 달라질 수 있다.
- 인공적으로 구성한 데이터라 실제 서비스 분포와 다르다.
- 실제 시스템은 시스템 프롬프트를 캐시 키에서 빼는 경우가 많다. 이 실험은 공통 부분이 요청마다 조금씩 달라 문자열로 분리할 수 없는 경우를 대상으로 한다.

## 산출물

- `prefix_pool.json`: 공통 문장 풀 (칸 8개 × 5개)
- `cls_prefix_L{0,2,4,8}.parquet`: 단계별 데이터
- `prefix_sanity.json`: 토큰 수 · 잘림 · 조각 수 점검
- `prefix_offline.csv`: 단계 × 조건 × 과제별 1단계 지표
- `prefix_w_L{2,4,8}.pt`: 단계별 학습 가중치
- (2단계) 시드별 41K 결과, 지연시간 표, `prefix_run_meta.json`

## 일정

- 2026-10-19 → 2026-10-23: 0단계 데이터 생성 · 점검
- 2026-10-26 → 2026-10-30: 1단계 오프라인 진단, 통과 여부 결정
- 2026-11-02 → 2026-11-13: 2단계 41K 3시드 (통과 시)
