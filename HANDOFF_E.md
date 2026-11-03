# 판정자 실험 E — 진행 상황 · 이어서 할 일

> 세션을 이어갈 때 이 파일을 먼저 읽는다. 설계는 `JUDGE_CAP.md`(실행 전 고정), 실행은 `notebooks/mvr_colab_judge_cap_E.ipynb`.

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

1. **Colab L4에서 노트북 "모두 실행"** (컴퓨팅 단위 구매 후)
   - 공유 드라이브 "캡스톤디자인(1)"을 내 드라이브에 바로가기로 추가해 둔다
   - 노트북이 멈추면 출력 메시지대로 `BASE_META_OVERRIDE` · `DATA_OVERRIDE` · `MD_OVERRIDE`에 경로를 넣는다
2. **팀원 확인**
   - p_low: 지시는 "예: 0.5". 노트북은 사전 점검 규칙(0.5 → 0.3 → 0.1 중 근사 추가 오답 ≥ 0.70%인 첫 값)으로 정한다. 팀원과 값을 정했으면 `P_LOW_OVERRIDE`
   - 선택 조건(Jover)을 돌릴지. 돌린다면 **사전 기록을 처음 쓰기 전에** `RUN_OVERCONF = True` (결과를 본 뒤 추가하지 않는다)
3. **결과 정리**: 노트북 §6 출력(판정 기준 ①~④)과 Drive `judge_e_cap/results_E.csv` · `record_E.json`을 가져와 보고한다

## 4. 이어서 진행할 때 Claude에게

- 이 파일 · `JUDGE_CAP.md`를 읽고, 노트북 §6 · §7 출력이나 `results_E.csv` · `PREREG.json` · `record_E.json`을 올리면 결과 해석(설계 6절 표)부터 이어간다
- 노트북을 고칠 때는 `scripts/judge_cap/gen_notebook_E.py`를 고치고 다시 만든다 (`python scripts/judge_cap/gen_notebook_E.py notebooks/mvr_colab_judge_cap_E.ipynb`)
- 데이터 · 결과 파일은 GitHub에 올리지 않는다

## 5. 확인하지 못한 것 (실행하면 알게 됨)

- Classification U_s1 · Jmed_s1 · JC1L_s1 기록의 실제 GPU · 파일 위치 · 이름 (노트북 §1이 찾아 출력)
- Jmed p* 값과 `learned.json` 위치 (§1이 Jmed_s1 인자에서 읽음)
- test 파일 이름 (§3이 JC1L_s1의 `--dataset` 이름으로 `mvr_splits`에서 찾음)
- 노트북은 문법 검사와 가짜 데이터로 §6 · §7 집계만 확인했다. Colab에서 실제로 돌려 본 적은 없다
