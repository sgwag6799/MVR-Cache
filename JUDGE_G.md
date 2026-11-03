# 판정자 실험 G: 공통문단 L8 + 실행 중 한도

> 질문: 공통문단 L8에서 고정 p*로는 덮어쓰기 양을 통제하지 못했다(CJ 추가 오답 +0.52%, 한도 0.35%의 1.5배). 실행 중 한도(JC1L)를 붙이면 명목 상한 안에서 같은 오류율의 C보다 히트율이 높은가? 그리고 EJ − CJ 이득은 같은 오류율에서도 남는가?

- 브랜치 `judge-prefix-online-cap` (기반: `judge-prefix-classification`, 실험 3의 C · E · CJ · EJ 코드)
- 조건 이름(U · Jmed · JC1 · JC1L)과 용어(p · p* · 전환 · 추가 오답 · probe · 실행 중 한도 · δ · ε)는 팀원 문서 "판정자 변경한 부분" **0절**을 따른다
- 실행은 SSH(Colab L4)에서 `scripts/judge_g/run_g.sh` 한 줄로 한다 (9절)

## 0. 실행 조건

| 항목 | 내용 |
|---|---|
| 같은 GPU | CJ-L · EJ-L · C δ 곡선, 그리고 비교에 쓰는 공통문단 C · E · CJ · EJ 기록을 같은 GPU에서 돌린 것으로 맞춘다. 다른 GPU면 비교에 쓰는 C · CJ(필요하면 E · EJ)도 그 GPU에서 다시 돌린다 |
| 기존 기록 확인 결과 | 실험 3 기록(C · E · CJ · EJ)은 Colab L4에서 돌렸지만 **실행별 GPU 기록이 없고**, 원본 결과는 그 VM에만 있었다. 그래서 GPU를 기록으로 확인할 수 없다 → **C_L8_s1 · CJ_L8_s1을 같은 세션에서 다시 돈다** (팀원 설계의 "2~4실행 추가" 중 2). E · EJ는 비교 기준에 들어가지 않아 선택 |
| 실행 기록 | 실행마다 `<이름>.meta.json`에 GPU 종류 · 커밋 해시 · 조건을 남긴다 |
| GPU | **L4** (판정자 실험 E와 같은 종류). L4가 아니면 `run_g.sh`가 멈춘다 |

## 1. 목적

- 공통문단 L8에서 CJ 25.24% / 1.12%(추가 오답 +0.52%, 허용 한도 0.35%의 1.5배). 원인은 덮어쓰는 비율이 학습 스트림의 약 2배로 늘어난 것이다. 고정 p*가 양을 통제하지 못한다.
- EJ − CJ +1.17%p(p 0.03)는 오류율 +0.14%p로 설명될 수 있다 [추정]. 같은 오류율 비교가 필요하다.
  - Classification δ 곡선 기울기 약 8.2 × 0.14 ≈ 1.15%p.
  - seg8 U 곡선(시드 1)의 기울기는 구간에 따라 약 29.7 · 15.6으로 더 가파르다. 그렇다면 이 추정은 더 강해진다.

## 2. 코드

| 항목 | 내용 |
|---|---|
| 기준선 · 판정자 | 실험 3과 같은 코드: 규칙 분할(구두점 전체, 상한 32, `cap_scope original`) + 균등 SMaxSim, 문장 벡터로 후보 10개, 판정자 learned_all_add35 (s · a_excl · 후보 수 로지스틱, 수집 시드 42~45 p* 중앙값) |
| 실행 중 한도 | 팀원 `online_budget.patch`(b07856972ddc) · `probe_label_only.patch`(92ca6d07db0a)의 규칙을 이 정책에 옮겼다. 구간 오답률 추정 · 구간 열기 · 누적 한도 · probe · 상태 출력 함수 4개는 **패치 원문과 같다**(label only 기록 필드 2개만 추가). 옵션 이름도 같다 |
| JC1L 설정 | `--consensus-online-budget 0.0035 --consensus-online-mode calibrated --consensus-online-z 1 --consensus-online-probe 0.1 --consensus-online-prior 20 --consensus-online-bin-width 0.02 --consensus-online-label-only --consensus-seed <시드>` |
| 판정 기록 | `policy_reason`: 전환 `judge_cold` · `judge_explore`, 한도가 LLM으로 돌림 `online_budget_blocked` · `online_budget_bin_closed` · `online_budget_probe`. 한도 상태는 `<이름>.budget.json` |
| label only | 한도가 LLM으로 돌린 요청은 vCache 관측 · 캐시 추가를 하지 않는다 → CJ-L의 캐시 경로가 CJ와 같다 (유효성 확인에 쓴다) |

## 3. 설계 (실행 전 고정)

| 조건 | 내용 | 실행 |
|---|---|---|
| **CJ-L** | C + 판정자 + JC1L (라벨 전용, 한도 0.35%) | 새로 |
| **EJ-L** | E + 판정자 + JC1L | 새로 |
| **C δ 곡선** | C를 δ 0.0125 · 0.015로 (같은 오류율 보간용). δ 0.01 점은 C | 새로 2개 |
| C · CJ | 같은 세션 · 같은 GPU 기준선 (0절 확인 결과) | 다시 돈다 |
| E · EJ | 참고 기준선 | 선택 |
| C δ 0.02 | **CJ-L 또는 EJ-L의 오류율이 C δ 0.015 오류율보다 높아 곡선 범위를 벗어날 때만** | 조건부 |

- 데이터: Classification 공용 풀 L8 (`prefix_pool.json`), test 중 모든 단계에서 512토큰 안에 드는 38,304행, 앞 2,000행 제외 집계.
- 시드 1 → 판정 기준을 통과하면 시드 2 · 3 (`jobs_g_s23.txt`).
- 판정자: 실험 3과 같은 규칙. 판정자 파일이 없으면 train + val 스트림(3,726행)을 수집 시드 42~45로 다시 모아 만든다. 실험 3의 p*(0.880)와 같은지 집계에서 확인한다.
- δ 0.01 · sleep 0.02 · 정답 = 8B 응답 문자열 일치 (실험 3과 같음).

### 같은 오류율 비교 방법 (실행 전 고정)

- C δ 곡선의 점(오류율, 히트율)을 오류율 순으로 놓고 **선형 보간**한다. 곡선의 기울기는 가정하지 않는다.
- 조건 X의 이득 = X 히트율 − (X 오류율에서 보간한 C 히트율).
- "EJ-L − CJ-L (같은 오류율)" = EJ-L 이득 − CJ-L 이득. 둘 다 같은 C 곡선을 기준으로 잰다.
- X의 오류율이 곡선 범위 밖이면 보간하지 않고, C δ 0.02(조건부)를 돌린 뒤 다시 잰다.

## 4. 판정 기준 (실행 전 기록)

1. **CJ-L: 오류율 95% 상한 ≤ 1.35%** (명목 δ + ε, one-sided Clopper–Pearson)
2. **CJ-L: 같은 오류율 C 곡선 보간 대비 > +0.4%p**
3. **EJ-L − CJ-L: 같은 오류율에서 > 0이면 "가중치와 판정자가 더해진다" 유지, 아니면 철회**
- 유효성: CJ와 CJ-L의 vCache 판정 이유 순서 · 자체 재사용 수가 같다 (label only). 다르면 원인을 먼저 기록한다.
- 같은 GPU: 모든 실행의 `meta.json` GPU가 같다.

## 5. 기대 [추정]

- CJ-L 히트율 약 22~24% (probe 10% 비용 약 2%p + 한도가 덮어쓰기 양을 줄임), 오류율 약 0.95%.

## 6. 비용

- 팀원 추정: 시드 1, 4실행(38K행, 약 4.4시간).
- 이 설계: 시드 1 기본 6실행(CJ-L · EJ-L · C δ 0.0125 · 0.015 + C · CJ 재실행) + 판정자 수집 4회(각 약 10분, 판정자 파일이 없을 때). 3개씩 병렬로 약 6시간 [추정, 실험 3이 L4에서 6실행 약 6시간].
- 선택: E · EJ 2실행, 조건부 C δ 0.02 1실행.

## 7. 한계 (실행 전 공개)

- 시드 1개로 먼저 판정한다.
- C δ 곡선만 잰다. E는 자기 δ 곡선이 없어, EJ-L의 같은 오류율 비교도 C 곡선을 기준으로 한다 (E와 C의 곡선이 같다는 가정).
- 판정자 p*는 실험 3과 같은 학습 스트림 규칙으로 정한다. 한도는 p* 위에서 양을 줄이는 장치라, p* 자체의 어긋남은 그대로다.
- 구간 비용(사후 평균 + 1σ)은 정규근사라 유한 표본 보장이 아니다. 논문에서는 **형식적 보장이 없는 경험적 한도**로 서술한다.
  - 구조는 Conformal Risk Control · Learn-then-Test와 비슷하고, 동시기 연구 Surplus(arXiv 2611.02498v1)와 거의 같다.
  - "기존 위험 통제 기법의 적용" · "보장"이라는 표현은 쓰지 않는다.
- 명목 상한은 δ + ε다. 한도가 막는 것은 판정자 추가 오답(ε)이고, vCache 자체 오답(δ)은 별도다.

## 8. 파일

| 파일 | 역할 |
|---|---|
| `JUDGE_G.md` | 이 설계 문서 |
| `HANDOFF_G.md` | 진행 상황 · 다음에 할 일 |
| `mvr-cache/vcache/vcache_policy/strategies/verified_splitter.py` | 실행 중 한도 · label only 추가 (팀원 패치 규칙 그대로) |
| `mvr-cache/benchmarks/eval_sembenchmark_verified_splitter.py` | `--consensus-online-*` · `--consensus-seed` 옵션, 한도 상태 출력 |
| `scripts/judge/run_judge_prefix_cls.sh` | 조건 CJL · EJL · C0125 · C015 · C020 추가, 실행별 `meta.json` |
| `scripts/judge_g/run_g.sh` | 환경 → 데이터 → 판정자 → 실행 → 집계 (끊기면 다시 실행하면 이어서) |
| `scripts/judge_g/jobs_g_s1.txt` · `jobs_g_s1_extra.txt` · `jobs_g_s23.txt` | 시드 1 기본 · 선택/조건부 · 시드 2 · 3 |
| `scripts/judge_g/report_g.py` | 결과 표, C δ 곡선 보간, 판정 기준 ①~③, 유효성 |

## 9. SSH 실행 (Colab L4)

맥 터미널:

```bash
ssh colab-l4                      # ~/.ssh/config 의 Host colab-l4 (세션 이름 l4, L4)
```

Colab VM 안 (처음 한 번):

```bash
apt-get -qq install -y tmux > /dev/null
cd /content
git clone -b judge-prefix-online-cap https://github.com/sgwag6799/MVR-Cache.git
tmux new -s g                     # 연결이 끊겨도 실행이 계속되게 tmux 안에서 돌린다
cd /content/MVR-Cache
bash scripts/judge_g/run_g.sh scripts/judge_g/jobs_g_s1.txt 3
```

- 빠져나오기(실행은 계속): `Ctrl+b` 다음 `d`. 다시 보기: `ssh colab-l4` → `tmux attach -t g`
- 진행 확인: `tail -f /content/MVR-Cache/data/judge_g_run.log`, 실행별 로그 `data/judge_cls/runs/shared/<이름>.log`
- 끊겨서 런타임이 새로 생겼으면 위 명령을 처음부터 다시 실행한다 (끝난 실행은 건너뛴다. 단 새 런타임이면 data/가 비어 있어 처음부터 다시 돈다)
- 결과를 맥으로 가져오기 (맥 터미널): `scp -r colab-l4:/content/MVR-Cache/data/judge_g_report ./judge_g_report`
- 결과 · 데이터는 GitHub에 올리지 않는다
