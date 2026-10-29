# 실험 8 결과 (2단계 41K 스트림, LmArena)

설정 · 결과 · 판정: [`PREFIX_EXPERIMENT.md`](../../../../PREFIX_EXPERIMENT.md) 2 · 3절, 요약: [`CLAIM_A.md`](../../../../CLAIM_A.md).

| 파일 | 내용 |
|---|---|
| `stage2_runs.csv` | 실행별 히트율 · 오류율 · 정밀도 (조건 A~G × 단계 × 시드, 앞 2,000행 제외 38,964요청) |
| `stage2_summary.csv` | 조건 × 단계 평균 · 표준편차 · 오류율 최대 |
| `claim_a_L8_comparison.csv` | 주장 A: L8에서 E(규칙 분할 + 학습 가중치)와 A · F · C의 시드별 차이 |

- `stage2_runs.csv`는 Colab VM에서 `prefix_stage2_report.py`가 출력한 표(소수 6자리)에서 복원했다. 원본 실행 JSON · timing 파일 · RL 학습 로그는 VM과 함께 없어졌다.
- `stage2_summary.csv` · `claim_a_L8_comparison.csv`는 `stage2_runs.csv`에서 계산했다.
- 조건: A/B vCache base/LoRA, C/D 규칙 분할 균등 base/LoRA, E 규칙 분할 + 학습 가중치(M3) base, F/G 논문 MVR-Cache(재학습 RL 분할기) base/LoRA.
