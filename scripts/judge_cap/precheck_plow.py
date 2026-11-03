"""판정자 실험 E 사전 점검: 과신 판정자의 p_low를 실행 전에 고정한다 (test 결과를 보기 전에 기존 판정 기록만 사용).

Jmed_s1의 판정 기록(DECISION_LOG jsonl, 요청마다 한 줄)에서, vCache가 재사용하지 않은 요청 중
P >= p_low인 요청을 모두 덮어쓴다고 가정하고 "추가 오답(전환 중 틀린 건수 / 전체 요청)"을 근사한다.
잡아먹기(덮어쓴 요청이 ⑤를 건너뛰어 판정자 대상이 더 늘어나는 효과)는 반영하지 않으므로
실제 값보다 작게 나오는 근사다.

규칙: 후보 p_low를 큰 값부터 보고, 근사 추가 오답 >= 목표(기본: ε 0.35%의 2배 = 0.70%. probe 10%와 잡아먹기를 감안)인
첫 값을 p_low로 고른다. 결과를 JUDGE_CAP.md 2절에 기록한 뒤 실행한다.

사용:
  python scripts/judge_cap/precheck_plow.py --log Jmed_s1.jsonl \
      --p-key p --correct-key nn_correct --reason-key reason --exploit-value exploit
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Jmed_s1 판정 기록 (jsonl)")
    ap.add_argument("--p-key", default="p", help="판정자 확률 p 필드 (팀원 기록에 맞춰 지정)")
    ap.add_argument("--correct-key", default="nn_correct", help="1등 후보가 정답인지 필드")
    ap.add_argument("--reason-key", default="reason", help="vCache 판정 이유 필드 (판정자 적용 전)")
    ap.add_argument("--exploit-value", default="exploit", help="vCache 자체 재사용을 뜻하는 reason 값")
    ap.add_argument("--skip", type=int, default=2000, help="집계에서 뺄 앞 요청 수")
    ap.add_argument("--candidates", type=float, nargs="+", default=[0.5, 0.3, 0.1])
    ap.add_argument("--budget", type=float, default=0.0035)
    ap.add_argument("--target-mult", type=float, default=2.0, help="목표 = 한도 x 이 배수")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()][args.skip:]
    n = len(rows)
    target = args.budget * args.target_mult
    print(f"rows {n} (앞 {args.skip}행 제외), 목표 추가 오답 >= {target:.2%}")
    chosen = None
    for p_low in sorted(args.candidates, reverse=True):
        conv = wrong = 0
        for r in rows:
            p = r.get(args.p_key)
            if p is None or r.get(args.reason_key) == args.exploit_value:
                continue
            if float(p) >= p_low:
                conv += 1
                wrong += 0 if r.get(args.correct_key) else 1
        rate = wrong / n if n else 0.0
        ok = rate >= target
        print(f"p_low {p_low:.2f}: 전환 {conv} ({conv / n:.2%}), 추가 오답 {wrong} ({rate:.3%}) {'<- 목표 충족' if ok else ''}")
        if ok and chosen is None:
            chosen = p_low
    print(f"선택 p_low = {chosen}" if chosen is not None else "후보 중 목표를 넘는 값이 없음 → 후보를 더 낮춰 다시 점검")


if __name__ == "__main__":
    main()
