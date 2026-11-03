"""판정자 실험 E 사전 점검: 과신 판정자의 세기를 실행 전에 고정한다 (test 결과를 보기 전에 기존 판정 기록만 사용).

두 가지 과신을 점검한다.
  pstar : 통과 임계값만 낮춤 (p >= p_low 이면 전환). 판정자 확률 p 자체는 그대로다
  shift : 판정자 확률을 부풀림 (logit p + Δ). p* 는 Jmed 그대로 둔다. learned.json 의 intercept 에 Δ 를 더한 것과 같다

Jmed_s1의 판정 기록(DECISION_LOG jsonl)에서 판정자 대상 요청
(팀원 노트북 eligible()과 같음: reason이 cold · explore, has_nn, candidate_count >= 5, a_excl 있음) 중
조건을 넘는 요청을 모두 전환한다고 가정하고 "추가 오답(전환 중 틀린 건수 / 전체 요청)"을 근사한다.
잡아먹기(전환한 요청이 ⑤를 건너뛰어 판정자 대상이 더 늘어나는 효과)는 반영하지 않으므로
실제 값보다 작게 나오는 근사다.

p는 판정 기록에 따로 남지 않으므로 판정자 모델(learned.json)로 다시 계산한다
(팀원 SearchQueries 노트북 C-2의 prob()와 같은 식: 표준화 후 로지스틱).

규칙: 후보를 약한 과신부터 보고, 근사 추가 오답 >= 목표(기본: ε 0.35%의 2배 = 0.70%)인 첫 값을 고른다.
shift 는 함께 "부풀린 p 로 본 기대 오답(1 - p')"도 출력한다. 이 값이 실제 오답보다 훨씬 작아야
한도의 사전값(1 - 구간 중앙)이 틀리는 조건, 즉 한도가 라벨로 고쳐야 하는 조건이 된다.

사용:
  python scripts/judge_cap/precheck_plow.py --log Jmed_s1.jsonl --model learned.json --mode pstar
  python scripts/judge_cap/precheck_plow.py --log Jmed_s1.jsonl --model learned.json --mode shift --pstar 0.9539
"""
import argparse
import json
import math

# learned.json의 특징 이름 → 판정 기록 필드
FEATURE_COL = {"similarity": "s", "agreement": "a_excl", "candidate_count": "candidate_count"}


def logit_of(row, m):
    z = float(m["intercept"])
    for nm, c, mu, sc in zip(m["features"], m["coefficients"], m["feature_means"], m["feature_scales"]):
        z += float(c) * (float(row[FEATURE_COL.get(nm, nm)]) - float(mu)) / float(sc)
    return z


def sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def eligible(r):
    return (r.get("reason") in ("cold", "explore") and bool(r.get("has_nn"))
            and (r.get("candidate_count") or 0) >= 5 and r.get("a_excl") is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Jmed_s1 판정 기록 (jsonl)")
    ap.add_argument("--model", required=True, help="Jmed 판정자 모델 learned.json")
    ap.add_argument("--mode", choices=["pstar", "shift"], default="pstar")
    ap.add_argument("--pstar", type=float, default=None, help="shift 모드: 고정할 p* (Jmed p*)")
    ap.add_argument("--candidates", type=float, nargs="+", default=None,
                    help="pstar: p_low 후보 (기본 0.5 0.3 0.1), shift: Δ 후보 (기본 1 2 3 4)")
    ap.add_argument("--skip", type=int, default=2000, help="집계에서 뺄 앞 요청 수")
    ap.add_argument("--budget", type=float, default=0.0035)
    ap.add_argument("--target-mult", type=float, default=2.0, help="목표 = ε x 이 배수")
    args = ap.parse_args()
    if args.mode == "shift" and args.pstar is None:
        ap.error("--mode shift needs --pstar (Jmed p*)")
    cands = args.candidates or ([0.5, 0.3, 0.1] if args.mode == "pstar" else [1.0, 2.0, 3.0, 4.0])
    # 약한 과신부터: pstar 는 큰 값부터, shift 는 작은 Δ 부터
    cands = sorted(cands, reverse=(args.mode == "pstar"))

    m = json.load(open(args.model, encoding="utf-8"))
    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("i", 0) >= args.skip] if "i" in rows[0] else rows[args.skip:]
    n = len(rows)
    el = [(logit_of(r, m), r.get("nn_correct") is True) for r in rows if eligible(r)]
    target = args.budget * args.target_mult
    print(f"집계 요청 {n} (앞 {args.skip}행 제외), 판정자 대상 {len(el)}, 목표 추가 오답 >= {target:.2%}")
    chosen = None
    for c in cands:
        if args.mode == "pstar":
            sel = [(sig(z), ok) for z, ok in el if sig(z) >= c]
            label = f"p_low {c:.2f}"
        else:
            sel = [(sig(z + c), ok) for z, ok in el if sig(z + c) >= args.pstar]
            label = f"Δ {c:+.1f} (p* {args.pstar})"
        conv = len(sel)
        wrong = sum(1 for _, ok in sel if not ok)
        believed = sum(1.0 - p for p, _ in sel)
        rate = wrong / n if n else 0.0
        ok = rate >= target
        print(f"{label}: 전환 {conv} ({conv / n:.2%}), 전환 중 오답 {wrong} / 판정자가 믿는 기대 오답 {believed:.0f}, "
              f"추가 오답 {rate:.3%} {'<- 목표 충족' if ok else ''}")
        if ok and chosen is None:
            chosen = c
    print(f"선택 = {chosen}" if chosen is not None else "후보 중 목표를 넘는 값이 없음 → 후보를 넓혀 다시 점검")


if __name__ == "__main__":
    main()
