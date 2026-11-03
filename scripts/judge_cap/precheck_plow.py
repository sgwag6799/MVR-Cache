"""판정자 실험 E 사전 점검: 과신 판정자의 p_low를 실행 전에 고정한다 (test 결과를 보기 전에 기존 판정 기록만 사용).

Jmed_s1의 판정 기록(DECISION_LOG jsonl, 요청마다 한 줄)에서 판정자 대상 요청
(팀원 노트북 eligible()과 같음: reason이 cold · explore, has_nn, candidate_count >= 5, a_excl 있음) 중
p >= p_low인 요청을 모두 전환한다고 가정하고 "추가 오답(전환 중 틀린 건수 / 전체 요청)"을 근사한다.
잡아먹기(전환한 요청이 ⑤를 건너뛰어 판정자 대상이 더 늘어나는 효과)는 반영하지 않으므로
실제 값보다 작게 나오는 근사다.

p는 판정 기록에 따로 남지 않으므로 판정자 모델(learned.json)로 다시 계산한다
(팀원 SearchQueries 노트북 C-2의 prob()와 같은 식: 표준화 후 로지스틱).

규칙: 후보 p_low를 큰 값부터 보고, 근사 추가 오답 >= 목표(기본: ε 0.35%의 2배 = 0.70%.
probe 10%와 잡아먹기를 감안)인 첫 값을 p_low로 고른다. 결과를 JUDGE_CAP.md 2절에 기록한 뒤 실행한다.

사용:
  python scripts/judge_cap/precheck_plow.py --log Jmed_s1.jsonl --model learned.json
"""
import argparse
import json
import math

# learned.json의 특징 이름 → 판정 기록 필드
FEATURE_COL = {"similarity": "s", "agreement": "a_excl", "candidate_count": "candidate_count"}


def prob(row, m):
    z = float(m["intercept"])
    for nm, c, mu, sc in zip(m["features"], m["coefficients"], m["feature_means"], m["feature_scales"]):
        z += float(c) * (float(row[FEATURE_COL.get(nm, nm)]) - float(mu)) / float(sc)
    return 1.0 / (1.0 + math.exp(-z))


def eligible(r):
    return (r.get("reason") in ("cold", "explore") and bool(r.get("has_nn"))
            and (r.get("candidate_count") or 0) >= 5 and r.get("a_excl") is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Jmed_s1 판정 기록 (jsonl)")
    ap.add_argument("--model", default=None, help="판정자 모델 learned.json (Jmed와 같은 모델). 없으면 --p-key 필드를 읽는다")
    ap.add_argument("--p-key", default="p", help="--model이 없을 때 읽을 확률 필드")
    ap.add_argument("--skip", type=int, default=2000, help="집계에서 뺄 앞 요청 수")
    ap.add_argument("--candidates", type=float, nargs="+", default=[0.5, 0.3, 0.1])
    ap.add_argument("--budget", type=float, default=0.0035)
    ap.add_argument("--target-mult", type=float, default=2.0, help="목표 = ε x 이 배수")
    args = ap.parse_args()

    m = json.load(open(args.model, encoding="utf-8")) if args.model else None
    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("i", 0) >= args.skip] if "i" in rows[0] else rows[args.skip:]
    n = len(rows)
    el = []
    for r in rows:
        if not eligible(r):
            continue
        p = prob(r, m) if m else r.get(args.p_key)
        if p is not None:
            el.append((float(p), r.get("nn_correct") is True))
    target = args.budget * args.target_mult
    print(f"집계 요청 {n} (앞 {args.skip}행 제외), 판정자 대상 {len(el)}, 목표 추가 오답 >= {target:.2%}")
    chosen = None
    for p_low in sorted(args.candidates, reverse=True):
        conv = sum(1 for p, _ in el if p >= p_low)
        wrong = sum(1 for p, ok in el if p >= p_low and not ok)
        rate = wrong / n if n else 0.0
        ok = rate >= target
        print(f"p_low {p_low:.2f}: 전환 {conv} ({conv / n:.2%}), 전환 중 오답 {wrong}, 추가 오답 {rate:.3%} {'<- 목표 충족' if ok else ''}")
        if ok and chosen is None:
            chosen = p_low
    print(f"선택 p_low = {chosen}" if chosen is not None else "후보 중 목표를 넘는 값이 없음 → 후보를 더 낮춰 다시 점검")


if __name__ == "__main__":
    main()
