"""
analyze_coldstart.py
판정 기록(DECISION_LOG jsonl)으로 '콜드 스타트 개선'의 천장을 계산한다. ※ 진단용(테스트 정답 사용)

출력
  1. 요청을 판정 이유별로 분류: exploit(재사용) / explore(확신 부족) / cold(기록 6개 미만이라 무조건 LLM) / no_candidate
  2. 놓친 기회(FN): 1등 후보가 정답이었는데 재사용 안 한 경우를 이유별로
  3. 천장 A (절대 상한): cold 때문에 놓친 정답을 전부 히트로 바꾸면
  4. 천장 B (δ를 지키는 상한): cold 판정들을 점수 높은 순으로 재사용하되
     전체 오류율이 δ를 넘지 않는 데까지 (점수 임계값 하나로 = 이상적인 공용 사전 곡선)

사용:
  python analyze_coldstart.py /workspace/results/cls41k/decisions/C_rule4_run1.jsonl --delta 0.01
  여러 파일: python analyze_coldstart.py /workspace/results/cls41k/decisions/*.jsonl --delta 0.01
"""
import argparse
import json
import os
from collections import Counter


def analyze(path, delta):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows)
    hits = [r for r in rows if r["hit"]]
    err = sum(1 for r in hits if r.get("nn_correct") is False)
    reason_cnt = Counter(r.get("reason", "no_candidate") for r in rows)

    fn = [r for r in rows if (not r["hit"]) and r.get("nn_correct") is True]
    fn_by_reason = Counter(r.get("reason") for r in fn)
    cold_fn = [r for r in fn if r.get("reason") == "cold"]

    # 천장 A: cold로 놓친 정답 전부 히트
    ceil_a = len(hits) + len(cold_fn)

    # 천장 B: cold 판정을 점수 내림차순으로 추가 재사용, 전체 오류 <= delta*n 유지
    cold = [r for r in rows if r.get("reason") == "cold" and not r["hit"] and r.get("nn_correct") is not None]
    cold.sort(key=lambda r: -r.get("s", 0.0))
    budget = delta * n - err
    add_ok = add_bad = 0
    best = (0, 0, None)  # (추가 정답, 추가 오답, 임계값)
    for r in cold:
        if r["nn_correct"]:
            add_ok += 1
        else:
            add_bad += 1
        if add_bad <= budget:
            best = (add_ok, add_bad, r.get("s"))
        else:
            break
    b_ok, b_bad, b_thr = best

    obs_fn = Counter(min(r.get("n_obs", 0), 6) for r in fn if r.get("reason") in ("cold", "explore", "exploit"))

    def pct(x):
        return f"{100 * x / max(n, 1):.2f}%"

    print("=" * 70)
    print(os.path.basename(path), f"(n={n}, delta={delta})")
    print(f"현재 히트율 {pct(len(hits))}  ({len(hits)}),  오류율 {pct(err)}  ({err})")
    print("판정 이유별 요청 수:", dict(reason_cnt))
    print(f"놓친 기회(FN) {len(fn)}건 = 히트의 {len(fn) / max(len(hits), 1):.1f}배")
    print("  이유별:", dict(fn_by_reason))
    print("  FN 시점 항목 기록 수(6=6개 이상):", dict(sorted(obs_fn.items())))
    print(f"[천장 A] cold FN 전부 히트로: 히트율 {pct(len(hits))} -> {pct(ceil_a)}  (+{pct(len(cold_fn))})")
    print(f"[천장 B] δ 유지하며 cold 재사용 (점수 ≥ {b_thr}): 히트율 {pct(len(hits))} -> "
          f"{pct(len(hits) + b_ok + b_bad)}  (정답 +{b_ok}, 오답 +{b_bad}), "
          f"오류율 {pct(err)} -> {pct(err + b_bad)}")
    return {"file": path, "n": n, "hits": len(hits), "errors": err, "fn": len(fn),
            "fn_by_reason": dict(fn_by_reason), "ceil_a_hits": ceil_a,
            "ceil_b_hits": len(hits) + b_ok + b_bad, "ceil_b_errors": err + b_bad, "ceil_b_threshold": b_thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--out", default=None, help="요약 json 저장 경로(선택)")
    args = ap.parse_args()
    res = [analyze(p, args.delta) for p in args.logs]
    print("=" * 70)
    print("주의: 천장은 테스트 정답을 본 진단값이며, 판정이 바뀌면 이후 캐시 내용도 달라지므로 근사치임")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("저장:", args.out)


if __name__ == "__main__":
    main()
