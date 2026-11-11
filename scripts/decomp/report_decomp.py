"""[cls-scale-decomp] 분해 실험 집계 (DECOMP 부분 6절).

사용: python scripts/decomp/report_decomp.py <OUT 폴더> [--skip 2000]
- OUT/r<k>/<칸>.json(per_sample)을 반복마다 읽어 앞 --skip 요청을 뺀 공식 집계를 낸다.
- 반복별 표, 칸별 평균 ± 표준편차, 요인 효과(같은 반복끼리 짝지은 차이의 평균과 95% 구간), 6절 판정
- report.md · report.csv로 남긴다.
"""
import argparse
import csv
import json
import math
import os
import re
import statistics

CELLS = ["V", "P1", "P2", "P3", "P4", "P5"]
DESC = {
    "V": "바닐라 vCache (CLS 포함 · cos · 후보 1, main 판정 코드)",
    "P1": "CLS 포함 · cos · 후보 1 (MVR 코드)",
    "P2": "CLS 포함 · (cos+1)/2 · 후보 1",
    "P3": "CLS 제외 · cos · 후보 1",
    "P4": "CLS 제외 · (cos+1)/2 · 후보 1",
    "P5": "CLS 제외 · (cos+1)/2 · 후보 10 (= vCache(MVR))",
}
COMPS = [
    ("코드 경로 (P1 − V)", "V", "P1"),
    ("점수식, CLS 포함일 때 (P2 − P1)", "P1", "P2"),
    ("점수식, CLS 제외일 때 (P4 − P3)", "P3", "P4"),
    ("CLS 제외, cos일 때 (P3 − P1)", "P1", "P3"),
    ("CLS 제외, (cos+1)/2일 때 (P4 − P2)", "P2", "P4"),
    ("후보 1 → 10 (P5 − P4)", "P4", "P5"),
    ("전체 (P5 − V)", "V", "P5"),
]
HIT_BIG = 1.0            # %p — 재실행 변동(약 0.4%p)의 2.5배 (사전 기록)
ERR_RATIO = (0.8, 1.25)  # 오류율 비가 이 범위 밖이면 "차이 큼" (사전 기록)
AGREE_MIN = 0.99         # P1 ↔ V 요청별 히트 일치율 (사전 기록)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))
    except Exception:  # scipy가 없으면 Wilson 상한
        z = 1.959964
        p = k / n
        d = 1 + z * z / n
        return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def load(run_dir: str, cell: str, skip: int):
    path = os.path.join(run_dir, f"{cell}.json")
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))["per_sample"][skip:]
    n = len(rows)
    hit = [int(bool(r["is_hit"])) for r in rows]
    fps = sum(int(r["fp"]) for r in rows)
    meta_p = os.path.join(run_dir, f"{cell}.meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    return {"n": n, "hit": hit, "hit_rate": 100 * sum(hit) / max(1, n), "err": 100 * fps / max(1, n),
            "err_up": 100 * cp_upper(fps, n), "gpu": meta.get("gpu", "?"),
            "commit": meta.get("git_commit", "?"), "data": meta.get("data_sha256_12", "?")}


def mean_ci(xs):
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, T95.get(len(xs) - 1, 1.96) * statistics.stdev(xs) / math.sqrt(len(xs))


def ratio(a, b):
    return b / a if a > 0 else float("nan")


def big(dh, r):
    # 오류율 비가 nan(앞 칸 오류 0)이면 비율 기준은 쓰지 않는다
    return abs(dh) >= HIT_BIG or (not math.isnan(r) and not (ERR_RATIO[0] <= r <= ERR_RATIO[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--skip", type=int, default=2000, help="공식 집계에서 뺄 앞 요청 수")
    args = ap.parse_args()
    reps = sorted(int(m.group(1)) for d in os.listdir(args.out_dir) if (m := re.fullmatch(r"r(\d+)", d)))
    R = {k: {c: load(os.path.join(args.out_dir, f"r{k}"), c, args.skip) for c in CELLS} for k in reps}
    lines = [f"# 분해 실험 집계 — {args.out_dir} (앞 {args.skip}행 제외, 반복 {reps})", ""]
    rows_csv = []

    lines += ["## 반복별", "", "| 반복 | 칸 | n | 히트율 % | 오류율 % (95% 상한) | GPU | 커밋 | 데이터 |", "|---|---|---|---|---|---|---|---|"]
    for k in reps:
        for c in CELLS:
            r = R[k][c]
            if r is None:
                lines.append(f"| {k} | {c} | — | 미실행 | | | | |")
                continue
            lines.append(f"| {k} | {c} | {r['n']} | {r['hit_rate']:.3f} | {r['err']:.3f} ({r['err_up']:.3f}) | {r['gpu']} | {r['commit']} | {r['data']} |")
            rows_csv.append({"rep": k, "row": c, "n": r["n"], "hit": r["hit_rate"], "err": r["err"], "err_up95": r["err_up"]})
        done = [c for c in CELLS if R[k][c]]
        if len({R[k][c]["data"] for c in done}) > 1:
            lines.append(f"\n⚠️ 반복 {k}: 칸마다 데이터가 다르다 — 이 반복은 비교 무효")
    allrun = [R[k][c] for k in reps for c in CELLS if R[k][c]]
    for key, name in (("gpu", "GPU"), ("commit", "커밋")):
        vals = {r[key] for r in allrun}
        if len(vals) > 1:
            lines.append(f"\n⚠️ 실행마다 {name}이(가) 다르다: {vals} — 비교 무효, 같은 조건으로 다시 돌린다")

    lines += ["", "## 칸별 평균 (반복 평균 ± 95% 구간)", "", "| 칸 | 설정 | 반복 수 | 히트율 % | 오류율 % |", "|---|---|---|---|---|"]
    for c in CELLS:
        rs = [R[k][c] for k in reps if R[k][c]]
        if not rs:
            lines.append(f"| {c} | {DESC[c]} | 0 | 미실행 | |")
            continue
        h, hci = mean_ci([r["hit_rate"] for r in rs])
        e, eci = mean_ci([r["err"] for r in rs])
        lines.append(f"| {c} | {DESC[c]} | {len(rs)} | {h:.3f} ± {hci:.3f} | {e:.3f} ± {eci:.3f} |")
        rows_csv.append({"rep": "mean", "row": c, "n": len(rs), "hit": h, "err": e})

    # 요인 효과: 같은 반복(같은 순서 · 시드)끼리 짝지은 차이
    lines += ["", "## 요인 효과 (뒤 칸 − 앞 칸, 같은 반복끼리 짝지은 차이)", "",
              "| 비교 | 반복 수 | Δ히트율 %p (95% 구간) | Δ오류율 %p | 오류율 비 | 판정 |", "|---|---|---|---|---|---|"]
    eff = {}
    for name, a, b in COMPS:
        ks = [k for k in reps if R[k][a] and R[k][b]]
        if not ks:
            lines.append(f"| {name} | 0 | — | — | — | 미실행 |")
            continue
        dh, dhci = mean_ci([R[k][b]["hit_rate"] - R[k][a]["hit_rate"] for k in ks])
        de, _ = mean_ci([R[k][b]["err"] - R[k][a]["err"] for k in ks])
        r = ratio(sum(R[k][a]["err"] for k in ks), sum(R[k][b]["err"] for k in ks))
        eff[(a, b)] = dh
        lines.append(f"| {name} | {len(ks)} | {dh:+.3f} ({dhci:.3f}) | {de:+.3f} | {r:.2f} | {'차이 큼' if big(dh, r) else '작음'} |")
        rows_csv.append({"rep": "effect", "row": f"{b}-{a}", "n": len(ks), "hit": dh, "err": de, "err_up95": r})

    if all(k in eff for k in [("P1", "P3"), ("P2", "P4"), ("P1", "P2"), ("P3", "P4")]):
        lines += ["", "2×2 (P1 · P2 · P3 · P4, 히트율 평균 기준)",
                  f"- CLS 제외 주효과: {(eff[('P1', 'P3')] + eff[('P2', 'P4')]) / 2:+.3f}%p",
                  f"- (cos+1)/2 주효과: {(eff[('P1', 'P2')] + eff[('P3', 'P4')]) / 2:+.3f}%p",
                  f"- 상호작용 (P4 − P3) − (P2 − P1): {eff[('P3', 'P4')] - eff[('P1', 'P2')]:+.3f}%p"]

    lines += ["", f"## 코드 경로 확인 (사전 기록: 반복마다 일치율 ≥ {AGREE_MIN:.0%} 이고 |Δ히트율| < 0.4%p)"]
    for k in reps:
        v, p = R[k]["V"], R[k]["P1"]
        if v and p and v["n"] == p["n"]:
            agree = sum(int(x == y) for x, y in zip(v["hit"], p["hit"])) / v["n"]
            ok = agree >= AGREE_MIN and abs(p["hit_rate"] - v["hit_rate"]) < 0.4
            lines.append(f"- 반복 {k}: P1 ↔ V 요청별 히트 일치율 {agree:.4%} → {'통과' if ok else '미통과: 이후 효과는 P1 기준으로만 읽는다'}")

    lines += ["", f"판정 규칙(사전 기록): 반복 평균 |Δ히트율| ≥ {HIT_BIG}%p 또는 오류율 비가 {ERR_RATIO[0]}~{ERR_RATIO[1]} 밖이면 '차이 큼'.",
              "95% 구간이 0을 포함하면 방향은 확정하지 않는다(DECOMP 부분 6절)."]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(args.out_dir, "report.md"), "w").write(text + "\n")
    with open(os.path.join(args.out_dir, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "row", "n", "hit", "err", "err_up95"])
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)


if __name__ == "__main__":
    main()
