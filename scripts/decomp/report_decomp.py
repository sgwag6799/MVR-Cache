"""[cls-scale-decomp] 분해 실험 집계 (DECOMP 부분 5 · 6절).

사용: python scripts/decomp/report_decomp.py <OUT 폴더> [--skip 2000]
- 각 칸의 <칸>.json(per_sample)을 읽어 앞 --skip 요청을 뺀 공식 집계를 낸다.
- 히트율 · 오류율(= 오답 재사용 ÷ 요청) · 오류율 95% 상한(Clopper-Pearson)
- 요인 효과(CLS · 점수 범위 · 후보 수 · 코드 경로)와 사전 기록 판정을 출력하고 report.md · report.csv로 남긴다.
"""
import argparse
import csv
import json
import math
import os

CELLS = ["V", "P1", "P2", "P3", "P4", "P5"]
DESC = {
    "V": "바닐라 vCache (CLS 포함 · 코사인 · 후보 1, main 판정 코드)",
    "P1": "CLS 포함 · 코사인 그대로 · 후보 1",
    "P2": "CLS 포함 · (x+1)/2 · 후보 1",
    "P3": "CLS 제외 · 코사인 그대로 · 후보 1",
    "P4": "CLS 제외 · (x+1)/2 · 후보 1 (A′)",
    "P5": "CLS 제외 · (x+1)/2 · 후보 10 (B0)",
}
HIT_BIG = 1.0          # %p — 재실행 변동(약 0.4%p)의 2.5배 (사전 기록)
ERR_RATIO = (0.8, 1.25)  # 오류율 비가 이 범위 밖이면 "차이 큼" (사전 기록)
AGREE_MIN = 0.99       # P1 ↔ V 요청별 히트 일치율 (사전 기록)


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


def load(out_dir: str, cell: str, skip: int):
    path = os.path.join(out_dir, f"{cell}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = d["per_sample"][skip:]
    n = len(rows)
    hit = [int(bool(r["is_hit"])) for r in rows]
    fp = [int(r["fp"]) for r in rows]
    meta_p = os.path.join(out_dir, f"{cell}.meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    return {"n": n, "hit": hit, "fp": fp, "hits": sum(hit), "fps": sum(fp),
            "hit_rate": 100 * sum(hit) / max(1, n), "err": 100 * sum(fp) / max(1, n),
            "err_up": 100 * cp_upper(sum(fp), n), "gpu": meta.get("gpu", "?"),
            "commit": meta.get("git_commit", "?"), "data": meta.get("data_sha256_12", "?")}


def diff(a, b):
    """b − a (히트율 %p, 오류율 %p, 오류율 비 b/a)"""
    ratio = (b["err"] / a["err"]) if a["err"] > 0 else float("nan")
    return b["hit_rate"] - a["hit_rate"], b["err"] - a["err"], ratio


def big(dh, ratio):
    return abs(dh) >= HIT_BIG or not (ERR_RATIO[0] <= ratio <= ERR_RATIO[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--skip", type=int, default=2000, help="공식 집계에서 뺄 앞 요청 수")
    args = ap.parse_args()
    R = {c: load(args.out_dir, c, args.skip) for c in CELLS}
    lines = [f"# 분해 실험 집계 — {args.out_dir} (앞 {args.skip}행 제외)", ""]
    lines += ["| 칸 | 설정 | n | 히트율 % | 오류율 % (95% 상한) | GPU | 커밋 | 데이터 |", "|---|---|---|---|---|---|---|---|"]
    rows_csv = []
    for c in CELLS:
        r = R[c]
        if r is None:
            lines.append(f"| {c} | {DESC[c]} | — | 미실행 | | | | |")
            continue
        lines.append(f"| {c} | {DESC[c]} | {r['n']} | {r['hit_rate']:.3f} | {r['err']:.3f} ({r['err_up']:.3f}) | {r['gpu']} | {r['commit']} | {r['data']} |")
        rows_csv.append({"cell": c, "n": r["n"], "hit_rate": r["hit_rate"], "err": r["err"], "err_up95": r["err_up"],
                         "gpu": r["gpu"], "commit": r["commit"], "data": r["data"]})

    # 유효성: 같은 데이터 · 같은 GPU · 같은 커밋
    done = [c for c in CELLS if R[c]]
    for key, name in (("data", "데이터"), ("gpu", "GPU"), ("commit", "커밋")):
        vals = {R[c][key] for c in done}
        if len(vals) > 1:
            lines.append(f"\n⚠️ 칸마다 {name}이(가) 다르다: {vals} — 비교 무효, 같은 조건으로 다시 돌린다")

    lines += ["", "## 요인 효과 (뒤 칸 − 앞 칸)", "", "| 비교 | Δ히트율 %p | Δ오류율 %p | 오류율 비 | 판정 |", "|---|---|---|---|---|"]
    comps = [
        ("코드 경로 (P1 − V)", "V", "P1"),
        ("CLS 제외, 코사인일 때 (P3 − P1)", "P1", "P3"),
        ("CLS 제외, (x+1)/2일 때 (P4 − P2)", "P2", "P4"),
        ("(x+1)/2, CLS 포함일 때 (P2 − P1)", "P1", "P2"),
        ("(x+1)/2, CLS 제외일 때 (P4 − P3)", "P3", "P4"),
        ("후보 1 → 10 (P5 − P4)", "P4", "P5"),
        ("전체 (P5 − V)", "V", "P5"),
    ]
    eff = {}
    for name, a, b in comps:
        if R[a] is None or R[b] is None:
            lines.append(f"| {name} | — | — | — | 미실행 |")
            continue
        dh, de, ratio = diff(R[a], R[b])
        eff[(a, b)] = (dh, de, ratio)
        lines.append(f"| {name} | {dh:+.3f} | {de:+.3f} | {ratio:.2f} | {'차이 큼' if big(dh, ratio) else '작음'} |")
        rows_csv.append({"cell": f"{b}-{a}", "hit_rate": dh, "err": de, "err_up95": ratio})

    # 주효과 · 상호작용 (2×2)
    if all((k in eff) for k in [("P1", "P3"), ("P2", "P4"), ("P1", "P2"), ("P3", "P4")]):
        cls_main = (eff[("P1", "P3")][0] + eff[("P2", "P4")][0]) / 2
        scale_main = (eff[("P1", "P2")][0] + eff[("P3", "P4")][0]) / 2
        inter = eff[("P3", "P4")][0] - eff[("P1", "P2")][0]
        lines += ["", f"- CLS 제외 주효과(히트율): {cls_main:+.3f}%p",
                  f"- (x+1)/2 주효과(히트율): {scale_main:+.3f}%p",
                  f"- 상호작용 (P4 − P3) − (P2 − P1): {inter:+.3f}%p"]

    # 코드 경로 확인: P1 ↔ V 요청별 일치
    if R["V"] and R["P1"] and R["V"]["n"] == R["P1"]["n"]:
        agree = sum(int(x == y) for x, y in zip(R["V"]["hit"], R["P1"]["hit"])) / R["V"]["n"]
        ok = agree >= AGREE_MIN and abs(R["P1"]["hit_rate"] - R["V"]["hit_rate"]) < 0.4
        lines += ["", f"## 코드 경로 확인 (사전 기록: 일치율 ≥ {AGREE_MIN:.0%} 이고 |Δ히트율| < 0.4%p)",
                  f"- P1 ↔ V 요청별 히트 일치율 {agree:.4%} → {'통과: V와 P1을 같은 출발점으로 본다' if ok else '미통과: 이후 효과는 P1 기준으로만 읽고, 경로 차이를 먼저 확인한다'}"]

    lines += ["", f"판정 규칙(사전 기록): |Δ히트율| ≥ {HIT_BIG}%p 또는 오류율 비가 {ERR_RATIO[0]}~{ERR_RATIO[1]} 밖이면 '차이 큼'.",
              "'차이 큼'인 요인은 같은 데이터셋에서 순서 2·3 반복 + 판정자 조건으로 확장한다(DECOMP 부분 6절)."]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(args.out_dir, "report.md"), "w").write(text + "\n")
    with open(os.path.join(args.out_dir, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "n", "hit_rate", "err", "err_up95", "gpu", "commit", "data"])
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)


if __name__ == "__main__":
    main()
