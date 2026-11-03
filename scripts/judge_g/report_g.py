"""판정자 실험 G 집계: 공통문단 L8 + 실행 중 한도 (CJ-L · EJ-L) 와 C δ 곡선 보간, 판정 기준.

입력: run_judge_prefix_cls.sh 결과 폴더 (<이름>.jsonl 판정 기록, <이름>.budget.json 한도 상태, <이름>.meta.json GPU · 커밋)
집계 구간: 앞 2,000행 제외 (i >= 2000). 정의는 팀원 문서 0절과 같다.
  오답 히트 = hit 이고 1등 후보가 정답이 아님 (nn_correct != True)
  전환     = hit 이고 policy_reason 이 judge_ 로 시작 (판정자가 재사용으로 바꾼 요청)
  추가 오답 = 전환 중 오답 ÷ 전체 요청

사용: python scripts/judge_g/report_g.py --runs data/judge_cls/runs/shared --judge data/judge_cls/judge/shared_L8_all_b0.0035.json --out data/judge_g_report
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import beta

SKIP = 2000
NOMINAL = 0.01 + 0.0035          # 명목 상한 δ + ε
GAIN_MIN = 0.4                   # 같은 오류율 C 곡선 대비 최소 이득 (%p)
PSTAR_EXP3 = 0.880               # 실험 3 의 공용 풀 L8 p* (판정자를 다시 만들었을 때 재현 확인)
OB = ("online_budget_blocked", "online_budget_bin_closed", "online_budget_probe")


def ub(k, n, conf=0.95):
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def load(path):
    d = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    d["pr"] = d["policy_reason"].fillna("").astype(str) if "policy_reason" in d else ""
    d["hit"] = d["hit"].astype(bool)
    d["fp"] = d["hit"] & (d["nn_correct"] != True)  # noqa: E712
    d["flip"] = d["hit"] & d["pr"].str.startswith("judge_")
    return d


def summarize(name, d, runs):
    g = d[d["i"] >= SKIP]
    n = len(g)
    r = {"이름": name, "n": n, "히트율%": 100 * g.hit.mean(), "오류율%": 100 * g.fp.mean(),
         "오류율95%상한%": 100 * ub(int(g.fp.sum()), n), "전환": int(g.flip.sum()),
         "추가오답%": 100 * (g.flip & g.fp).sum() / n, "자체재사용": int((g.hit & ~g.flip).sum())}
    for why in OB:
        m = g.pr == why
        r[f"{why[14:]}"] = int(m.sum())
        r[f"{why[14:]}_오답"] = int((m & (g["nn_correct"] != True)).sum())  # noqa: E712
    meta = os.path.join(runs, f"{name}.meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta, encoding="utf-8"))
        r["GPU"], r["커밋"] = m.get("gpu"), str(m.get("commit", ""))[:7]
    st = os.path.join(runs, f"{name}.budget.json")
    if os.path.exists(st):
        s = json.load(open(st, encoding="utf-8"))
        r["spent/allowed"] = f"{s['spent']:.0f}/{s['allowed']:.0f}"
        adm = [b["lo"] for b, ok in zip(s["bins"], s["admitted_final"]) if ok]
        r["실질통과임계값"] = min(adm) if adm else None
        r["vCache갱신건너뜀"] = s.get("skipped_updates")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731

    if a.judge and os.path.exists(a.judge):
        j = json.load(open(a.judge, encoding="utf-8"))
        ps = float(j["min_probability"])
        say(f"판정자 {a.judge}: p* = {ps:.4f} (실험 3 기록 {PSTAR_EXP3:.3f}"
            f"{', 같음' if abs(ps - PSTAR_EXP3) < 5e-4 else ', ⚠️ 다르다 — 판정자 재현 차이를 기록'})")

    D, rows = {}, []
    for p in sorted(glob.glob(os.path.join(a.runs, "*_L8_s[0-9].jsonl"))):
        name = os.path.basename(p)[:-6]
        if not os.path.exists(os.path.join(a.runs, f"{name}.json")):
            continue                                   # 아직 안 끝난 실행
        D[name] = load(p)
        rows.append(summarize(name, D[name], a.runs))
    if not rows:
        say("끝난 실행이 없다"); return
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    say(R.round(3).to_string(index=False))
    R.to_csv(os.path.join(a.out, "results_g.csv"), index=False)
    gpus = sorted({str(x) for x in R.get("GPU", pd.Series(dtype=str)).dropna()})
    say(f"\nGPU 종류: {gpus}{'  ⚠️ 실행마다 다르다 — 짝 비교 무효' if len(gpus) > 1 else ''}")

    get = lambda nm, col: float(R.loc[R["이름"] == nm, col].iloc[0]) if (R["이름"] == nm).any() else None  # noqa: E731
    seeds = sorted({int(m.group(1)) for nm in R["이름"] for m in [re.search(r"_s(\d+)$", nm)] if m})
    for sd in seeds:
        say(f"\n■ 시드 {sd}")
        pts = [(get(f"{c}_L8_s{sd}", "오류율%"), get(f"{c}_L8_s{sd}", "히트율%"), c)
               for c in ("C", "C0125", "C015", "C020") if get(f"{c}_L8_s{sd}", "오류율%") is not None]
        pts.sort()
        say("C δ 곡선 (오류율%, 히트율%): " + ", ".join(f"{c} ({e:.3f}, {h:.2f})" for e, h, c in pts))
        mono = all(pts[i][1] <= pts[i + 1][1] for i in range(len(pts) - 1))
        if not mono:
            say("  ⚠️ 오류율이 오를 때 히트율이 내려가는 구간이 있다 — 보간 결과는 참고만")

        def gain(nm):
            e, h = get(nm, "오류율%"), get(nm, "히트율%")
            if e is None or len(pts) < 2:
                return None, "기록 없음"
            if e < pts[0][0] or e > pts[-1][0]:
                return None, f"오류율 {e:.3f}% 가 곡선 범위({pts[0][0]:.3f}~{pts[-1][0]:.3f}%) 밖 → C020 (선택 조건)"
            base = float(np.interp(e, [p[0] for p in pts], [p[1] for p in pts]))
            return h - base, f"같은 오류율 C {base:.2f}% 대비 {h - base:+.2f}%p"
        gains = {}
        for nm in (f"CJL_L8_s{sd}", f"EJL_L8_s{sd}", f"CJ_L8_s{sd}", f"EJ_L8_s{sd}"):
            gv, msg = gain(nm)
            gains[nm] = gv
            if get(nm, "히트율%") is not None:
                say(f"  {nm}: 히트율 {get(nm, '히트율%'):.2f}% · 오류율 {get(nm, '오류율%'):.3f}% · {msg}")

        cjl = f"CJL_L8_s{sd}"
        if get(cjl, "오류율95%상한%") is not None:
            u = get(cjl, "오류율95%상한%")
            say(f"① CJ-L 오류율 95% 상한 {u:.3f}% ≤ {100 * NOMINAL:.2f}% : {'✅' if u <= 100 * NOMINAL else '❌'}")
        if gains.get(cjl) is not None:
            say(f"② CJ-L 같은 오류율 C 곡선 대비 {gains[cjl]:+.2f}%p > +{GAIN_MIN}%p : {'✅' if gains[cjl] > GAIN_MIN else '❌'}")
        ejl = f"EJL_L8_s{sd}"
        if gains.get(cjl) is not None and gains.get(ejl) is not None:
            dd = gains[ejl] - gains[cjl]
            verdict = '> 0 → "가중치와 판정자가 더해진다" 유지' if dd > 0 else "≤ 0 → 철회"
            say(f"③ EJ-L − CJ-L (같은 오류율, C 곡선 기준) {dd:+.2f}%p : {verdict}")
        cj = f"CJ_L8_s{sd}"
        if cj in D and cjl in D:
            same = float((D[cj]["reason"].to_numpy() == D[cjl]["reason"].to_numpy()).mean())
            ok = same == 1.0 and get(cj, "자체재사용") == get(cjl, "자체재사용")
            say(f"유효성: CJ vs CJ-L vCache 판정 이유 같은 비율 {same:.5f}, 자체 재사용 {get(cj, '자체재사용'):.0f} vs "
                f"{get(cjl, '자체재사용'):.0f} : {'✅ label only 로 캐시 경로 같음' if ok else '⚠️ 다르다 — 원인을 먼저 기록'}")
            blk = get(cjl, "blocked_오답") + get(cjl, "bin_closed_오답")
            say(f"참고: 한도가 막은 오답 {blk:.0f} (blocked + bin_closed), probe 로 빠진 오답 {get(cjl, 'probe_오답'):.0f}, "
                f"CJ → CJ-L 히트율 {get(cjl, '히트율%') - get(cj, '히트율%'):+.2f}%p · 추가 오답 "
                f"{get(cj, '추가오답%'):.3f}% → {get(cjl, '추가오답%'):.3f}%")
    open(os.path.join(a.out, "report_g.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
