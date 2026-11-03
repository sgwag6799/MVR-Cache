"""무료 집계: 실행마다 오류율의 one-sided 95% Clopper-Pearson 상한이 명목 상한을 넘는지 센다.

입력 CSV 열: name, n (집계 요청 수), wrong (오답 수), [delta] (명목 상한: 기반 조건은 δ, 판정자 조건은 δ + ε. 없으면 --delta)
사용: python scripts/judge_cap/cp_upper_count.py --csv runs.csv --delta 0.01
"""
import argparse
import csv

from scipy.stats import beta


def cp_upper(wrong, n, conf=0.95):
    if wrong >= n:
        return 1.0
    return float(beta.ppf(conf, wrong + 1, n - wrong))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--conf", type=float, default=0.95)
    args = ap.parse_args()
    over = total = 0
    print("name\tn\twrong\terror\tcp_upper\tdelta\tover")
    for r in csv.DictReader(open(args.csv, encoding="utf-8")):
        n, w = int(r["n"]), int(r["wrong"])
        d = float(r.get("delta") or args.delta)
        up = cp_upper(w, n, args.conf)
        flag = up > d
        over += flag
        total += 1
        print(f"{r['name']}\t{n}\t{w}\t{w / n:.3%}\t{up:.3%}\t{d:.2%}\t{'O' if flag else ''}")
    print(f"95% 상한 > 명목 상한: {over} / {total}")


if __name__ == "__main__":
    main()
