"""Check whether the verified policy's (s, c) observations meet the logistic-model assumption.

vCache turns the similarity s into P(c=1 | s) with a logistic curve. That curve is exact
when s | c=0 and s | c=1 are normal with equal variance (then the log-odds is linear in s).
This script reads the observation dumps written by
`eval_sembenchmark_verified_splitter.py --dump-observations` and, per weighting condition,
reports:

  - per class: n, mean, sd, skewness, excess kurtosis, normality tests
  - equal variance: sd ratio and Brown-Forsythe (Levene, median-centred) test
  - logistic fit quality: slope of a fitted logistic vs the slope implied by the
    equal-variance normal model (LDA), and calibration error of the fitted curve

With tens of thousands of points every normality test rejects, so read the effect sizes
(skewness, kurtosis, sd ratio, calibration error) rather than the p-values.

Example:
  python benchmarks/analyze_observation_normality.py \
    results/obs/obs_wuniform.json results/obs/obs_widf.json --out-dir results/obs
"""
# [새 파일] vCache가 모은 (s, c) 관측이 "로지스틱 모델의 전제"를 만족하는지 확인하는 분석 스크립트.
# 배경: c=0일 때의 s 분포와 c=1일 때의 s 분포가 둘 다 정규분포이고 분산이 같으면,
#       P(c=1|s)는 정확히 로지스틱 곡선이 된다. vCache는 이 로지스틱 곡선을 쓴다.
# 입력: 평가 스크립트의 --dump-observations 결과 파일(obs_w*.json)
# 출력: observation_normality.png (분포/QQ/보정 그래프), observation_normality.json (수치)

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

# 그래프 색상 (c=0 주황, c=1 파랑, 글자/보조/격자 색)
C0_COLOR, C1_COLOR = "#eb6834", "#2a78d6"  # validated categorical slots (orange, blue)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# 관측 파일 읽기 → (조건 이름, s 배열, c 배열)
def load(path: str) -> tuple[str, np.ndarray, np.ndarray]:
    d = json.load(open(path))
    # observations: [캐시 항목 번호, s, c] 목록 → 1번 열이 s, 2번 열이 c
    obs = np.asarray(d["observations"], dtype=float)
    # 조건 이름은 실행 인자에 기록된 가중치 방식 (없으면 파일 이름)
    name = str(d["args"].get("segment_weighting", os.path.basename(path)))
    return name, obs[:, 1], obs[:, 2].astype(int)


# 한 클래스(c=0 또는 c=1)의 s 분포 요약 통계
def class_stats(x: np.ndarray, rng: np.random.Generator) -> dict:
    # Shapiro 검정은 표본이 5000개를 넘으면 부정확해서 무작위 5000개만 뽑아 사용
    sub = rng.choice(x, size=min(len(x), 5000), replace=False)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "skew": float(stats.skew(x)),  # 왜도: 0이면 좌우 대칭, 양수면 오른쪽 꼬리가 김
        "excess_kurtosis": float(stats.kurtosis(x)),  # 첨도: 0이면 정규분포와 같은 뾰족함
        "dagostino_p": float(stats.normaltest(x).pvalue),  # 정규성 검정 p값 (작을수록 정규분포 아님)
        "shapiro_p_5k": float(stats.shapiro(sub).pvalue),  # 다른 정규성 검정 p값
    }


def calibration(s: np.ndarray, c: np.ndarray, prob: np.ndarray, bins: int = 20) -> tuple:
    """Quantile-binned empirical P(c=1) vs predicted; returns (ECE, bin centers, emp, pred)."""
    # 보정(calibration) 확인: s를 개수가 같은 20개 구간으로 나누고,
    # 구간마다 "실제 정답 비율"과 "모델이 예측한 확률"을 비교한다.
    # ECE = 구간별 |실제 − 예측| 차이를 구간 크기로 가중 평균한 값 (0에 가까울수록 잘 맞음)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    centers, emp, pred, ece = [], [], [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        centers.append(s[m].mean())
        emp.append(c[m].mean())
        pred.append(prob[m].mean())
        ece += m.mean() * abs(c[m].mean() - prob[m].mean())
    return float(ece), np.array(centers), np.array(emp), np.array(pred)


def analyze(name: str, s: np.ndarray, c: np.ndarray, rng: np.random.Generator) -> dict:
    # c=0인 s들과 c=1인 s들로 나눈다
    s0, s1 = s[c == 0], s[c == 1]
    st0, st1 = class_stats(s0, rng), class_stats(s1, rng)

    # Fitted logistic (what vCache assumes, fitted globally here).
    # (1) 데이터에 로지스틱 회귀를 직접 맞춘 곡선. vCache는 항목별로 맞추지만 여기서는 전체를 한 번에 맞춘다.
    #     gamma = 기울기, t_hat = 확률 0.5가 되는 s 값(임계값)
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(s.reshape(-1, 1), c)
    gamma_fit, b_fit = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    p_fit = lr.predict_proba(s.reshape(-1, 1))[:, 1]

    # Logistic implied by equal-variance normal classes (LDA).
    # (2) "두 클래스가 분산이 같은 정규분포"라고 가정했을 때 이론적으로 나오는 로지스틱 곡선.
    #     평균과 공통 분산만으로 기울기·절편을 계산한다. (1)과 비슷하면 정규성 전제가 크게 어긋나지 않은 것
    n0, n1 = len(s0), len(s1)
    pooled_var = ((n0 - 1) * s0.var(ddof=1) + (n1 - 1) * s1.var(ddof=1)) / (n0 + n1 - 2)
    gamma_lda = (s1.mean() - s0.mean()) / pooled_var
    b_lda = -(s1.mean() ** 2 - s0.mean() ** 2) / (2 * pooled_var) + np.log(n1 / n0)
    p_lda = 1.0 / (1.0 + np.exp(-(gamma_lda * s + b_lda)))

    ece_fit, centers, emp, pred_fit = calibration(s, c, p_fit)
    ece_lda, _, _, pred_lda = calibration(s, c, p_lda)
    return {
        "condition": name,
        "c0": st0,
        "c1": st1,
        "sd_ratio_c1_over_c0": st1["sd"] / st0["sd"],  # 표준편차 비 (1이면 등분산)
        "brown_forsythe_p": float(stats.levene(s0, s1, center="median").pvalue),  # 등분산 검정 p값
        # AUC: 무작위로 c=1 하나와 c=0 하나를 뽑았을 때 c=1의 s가 더 클 확률 (0.5 = 구분 못 함)
        "auc": float(stats.mannwhitneyu(s1, s0).statistic / (n0 * n1)),
        "logistic_fit": {"gamma": gamma_fit, "t_hat": -b_fit / gamma_fit, "ece": ece_fit},
        "lda_implied": {"gamma": float(gamma_lda), "t_hat": float(-b_lda / gamma_lda), "ece": ece_lda},
        "_plot": (s0, s1, centers, emp, pred_fit, pred_lda),  # 그래프용 데이터 (JSON에는 저장 안 함)
    }


# 그래프 공통 스타일 (위/오른쪽 테두리 제거, 연한 격자 등)
def style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# 조건마다 한 줄, 그래프 3개: ① s 분포 + 정규분포 곡선 ② QQ 그림 ③ 보정 곡선
def plot(results: list, path: str) -> None:
    fig, axes = plt.subplots(len(results), 3, figsize=(13, 3.2 * len(results)), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for row, r in zip(axes, results):
        s0, s1, centers, emp, pred_fit, pred_lda = r["_plot"]
        lo, hi = min(s0.min(), s1.min()), max(s0.max(), s1.max())
        grid = np.linspace(lo, hi, 300)
        bins = np.linspace(lo, hi, 60)

        # ① 히스토그램에 같은 평균·표준편차의 정규분포 곡선을 겹쳐 그림 → 모양이 얼마나 다른지 확인
        ax = row[0]
        for x, col, lab in ((s0, C0_COLOR, "c=0"), (s1, C1_COLOR, "c=1")):
            ax.hist(x, bins=bins, density=True, color=col, alpha=0.35, edgecolor="#fcfcfb", linewidth=0.5)
            ax.plot(grid, stats.norm.pdf(grid, x.mean(), x.std(ddof=1)), color=col, linewidth=2, label=f"{lab} (n={len(x):,})")
        ax.set_title(f"{r['condition']}: s distribution + fitted normal", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("similarity s", fontsize=8, color=MUTED)
        ax.legend(fontsize=8, frameon=False)

        # ② QQ 그림: 표준화한 s의 분위수 vs 정규분포 분위수. 점선 위에 있으면 정규분포
        ax = row[1]
        for x, col, lab in ((s0, C0_COLOR, "c=0"), (s1, C1_COLOR, "c=1")):
            z = (np.sort(x) - x.mean()) / x.std(ddof=1)
            q = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
            step = max(1, len(z) // 2000)
            ax.plot(q[::step], z[::step], ".", color=col, markersize=3, label=lab)
        ax.plot([-4, 4], [-4, 4], color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-5, 5)
        ax.set_title("QQ plot (standardised; on the dashed line = normal)", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("normal quantile", fontsize=8, color=MUTED)
        ax.legend(fontsize=8, frameon=False)

        # ③ 보정 곡선: 실제 정답 비율(점) vs 로지스틱 예측(실선) vs 등분산 정규 가정(점선)
        ax = row[2]
        ax.plot(centers, emp, "o", color=INK, markersize=5, label="observed P(c=1)")
        ax.plot(centers, pred_fit, color=C1_COLOR, linewidth=2, label=f"fitted logistic (ECE {r['logistic_fit']['ece']:.3f})")
        ax.plot(centers, pred_lda, color=C0_COLOR, linewidth=2, linestyle="--", label=f"equal-var normal (ECE {r['lda_implied']['ece']:.3f})")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("P(c=1 | s): observed vs logistic", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("similarity s (20 quantile bins)", fontsize=8, color=MUTED)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        for a in row:
            style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs="+")  # 관측 파일 여러 개 (조건별)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    # 파일마다 읽고 분석
    results = [analyze(*load(path), rng) for path in args.dumps]
    os.makedirs(args.out_dir, exist_ok=True)
    plot(results, os.path.join(args.out_dir, "observation_normality.png"))
    clean = [{k: v for k, v in r.items() if k != "_plot"} for r in results]
    with open(os.path.join(args.out_dir, "observation_normality.json"), "w") as f:
        json.dump(clean, f, indent=2)

    # 화면에 요약 표 출력: 클래스별 통계, 그다음 조건별 로지스틱 비교
    hdr = f"{'condition':10s} {'class':5s} {'n':>7s} {'mean':>6s} {'sd':>6s} {'skew':>6s} {'kurt':>6s}"
    print(hdr)
    for r in clean:
        for cls in ("c0", "c1"):
            st = r[cls]
            print(f"{r['condition']:10s} {cls:5s} {st['n']:7d} {st['mean']:6.3f} {st['sd']:6.3f} {st['skew']:6.2f} {st['excess_kurtosis']:6.2f}")
    print()
    print(f"{'condition':10s} {'sd1/sd0':>7s} {'AUC':>6s} {'γ fit':>7s} {'γ LDA':>7s} {'t fit':>6s} {'t LDA':>6s} {'ECE fit':>7s} {'ECE LDA':>7s}")
    for r in clean:
        lf, ld = r["logistic_fit"], r["lda_implied"]
        print(f"{r['condition']:10s} {r['sd_ratio_c1_over_c0']:7.2f} {r['auc']:6.3f} {lf['gamma']:7.1f} {ld['gamma']:7.1f} "
              f"{lf['t_hat']:6.3f} {ld['t_hat']:6.3f} {lf['ece']:7.3f} {ld['ece']:7.3f}")


if __name__ == "__main__":
    main()
