"""판정자 실험 E: 과신 모델 만들기. learned.json 의 intercept 에 Δ 를 더해 logit p' = logit p + Δ 가 되게 한다.

특징은 표준화되어 있으므로 intercept 만 바꾸면 모든 요청의 logit 이 Δ 만큼 올라간다.
판정자의 순서(어느 요청이 더 안전한지)는 그대로이고, 확률만 부풀려진다.
코드는 바꾸지 않고 --consensus-model 에 이 파일을 넘긴다.

사용: python scripts/judge_cap/make_overconf_model.py --model learned.json --shift 2 --out learned_overconf_d2.json
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--shift", type=float, required=True, help="logit 에 더할 Δ (> 0 이면 과신)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
m = json.load(open(args.model, encoding="utf-8"))
m["intercept"] = float(m["intercept"]) + args.shift
m["judge_cap_overconf"] = {"source": args.model, "logit_shift": args.shift}
json.dump(m, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"intercept {m['intercept'] - args.shift:.4f} -> {m['intercept']:.4f}, 저장 {args.out}")
