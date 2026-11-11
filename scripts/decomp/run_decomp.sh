#!/usr/bin/env bash
# [cls-scale-decomp] 바닐라 vCache ↔ vCache(MVR) 차이 분해 실행기 (DECOMP 부분 3 · 5절)
#
# 사용 예 (RunPod, 저장소 루트에서):
#   DATA=/workspace/data/lmarena_test_seed42_repro.parquet DS=lm OUT=/workspace/decomp/lm REP=1 \
#     bash scripts/decomp/run_decomp.sh                 # 반복 1, 6칸 전부 (V P1 P2 P3 P4 P5)
#   ... REP=3 bash scripts/decomp/run_decomp.sh P4 P5   # 반복 3, 일부 칸만
#
# 환경 변수: DATA(순서 1 파일, 필수) DS(cls|lm|sq, 필수) OUT(필수) REP(반복 번호, 1)
#            SEED(기본 = REP) DELTA(0.01) SLEEP(0.02) DEVICE(cuda)
# 반복 k = 순서 k + 시드 k (팀 규칙). 순서 k 파일은 OUT/data/order_k.parquet 로 만든다(make_order.py).
# 결과: OUT/r<REP>/<칸>.json · .log · .meta.json. 이미 끝난 칸은 건너뛴다. 한 Pod에서 한 번에 하나씩만 돌린다.
set -euo pipefail
: "${DATA:?DATA=<order-1 test parquet>}" "${DS:?DS=cls|lm|sq}" "${OUT:?OUT=<output dir>}"
REP="${REP:-1}"; SEED="${SEED:-$REP}"
DELTA="${DELTA:-0.01}"; SLEEP="${SLEEP:-0.02}"; DEVICE="${DEVICE:-cuda}"
if [ "$#" -gt 0 ]; then CELLS=("$@"); else CELLS=(V P1 P2 P3 P4 P5); fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$(cd "$(dirname "$DATA")" && pwd)/$(basename "$DATA")"
mkdir -p "$OUT/data" "$OUT/r$REP"
OUT="$(cd "$OUT" && pwd)"
if [ "$REP" != "1" ]; then
  ORDER_DATA="$OUT/data/order_$REP.parquet"
  [ -s "$ORDER_DATA" ] || python "$ROOT/scripts/decomp/make_order.py" "$DATA" "$REP" "$ORDER_DATA"
  DATA="$ORDER_DATA"
fi
RUN_OUT="$OUT/r$REP"
cd "$ROOT/mvr-cache"

case "$DS" in
  cls) EVAL=(--llm-col response_llama_3_8b --similarity-evaluator string) ;;
  lm|sq) EVAL=(--similarity-evaluator benchmark_id_set) ;;
  *) echo "DS must be cls|lm|sq" >&2; exit 1 ;;
esac
COMMON=(--dataset "$DATA" --delta "$DELTA" --seed "$SEED" --sleep "$SLEEP" "${EVAL[@]}")
# 분할 0 = 자르는 지점 0개 → 조각 1개(문장 전체).
SPLIT=(--splitter-mode prefix_rule --rule-max-cuts 0 --rule-cap-scope all
       --use-cached-candidate-segments --splitter-device "$DEVICE")

# 후보 찾기: top_k = CLS 포함 벡터(HNSW, 바닐라와 같음), multivector_top_k = 조각 벡터(CLS 제외, vCache(MVR)와 같음)
cell_args() {
  case "$1" in
    V)  echo "VANILLA" ;;                                                                         # 바닐라 vCache (main 판정 코드)
    P1) echo "--candidate-selection top_k --candidate-k 1 --segment-include-cls --score-scale raw" ;;   # CLS 포함 · cos · 후보 1 (= V를 MVR 코드로)
    P2) echo "--candidate-selection top_k --candidate-k 1 --segment-include-cls --score-scale unit" ;;  # CLS 포함 · (cos+1)/2 · 후보 1
    P3) echo "--candidate-selection multivector_top_k --candidate-k 1 --score-scale raw" ;;             # CLS 제외 · cos · 후보 1
    P4) echo "--candidate-selection multivector_top_k --candidate-k 1 --score-scale unit" ;;            # CLS 제외 · (cos+1)/2 · 후보 1
    P5) echo "--candidate-selection multivector_top_k --candidate-k 10 --score-scale unit" ;;           # CLS 제외 · (cos+1)/2 · 후보 10 (= vCache(MVR))
    *) echo "unknown cell $1" >&2; return 1 ;;
  esac
}

for c in "${CELLS[@]}"; do
  out_json="$RUN_OUT/$c.json"
  if [ -s "$out_json" ]; then echo "[skip] r$REP $c (done)"; continue; fi
  a="$(cell_args "$c")"
  if [ "$a" = "VANILLA" ]; then
    cmd=(python benchmarks/eval_sembenchmark_verified.py "${COMMON[@]}" --device "$DEVICE" --output-json "$out_json")
  else
    # shellcheck disable=SC2206
    cmd=(python benchmarks/eval_sembenchmark_verified_splitter.py "${COMMON[@]}" "${SPLIT[@]}" $a --output-json "$out_json")
  fi
  python - "$RUN_OUT/$c.meta.json" "$DATA" "$c" "$REP" "$SEED" "${cmd[@]}" <<'PY'
import hashlib, json, subprocess, sys, datetime
out, data, cell, rep, seed, cmd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6:]
def sh(x):
    try: return subprocess.run(x, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as e: return f"ERR {e}"
h = hashlib.sha256()
with open(data, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
meta = {"cell": cell, "rep": rep, "order": rep, "seed": seed, "cmd": cmd, "data": data, "data_sha256_12": h.hexdigest()[:12],
        "git_commit": sh("git rev-parse --short HEAD"), "git_dirty": bool(sh("git status --porcelain -- vcache benchmarks")),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "versions": sh("python -c \"import torch,transformers;print(torch.__version__, torch.version.cuda, transformers.__version__)\""),
        "start": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(meta, open(out, "w"), ensure_ascii=False, indent=2)
print(f"[meta] r{rep} {cell}: seed={seed} gpu={meta['gpu']} commit={meta['git_commit']} data={meta['data_sha256_12']}")
PY
  echo "[run] r$REP $c: ${cmd[*]}"
  start=$(date +%s)
  "${cmd[@]}" 2>&1 | tee "$RUN_OUT/$c.log"
  echo "[done] r$REP $c in $(( ($(date +%s) - start) / 60 )) min"
done
