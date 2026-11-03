"""Generate notebooks/mvr_colab_judge_cap_E.ipynb (judge experiment E). 사용: python scripts/judge_cap/gen_notebook_E.py notebooks/mvr_colab_judge_cap_E.ipynb"""
import json
import sys

OUT = sys.argv[1]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
# 판정자 실험 E: 한도 스트레스 — 과신하는 판정자에서 한도가 오류를 막는가

설계: 브랜치 `judge-cap-overconfident`의 `JUDGE_CAP.md` (실행 전 고정). 이 노트북은 그 설계를 그대로 실행한다.

**"모두 실행" 해도 된다.** 끝난 단계 · 실행은 Drive에 남아 건너뛰므로, 세션이 끊기거나 컴퓨팅 단위를 다시 사서 열면 **다시 "모두 실행"** 하면 이어서 돈다.

| 단계 | 하는 일 | 다시 열면 |
|---|---|---|
| §0 | GPU(L4) 확인 · Drive 작업 폴더 찾기 | 매번 |
| §1 | 재사용할 기록(U_s1 · Jmed_s1 · JC1L_s1) 찾기 · GPU 같은지 확인 | 매번 (다르면 세 기록을 다시 돌릴 목록에 넣는다) |
| §2 | 저장소 `8cfaa0b` + 패치 3개 (지문 확인) · 평가 환경 | 매번 |
| §3 | 데이터 (JC1L_s1이 쓴 test 파일을 Drive에서 찾아 복사, 행 수 확인) | 매번 |
| §4 | **사전 기록**: p_low 사전 점검 → `PREREG.json` | 처음 한 번만 (있으면 읽기만, 덮어쓰지 않는다) |
| §5 | 본 실행: Jlow_s1 → JlowC35_s1 → JlowC10_s1 (+ 선택 Jover · JoverC35) | 끝난 실행은 건너뛴다 |
| §6 | 결과 · 판정 기준 | 끝난 실행만으로 집계 |
| §7 | 무료 집계: 오류율 95% 상한 > 명목 상한 횟수 | 매번 |

- 바꾸는 것은 JC1L_s1 · Jmed_s1의 **실행 인자 중 `--consensus-pstar` · `--consensus-online-budget` · (선택) `--consensus-model`뿐**이다. 나머지 인자는 기존 기록의 `_meta.json`에서 그대로 가져온다
- 결과는 Drive `mvr_eval/pair_judge/judge_e_cap/`에 쌓인다. 데이터 · 결과는 GitHub에 올리지 않는다
""")

md("## 0. 설정 · GPU · Drive")
code(r"""
REQUIRE_L4 = True        # 기존 기록(B · C · D · E 기준선)이 L4 다
MD_OVERRIDE = ''         #@param {type:"string"}  자동 탐색이 실패할 때만 작업 폴더 경로
DATA_OVERRIDE = ''       #@param {type:"string"}  test 파일을 못 찾을 때만 Drive 경로
BASE_META_OVERRIDE = {'U': '', 'Jmed': '', 'JC1L': ''}   # 기존 기록 _meta.json 경로 (자동 탐색 실패 시)
P_LOW_OVERRIDE = None    # 팀원과 p_low 값을 정했으면 숫자 (예: 0.5). None 이면 사전 점검 규칙으로 정한다
RUN_OVERCONF = False     # 선택 조건 Jover · JoverC35 (팀원 확인 후 True)
RUN_FREE_AGG = True      # §7 무료 집계
AUTO_SHUTDOWN = True     # 셀 에러 · 끝 · 시간 초과 시 런타임을 끈다 (컴퓨팅 단위 절약)
MAX_HOURS = 23.5
BRANCH = 'judge-cap-overconfident'

import subprocess, json, os, datetime, hashlib, shutil, glob, time, threading, unicodedata, re
def _sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()
GPU_NAME = _sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines()[0]
print('GPU:', _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"))
if REQUIRE_L4:
    assert 'L4' in GPU_NAME, f'GPU 가 {GPU_NAME} 다 — 런타임 유형을 L4 로 바꾼다 (기존 기록과 같은 GPU)'

from google.colab import drive
# 인증 창(팝업)에서 '허용'을 눌러야 끝난다. 큰 공유 드라이브는 마운트가 느려 기본 2분을 넘길 수 있어 10분으로 늘린다
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive', force_remount=True, timeout_ms=600000)
NEED = ('mvr_eval', 'mvr_patches', 'mvr_splits')
def _cands():
    roots = ['/content/drive/MyDrive'] + sorted(glob.glob('/content/drive/Shareddrives/*'))
    out = []
    for r in roots:
        for d1 in sorted(os.listdir(r)) if os.path.isdir(r) else []:
            p1 = f'{r}/{d1}'
            if not os.path.isdir(p1):
                continue
            for p in [p1] + [f'{p1}/{d2}' for d2 in sorted(os.listdir(p1)) if os.path.isdir(f'{p1}/{d2}')]:
                if all(os.path.isdir(f'{p}/{n}') for n in NEED):
                    out.append(p)
    return out
if MD_OVERRIDE:
    MD = MD_OVERRIDE
else:
    C = _cands()
    pref = [p for p in C if '캡스톤디자인' in unicodedata.normalize('NFC', p)]
    assert len(pref) == 1, (f'작업 폴더를 하나로 못 정했다: {C}\n'
                            '→ 공유 문서함의 "캡스톤디자인(1)" 을 내 드라이브에 바로가기로 추가했는지 확인, 아니면 MD_OVERRIDE 에 경로를 넣는다')
    MD = pref[0]
assert all(os.path.isdir(f'{MD}/{n}') for n in NEED), MD
PATCH_DIR = f'{MD}/mvr_patches'
E_DIR = f'{MD}/mvr_eval/pair_judge/judge_e_cap'      # 이 실험의 모든 산출물
RUN_DIR = f'{E_DIR}/runs'
for d in (E_DIR, RUN_DIR):
    os.makedirs(d, exist_ok=True)
LOC = '/content/runs'
os.makedirs(LOC, exist_ok=True)
print('작업 폴더 MD =', MD, '\n산출물 =', E_DIR)
""")

md("### 0-1. 자동 종료 — 셀 에러 · 마지막 셀 · 시간 초과 시 런타임을 끈다 (이유는 Drive `judge_e_cap/_session_end.txt`)")
code(r"""
from IPython import get_ipython
_T0 = time.time()
AUTO = {'reason': None}
def _shutdown(reason):
    try:
        with open(f'{E_DIR}/_session_end.txt', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {GPU_NAME} {reason}\n")
    except Exception as e:
        print('종료 기록 실패', e)
    from google.colab import drive as _drive, runtime as _runtime
    try:
        _drive.flush_and_unmount()
    except Exception as e:
        print('unmount 실패', e)
    _runtime.unassign()
def _watch():
    while AUTO['reason'] is None:
        if time.time() - _T0 > MAX_HOURS * 3600:
            AUTO['reason'] = f'시간 초과 ({MAX_HOURS}시간)'
            break
        time.sleep(20)
    time.sleep(60)
    _shutdown(AUTO['reason'])
def _on_cell(result):
    err = result.error_before_exec or result.error_in_exec
    if err is not None and AUTO['reason'] is None:
        AUTO['reason'] = f'셀 에러: {type(err).__name__}: {str(err)[:300]}'
if AUTO_SHUTDOWN:
    get_ipython().events.register('post_run_cell', _on_cell)
    threading.Thread(target=_watch, daemon=True).start()
    print(f'✅ 자동 종료 켜짐 — 셀 에러 · 마지막 셀 · {MAX_HOURS}시간 초과 시')
""")

md(r"""
## 1. 재사용할 기록 찾기 · GPU 확인

- Drive `mvr_eval` 아래에서 `U_s1_meta.json` · `Jmed_s1_meta.json` · `JC1L_s1_meta.json`을 찾는다 (Classification 실행만, SearchQueries `sq_` 데이터는 뺀다)
- 기록의 GPU가 지금 GPU와 다르면 **그 기록을 이 GPU에서 다시 돌린다** (설계 0절: 3실행 추가)
- Drive 경로는 마운트 위치에 따라 앞부분이 다를 수 있어, `mvr_eval` 앞을 지금 작업 폴더로 바꿔 찾는다
""")
code(r"""
def remap(p):
    # 기록 안의 Drive 경로(예: /content/drive/MyDrive/mvr_eval/...)를 지금 작업 폴더 기준으로 바꾼다
    if p is None or os.path.exists(p):
        return p
    i = p.find('/mvr_eval/')
    return f'{MD}{p[i:]}' if i >= 0 and os.path.exists(f'{MD}{p[i:]}') else p
def _arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
def _is_cls(meta):
    ds = os.path.basename(_arg(meta.get('args', []), '--dataset', '') or '')
    return not ds.startswith('sq_')
ALL_META = sorted(glob.glob(f'{MD}/mvr_eval/**/*_meta.json', recursive=True))
print(f'Drive mvr_eval 의 _meta.json {len(ALL_META)}개')
BASE = {}
for key, name in [('U', 'U_s1'), ('Jmed', 'Jmed_s1'), ('JC1L', 'JC1L_s1')]:
    if BASE_META_OVERRIDE.get(key):
        cands = [BASE_META_OVERRIDE[key]]
    else:
        cands = [p for p in ALL_META if os.path.basename(p) == f'{name}_meta.json' and not p.startswith(E_DIR)]
        cands = [p for p in cands if _is_cls(json.load(open(p, encoding='utf-8')))]
    assert len(cands) == 1, f'{name}: 기록을 하나로 못 정했다 {cands} — BASE_META_OVERRIDE["{key}"] 에 경로를 넣는다'
    m = json.load(open(cands[0], encoding='utf-8'))
    jl = cands[0][:-len('_meta.json')] + '.jsonl'
    BASE[key] = {'name': name, 'meta_path': cands[0], 'meta': m, 'args': list(m['args']),
                 'jsonl': jl if os.path.exists(jl) else None, 'gpu': m.get('gpu'), 'rerun': m.get('gpu') != GPU_NAME}
    print(f"  {name}: {cands[0]}\n     GPU {m.get('gpu')} · 커밋 {str(m.get('commit'))[:7]} · 판정 기록 {'있음' if BASE[key]['jsonl'] else '없음'}"
          f"{' → GPU 가 달라 이 GPU 에서 다시 돌린다' if BASE[key]['rerun'] else ''}")
RERUN = [k for k in BASE if BASE[k]['rerun']]

# 설계와 맞는지: JC1L = calibrated · z 1 · probe 0.1 · prior 20 · 폭 0.02 · label only, Jmed 와 같은 모델 · p*
a = BASE['JC1L']['args']
assert '--consensus-online-label-only' in a, 'JC1L_s1 인자에 --consensus-online-label-only 가 없다 — JC1L 기록이 맞는지 확인'
assert _arg(a, '--consensus-online-mode') == 'calibrated' and float(_arg(a, '--consensus-online-z')) == 1.0, a
assert float(_arg(a, '--consensus-online-budget')) == 0.0035, _arg(a, '--consensus-online-budget')
for f, v in [('--consensus-online-probe', 0.1), ('--consensus-online-prior', 20.0), ('--consensus-online-bin-width', 0.02)]:
    assert float(_arg(a, f, v)) == v, (f, _arg(a, f))
JMED_PSTAR = float(_arg(BASE['Jmed']['args'], '--consensus-pstar'))
JMED_MODEL = remap(_arg(BASE['Jmed']['args'], '--consensus-model'))
assert float(_arg(a, '--consensus-pstar')) == JMED_PSTAR, 'JC1L 과 Jmed 의 p* 가 다르다'
assert remap(_arg(a, '--consensus-model')) == JMED_MODEL, 'JC1L 과 Jmed 의 판정자 모델이 다르다'
assert os.path.exists(JMED_MODEL), JMED_MODEL
print(f'\n✅ Jmed p* = {JMED_PSTAR} · 모델 {JMED_MODEL}')
print('다시 돌릴 기존 기록:', [BASE[k]['name'] for k in RERUN] or '없음 (재사용)')
""")

md("## 2. 저장소 (`8cfaa0b` + 패치 3개) · 평가 환경")
code(r"""
REPO = '/content/repo_M'
REPO_COMMIT = '8cfaa0b86c4cb6a650cb207caa053fc1932bda7b'
PATCHES = [('pair_judge.patch', '07c4751c4be0'), ('online_budget.patch', 'b07856972ddc'), ('probe_label_only.patch', '92ca6d07db0a')]
sha12 = lambda p: hashlib.sha1(open(p, 'rb').read()).hexdigest()[:12]
def _git(*a, cwd=None):
    r = subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(a)} 실패:\n{r.stderr}"
    return r.stdout.strip()
for f, want in PATCHES:
    assert os.path.exists(f'{PATCH_DIR}/{f}'), f'{PATCH_DIR}/{f} 가 없다'
    assert sha12(f'{PATCH_DIR}/{f}') == want, f'{f} 지문 {sha12(f"{PATCH_DIR}/{f}")} ≠ {want}'
PATCH_SHAS = {f: w for f, w in PATCHES}
if os.path.exists(REPO):
    shutil.rmtree(REPO)
_git('clone', '-q', 'https://github.com/PKU-SDS-lab/MVR-Cache.git', REPO)
_git('checkout', '-q', REPO_COMMIT, cwd=REPO)
for f, _ in PATCHES:
    _git('apply', '--whitespace=nowarn', f'{PATCH_DIR}/{f}', cwd=REPO)
src = open(f'{REPO}/mvr-cache/vcache/vcache_policy/strategies/verified_splitter.py').read()
assert 'consensus_top_k' in src and '_online_budget_admit' in src and '_ob_label_only' in src
# 이 실험의 설계 · 사전 점검 스크립트
TOOLS = '/content/judge_cap'
if os.path.exists(TOOLS):
    shutil.rmtree(TOOLS)
_git('clone', '-q', '--depth', '1', '-b', BRANCH, 'https://github.com/sgwag6799/MVR-Cache.git', TOOLS)
TOOLS_COMMIT = _git('rev-parse', 'HEAD', cwd=TOOLS)
print(f'✅ 저장소 M {REPO_COMMIT[:7]} + 패치 {list(PATCH_SHAS.values())} · 실험 도구 {BRANCH}@{TOOLS_COMMIT[:7]}')
""")
code(r"""
def _run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
_run('pip install -q uv')
_run('uv python install 3.11')
_run('uv venv -q --clear --python 3.11 /content/py311')
PY = '/content/py311/bin/python'
_run(f'uv pip install -q --python {PY} '
     'torch==2.8.0 numpy==1.26.4 pandas==2.3.3 pyarrow==22.0.0 '
     'transformers==4.57.3 sentence-transformers==4.1.0 datasets==4.4.2 '
     'rl4co==0.6.0 torchrl==0.10.0 tensordict==0.10.0 lightning==2.5.5 '
     'scikit-learn==1.8.0 scipy==1.15.3 statsmodels==0.14.6 tqdm==4.67.1 '
     'typing-extensions pybind11 setuptools wheel')
_run(f'uv pip install -q --python {PY} {REPO}/mvr-cache/vcache/vcache_core/cache/embedding_store/hnswlib')

STUBS = '/content/stubs'
_STUB_SRC = ("def __getattr__(name):\n"
             "    if name.startswith('__'):\n"
             "        raise AttributeError(name)\n"
             "    return type(name, (), {})\n")
for mod in ['vllm', 'langchain', 'langchain/schema', 'langchain_community',
            'langchain_community/embeddings', 'openai', 'chromadb', 'faiss']:
    os.makedirs(f'{STUBS}/{mod}', exist_ok=True)
    open(f'{STUBS}/{mod}/__init__.py', 'w').write(_STUB_SRC)
def env_for(**extra):
    return {**os.environ,
            'PYTHONPATH': f'{STUBS}:{REPO}/mvr-cache:{REPO}/rl-training-algorithm',
            'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
            'MPLBACKEND': 'Agg', 'HF_ENDPOINT': 'https://huggingface.co', 'HF_HOME': '/content/hf',
            **{k: str(v) for k, v in extra.items()}}
SEEDED = '/content/seeded_run.py'
open(SEEDED, 'w').write('import os, random, runpy\nimport numpy as np\n'
                        'seed = int(os.environ.get("EVAL_SEED", "0"))\n'
                        'random.seed(seed)\nnp.random.seed(seed)\n'
                        'runpy.run_path(os.environ["RUN_EVAL_PY"], run_name="__main__")\n')
r = subprocess.run([PY, '-c', "import sys, torch, hnswlib; print(torch.cuda.is_available(), hasattr(hnswlib.Index, 'knn_query_skipping_duplicates_with_parent'))"],
                   env=env_for(), capture_output=True, text=True)
assert r.stdout.split() == ['True', 'True'], r.stdout + r.stderr[-1500:]
h = subprocess.run([PY, 'benchmarks/eval_sembenchmark_verified_splitter.py', '--help'],
                   cwd=f'{REPO}/mvr-cache', env=env_for(), capture_output=True, text=True).stdout
assert '--consensus-online-budget' in h and '--consensus-online-label-only' in h and '--consensus-pstar' in h
assert os.path.exists(f'{REPO}/run_eval.py'), 'run_eval.py 가 없다 (pair_judge.patch)'
print('✅ 환경 · CUDA · hnswlib 포크 · 한도 옵션')
""")

md(r"""
## 3. 데이터 — JC1L_s1이 쓴 test 파일

- JC1L_s1 인자의 `--dataset`(로컬 경로)과 같은 이름의 파일을 Drive `mvr_splits`에서 찾아 같은 로컬 경로로 복사한다
- 행 수가 기존 판정 기록(U_s1 · Jmed_s1 · JC1L_s1)의 행 수와 같은지 확인한다. 같은 폴더에 `manifest.json`이 있으면 지문도 확인한다
""")
code(r"""
import pandas as pd, numpy as np
TEST_P = _arg(BASE['JC1L']['args'], '--dataset')
for k in BASE:
    assert _arg(BASE[k]['args'], '--dataset') == TEST_P, f'{k} 의 --dataset 이 JC1L 과 다르다'
bn = os.path.basename(TEST_P)
if DATA_OVERRIDE:
    src = DATA_OVERRIDE
else:
    names = {bn, re.sub(r'^[a-z0-9]+_', '', bn)}        # 예: cls_test.parquet → test.parquet 도 후보
    cs = [p for p in glob.glob(f'{MD}/mvr_splits/**/*.parquet', recursive=True)
          if os.path.basename(p) in names and '/sq' not in p]
    assert len(cs) == 1, f'test 파일을 하나로 못 정했다 ({bn}): {cs} — DATA_OVERRIDE 에 Drive 경로를 넣는다'
    src = cs[0]
os.makedirs(os.path.dirname(TEST_P), exist_ok=True)
shutil.copy2(src, TEST_P)
_te = pd.read_parquet(TEST_P)
N_TEST = len(_te)
man = os.path.join(os.path.dirname(src), 'manifest.json')
if os.path.exists(man) and 'id' in _te and 'dataset_name' in _te:
    # Classification manifest: fingerprint = sha1("dataset_name:id" 를 행 순서대로 이은 것)[:12]
    fp = hashlib.sha1(','.join(f'{a}:{b}' for a, b in zip(_te['dataset_name'], _te['id'])).encode()).hexdigest()[:12]
    want = json.load(open(man, encoding='utf-8')).get('fingerprint')
    assert fp == want, f'지문 {fp} ≠ manifest {want}'
    print('지문 확인', fp)
for k in BASE:
    if BASE[k]['jsonl']:
        n = sum(1 for _ in open(BASE[k]['jsonl'], encoding='utf-8'))
        assert n == N_TEST, f"{BASE[k]['name']} 판정 기록 {n}행 ≠ test {N_TEST}행"
print(f'✅ test {N_TEST:,}행 ({src} → {TEST_P})')
del _te
""")

md("## 4. 사전 기록 (실행 전) — 처음 한 번만 쓰고, 있으면 읽기만 한다")
code(r"""
def set_arg(args, flag, value):
    a = list(args)
    if flag in a:
        a[a.index(flag) + 1] = str(value)
    else:
        a += [flag, str(value)]
    return a
PREREG_P = f'{E_DIR}/PREREG.json'
def _precheck(mode, extra):
    jl = BASE['Jmed']['jsonl']
    assert jl, 'Jmed_s1 판정 기록(.jsonl)이 없다 — 사전 점검을 할 수 없다'
    cmd = ['python3', f'{TOOLS}/scripts/judge_cap/precheck_plow.py', '--log', jl, '--model', JMED_MODEL, '--mode', mode, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    m = re.search(r'선택 = ([-+0-9.eE]+)', r.stdout)
    return r.stdout, (float(m.group(1)) if m else None)
if not os.path.exists(PREREG_P):
    out_p, p_low = _precheck('pstar', [])
    print(out_p)
    rule = '사전 점검 규칙 (0.5 → 0.3 → 0.1 중 근사 추가 오답 ≥ 0.70% 인 첫 값)'
    if P_LOW_OVERRIDE is not None:
        p_low, rule = float(P_LOW_OVERRIDE), '팀원과 정한 값 (P_LOW_OVERRIDE)'
    assert p_low is not None and p_low < JMED_PSTAR, f'p_low 를 정하지 못했다 ({p_low})'
    pre = {'작성': datetime.datetime.now().isoformat(timespec='seconds'), 'GPU': GPU_NAME,
           '설계': f'{BRANCH}@{TOOLS_COMMIT} JUDGE_CAP.md', '저장소': REPO_COMMIT, '패치': PATCH_SHAS,
           'Jmed p*': JMED_PSTAR, '판정자 모델': JMED_MODEL, '판정자 모델 sha': sha12(JMED_MODEL),
           'p_low': p_low, 'p_low 규칙': rule, 'p_low 사전 점검 출력': out_p,
           '판정 기준': {'① 전제': '한도 없음(Jlow · Jover) 추가 오답 > 0.35% (공식 39K)',
                     '② 주 기준': 'JlowC35 · JoverC35 추가 오답 ≤ 0.42%, JlowC10 ≤ 0.12%',
                     '③ 보조': 'blocked + bin_closed 로 막은 오답 > 0 이고 probe 로 막은 오답보다 많다',
                     '④ 유효성': '짝끼리 vCache 판정 이유(reason) 순서 · 자체 재사용 수가 같다'},
           '추가 오답 정의': '전환(hit 이고 policy_reason 이 policy_ 로 시작) 중 오답 ÷ 전체 요청 (i ≥ 2000)',
           '선택 조건 Jover': RUN_OVERCONF}
    if RUN_OVERCONF:
        out_d, dlt = _precheck('shift', ['--pstar', str(JMED_PSTAR)])
        print(out_d)
        assert dlt is not None, 'Δ 를 정하지 못했다'
        pre.update({'Δ': dlt, 'Δ 사전 점검 출력': out_d})
    json.dump(pre, open(PREREG_P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('✅ 사전 기록 저장 — 이 값으로 실행한다 (덮어쓰지 않는다)')
PRE = json.load(open(PREREG_P, encoding='utf-8'))
assert PRE['판정자 모델 sha'] == sha12(JMED_MODEL), '판정자 모델 파일이 사전 기록 뒤 바뀌었다'
P_LOW = float(PRE['p_low'])
if RUN_OVERCONF and 'Δ' not in PRE:
    raise RuntimeError('RUN_OVERCONF=True 인데 사전 기록에 Δ 가 없다 — 사전 기록을 처음 쓸 때부터 켜야 한다 (결과를 본 뒤 추가하지 않는다)')
OVER_MODEL = f'{E_DIR}/learned_overconf.json'
if RUN_OVERCONF and not os.path.exists(OVER_MODEL):
    r = subprocess.run(['python3', f'{TOOLS}/scripts/judge_cap/make_overconf_model.py', '--model', JMED_MODEL,
                        '--shift', str(PRE['Δ']), '--out', OVER_MODEL], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
print(json.dumps({k: PRE[k] for k in ('작성', 'GPU', 'Jmed p*', 'p_low', 'p_low 규칙')}, ensure_ascii=False))

# 조건: 기존 기록 인자에서 p* · 한도 · (선택) 모델만 바꾼다
JA, CA = BASE['Jmed']['args'], BASE['JC1L']['args']
CONDS = {}
for k in RERUN:                                    # GPU 가 달라 다시 돌릴 기존 기록 (인자 그대로)
    CONDS[BASE[k]['name']] = list(BASE[k]['args'])
CONDS['Jlow_s1'] = set_arg(JA, '--consensus-pstar', repr(P_LOW))
CONDS['JlowC35_s1'] = set_arg(set_arg(CA, '--consensus-pstar', repr(P_LOW)), '--consensus-online-budget', '0.0035')
CONDS['JlowC10_s1'] = set_arg(set_arg(CA, '--consensus-pstar', repr(P_LOW)), '--consensus-online-budget', '0.001')
if RUN_OVERCONF:
    CONDS['Jover_s1'] = set_arg(JA, '--consensus-model', OVER_MODEL)
    CONDS['JoverC35_s1'] = set_arg(CA, '--consensus-model', OVER_MODEL)
for n in CONDS:                                     # 판정자 모델 경로를 지금 작업 폴더 기준으로
    if '--consensus-model' in CONDS[n]:
        CONDS[n] = set_arg(CONDS[n], '--consensus-model', remap(_arg(CONDS[n], '--consensus-model')))
        assert os.path.exists(_arg(CONDS[n], '--consensus-model')), (n, _arg(CONDS[n], '--consensus-model'))
for n, a in CONDS.items():
    print(f"  {n}: p* {_arg(a, '--consensus-pstar', '-')} · 한도 {_arg(a, '--consensus-online-budget', '없음')} · 모델 {os.path.basename(str(_arg(a, '--consensus-model', '-')))}")
""")

md(r"""
## 5. 본 실행 (끝난 것은 건너뛴다)

- 한 실행 약 1시간 이상 (L4, sleep 0.02). 세션이 끊기면 진행 중이던 실행은 다음 세션에서 처음부터 다시 돈다
- 실행마다 `_meta.json`(GPU · 커밋 · 패치 지문 · 인자), `.jsonl`(판정 기록), `_budget.json`(한도 상태)을 Drive에 남긴다
""")
code(r"""
SEED = 1
MODULE = 'benchmarks.eval_sembenchmark_verified_splitter'
def run_eval(name, args):
    a = [x for x in args]
    for f in ('--output-json', '--timing-output'):
        if f in a:
            i = a.index(f); del a[i:i + 2]
    open(f'{LOC}/{name}.cmd', 'w').write(f'EVAL_SEED={SEED} {MODULE} ' + ' '.join(a) + '\n')
    cmd = [PY, SEEDED, MODULE, *a, '--output-json', f'{LOC}/{name}.json', '--timing-output', f'{LOC}/{name}_timing.txt']
    t0 = time.time()
    with open(f'{LOC}/{name}.log', 'w') as lf:
        proc = subprocess.Popen(cmd, cwd=f'{REPO}/mvr-cache', stdout=lf, stderr=subprocess.STDOUT,
                                env=env_for(EVAL_SEED=SEED, RUN_EVAL_PY=f'{REPO}/run_eval.py',
                                            DECISION_LOG=f'{LOC}/{name}.jsonl',
                                            ONLINE_BUDGET_STATE=f'{LOC}/{name}_budget.json'))
        while True:
            try:
                rc = proc.wait(timeout=300); break
            except subprocess.TimeoutExpired:
                lf.flush()
                try: shutil.copy2(f'{LOC}/{name}.log', f'{RUN_DIR}/{name}.log')
                except Exception: pass
    mins = (time.time() - t0) / 60
    json.dump({'name': name, 'seed': SEED, 'module': MODULE, 'gpu': GPU_NAME, 'minutes': round(mins, 1), 'rc': rc,
               'commit': REPO_COMMIT, 'patches': PATCH_SHAS, 'tools': f'{BRANCH}@{TOOLS_COMMIT}', 'args': a},
              open(f'{LOC}/{name}_meta.json', 'w'), ensure_ascii=False, indent=1)
    for f in glob.glob(f'{LOC}/{name}*'):
        shutil.copy2(f, RUN_DIR)
    n = sum(1 for _ in open(f'{LOC}/{name}.jsonl', encoding='utf-8')) if os.path.exists(f'{LOC}/{name}.jsonl') else 0
    if rc != 0 or n != N_TEST:
        raise RuntimeError(f'실행 실패 {name} (코드 {rc}, 판정 기록 {n}행) — {RUN_DIR}/{name}.log')
    open(f'{RUN_DIR}/{name}.done', 'w').close()
    return mins
def est_hours():
    ms = [json.load(open(f[:-5] + '_meta.json'))['minutes'] for f in glob.glob(f'{RUN_DIR}/*.done')]
    ms += [BASE[k]['meta'].get('minutes', 0) for k in BASE]
    return max(ms) / 60 * 1.15 if ms else 2.0
STATUS = {}
for name, args in CONDS.items():
    if os.path.exists(f'{RUN_DIR}/{name}.done'):
        STATUS[name] = '완료 (건너뜀)'; continue
    left = MAX_HOURS - (time.time() - _T0) / 3600
    if left < est_hours():
        STATUS[name] = f'남은 {left:.1f}시간 < 예상 {est_hours():.1f}시간 — 다음 세션에서 "모두 실행"'; break
    print(f"▶ {name} 시작 {time.strftime('%H:%M')}", flush=True)
    STATUS[name] = f'완료 {run_eval(name, args):.0f}분'
    print(f'  ✅ {name} {STATUS[name]}', flush=True)
print(json.dumps(STATUS, ensure_ascii=False, indent=1))
ALL_DONE = all(os.path.exists(f'{RUN_DIR}/{n}.done') for n in CONDS)
print('✅ 전부 끝' if ALL_DONE else '⏸ 남은 실행이 있다 — 다음 세션에서 "모두 실행"')
""")

md(r"""
## 6. 결과 · 판정 기준 (끝난 실행만으로 집계)

- 집계 구간은 공식 39K (`i ≥ 2000`). 추가 오답 = 전환 중 오답 ÷ 전체 요청 (0절 정의)
- 한도 조건에서 LLM으로 돌린 요청(`online_budget_*`) 중 1등 후보가 틀린 건수 = 한도가 막은 오답 (label only라 짝의 캐시 경로가 같다)
""")
code(r"""
from scipy.stats import beta as _beta
ub = lambda k, n: 1.0 if k >= n else float(_beta.ppf(0.95, k + 1, n - k))
def load_df(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding='utf-8') if l.strip()])
def rec_path(name):
    # 이 실험에서 돌린 것이 있으면 그것, 없으면 재사용 기록
    p = f'{RUN_DIR}/{name}.jsonl'
    if os.path.exists(f'{RUN_DIR}/{name}.done'):
        return p, json.load(open(f'{RUN_DIR}/{name}_meta.json', encoding='utf-8'))
    for k in BASE:
        if BASE[k]['name'] == name and not BASE[k]['rerun'] and BASE[k]['jsonl']:
            return BASE[k]['jsonl'], BASE[k]['meta']
    return None, None
def budget_state(name):
    for p in (f'{RUN_DIR}/{name}_budget.json',):
        if os.path.exists(p):
            return json.load(open(p, encoding='utf-8'))
    return None
OB = ('online_budget_blocked', 'online_budget_bin_closed', 'online_budget_probe')
D, ROWS = {}, []
for name in ['U_s1', 'Jmed_s1', 'JC1L_s1', 'Jlow_s1', 'JlowC35_s1', 'JlowC10_s1', 'Jover_s1', 'JoverC35_s1']:
    p, meta = rec_path(name)
    if p is None:
        continue
    d = load_df(p)
    assert len(d) == N_TEST, (name, len(d))
    pr = d['policy_reason'].fillna('').astype(str) if 'policy_reason' in d else pd.Series([''] * len(d))
    d['pr'] = pr
    d['fp'] = d['hit'].astype(bool) & (d['nn_correct'] != True)
    d['flip'] = d['hit'].astype(bool) & pr.str.startswith('policy_')
    D[name] = d
    g = d[d['i'] >= 2000]
    n = len(g)
    r = {'조건': name, 'GPU': meta.get('gpu'), 'n': n, '히트율%': 100 * g.hit.mean(), '오류율%': 100 * g.fp.mean(),
         '오류율 95%상한%': 100 * ub(int(g.fp.sum()), n), '전환': int(g.flip.sum()),
         '추가 오답%': 100 * (g.flip & g.fp).sum() / n, '자체 재사용': int((g.hit.astype(bool) & ~g.flip).sum())}
    for why in OB:
        m = g.pr == why
        r[f'{why[14:]} 건수'] = int(m.sum())
        r[f'{why[14:]} 중 오답'] = int((m & (g['nn_correct'] != True)).sum())
    st = budget_state(name)
    if st:
        r['spent/allowed'] = f"{st['spent']:.0f} / {st['allowed']:.0f}"
        adm = [b['lo'] for b, ok in zip(st['bins'], st['admitted_final']) if ok]
        r['실질 통과 임계값'] = min(adm) if adm else None
    ROWS.append(r)
R = pd.DataFrame(ROWS)
pd.set_option('display.width', 250)
print(R.round(3).to_string(index=False))
R.to_csv(f'{E_DIR}/results_E.csv', index=False)

def _v(name, col):
    x = R.loc[R['조건'] == name, col]
    return float(x.iloc[0]) if len(x) else None
print('\n■ 판정 기준 (실행 전 기록, 공식 39K)')
for free, caps in [('Jlow_s1', [('JlowC35_s1', 0.42), ('JlowC10_s1', 0.12)]), ('Jover_s1', [('JoverC35_s1', 0.42)])]:
    a0 = _v(free, '추가 오답%')
    if a0 is None:
        continue
    ok1 = a0 > 0.35
    print(f'① {free} 추가 오답 {a0:.3f}% > 0.35% : {"✅" if ok1 else "❌ 과신 유도 실패 → 이 짝은 판정하지 않는다"}')
    if not ok1:
        continue
    for cn, lim in caps:
        a1 = _v(cn, '추가 오답%')
        if a1 is None:
            print(f'   {cn}: 아직 없음'); continue
        bl = _v(cn, 'blocked 중 오답') + _v(cn, 'bin_closed 중 오답')
        pb = _v(cn, 'probe 중 오답')
        print(f'② {cn} 추가 오답 {a1:.3f}% ≤ {lim}% : {"✅" if a1 <= lim else "❌"}')
        print(f'③ {cn} 한도가 막은 오답 {bl:.0f} (blocked + bin_closed) > 0 이고 > probe {pb:.0f} : {"✅" if bl > 0 and bl > pb else "❌"}')
        same = float((D[free]['reason'].to_numpy() == D[cn]['reason'].to_numpy()).mean())
        print(f'④ {free} vs {cn} vCache 판정 이유 같은 비율 {same:.5f}, 자체 재사용 {_v(free, "자체 재사용"):.0f} vs {_v(cn, "자체 재사용"):.0f} : '
              f'{"✅" if same == 1.0 and _v(free, "자체 재사용") == _v(cn, "자체 재사용") else "⚠️ 다르다 — 원인을 먼저 기록"}')

# 참고: U δ 곡선 보간 (같은 오류율 히트율, 0절 3-7). U0125_s1 · U015_s1 기록이 Drive 에 있을 때만
curve = []
for nm in ['U_s1', 'U0125_s1', 'U015_s1', 'U020_s1']:
    if nm == 'U_s1' and 'U_s1' in D:
        g = D['U_s1'][D['U_s1']['i'] >= 2000]; curve.append((100 * g.fp.mean(), 100 * g.hit.mean())); continue
    ps = [q for q in ALL_META if os.path.basename(q) == f'{nm}_meta.json' and _is_cls(json.load(open(q, encoding='utf-8')))]
    jl = ps[0][:-len('_meta.json')] + '.jsonl' if len(ps) == 1 else None
    if jl and os.path.exists(jl):
        x = load_df(jl); x = x[x['i'] >= 2000]
        curve.append((100 * (x['hit'].astype(bool) & (x['nn_correct'] != True)).mean(), 100 * x['hit'].mean()))
if len(curve) >= 2:
    curve.sort()
    xs, ys = [c[0] for c in curve], [c[1] for c in curve]
    print('\n■ 참고: U δ 곡선 (오류율%, 히트율%)', [(round(a, 3), round(b, 2)) for a, b in curve])
    for _, r in R.iterrows():
        if r['오류율%'] < xs[0] or r['오류율%'] > xs[-1]:
            print(f"  {r['조건']}: 오류율 {r['오류율%']:.3f}% 가 곡선 범위 밖 — 보간하지 않는다"); continue
        base = float(np.interp(r['오류율%'], xs, ys))
        print(f"  {r['조건']}: 같은 오류율 U {base:.2f}% 대비 {r['히트율%'] - base:+.2f}%p")
""")

md("## 7. 무료 집계 — 지금까지 모든 실행에서 오류율 95% 상한 > 명목 상한 횟수 (명목 상한: 기반 δ, 판정자 δ + ε)")
code(r"""
if RUN_FREE_AGG:
    rows = []
    metas = sorted(set(glob.glob(f'{MD}/mvr_eval/**/*_meta.json', recursive=True)))
    for q in metas:
        jl = q[:-len('_meta.json')] + '.jsonl'
        if 'smoke' in q or 'collect' in os.path.basename(q) or not os.path.exists(jl):
            continue
        try:
            m = json.load(open(q, encoding='utf-8')); a = m.get('args', [])
            x = pd.read_json(jl, lines=True)
            if 'i' in x:
                x = x[x['i'] >= 2000]
            k, n = int((x['hit'].astype(bool) & (x['nn_correct'] != True)).sum()), len(x)
        except Exception as e:
            print('건너뜀', q, e); continue
        delta = float(_arg(a, '--delta', 0.01))
        eps = float(_arg(a, '--consensus-online-budget', 0.0035 if '--consensus-policy' in a else 0.0))
        u = ub(k, n)
        rows.append({'실행': q.replace(f'{MD}/mvr_eval/', '')[:-len('_meta.json')], 'n': n, '오류율%': 100 * k / n,
                     '95%상한%': 100 * u, '명목 상한%': 100 * (delta + eps), '상한>명목': u > delta + eps, '상한>δ': u > delta})
    FA = pd.DataFrame(rows)
    if len(FA):
        print(FA.round(3).to_string(index=False))
        print(f"\n95% 상한 > 명목 상한: {int(FA['상한>명목'].sum())} / {len(FA)}  (참고: > δ {int(FA['상한>δ'].sum())} / {len(FA)})")
        FA.to_csv(f'{E_DIR}/free_agg_cp_upper.csv', index=False)
""")

md("## 8. 실험 기록 · 종료")
code(r"""
rec = {'목적': '판정자 실험 E: 과신 판정자에서 실행 중 한도(JC1L)가 오답을 막는가', '일시': datetime.datetime.now().isoformat(timespec='seconds'),
       'GPU': GPU_NAME, '사전 기록': PRE, '재사용 기록': {BASE[k]['name']: BASE[k]['meta_path'] for k in BASE if not BASE[k]['rerun']},
       '다시 돌린 기록': [BASE[k]['name'] for k in RERUN], '상태': STATUS, '결과표': f'{E_DIR}/results_E.csv'}
json.dump(rec, open(f'{E_DIR}/record_E.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
print('저장:', f'{E_DIR}/record_E.json')
AUTO['reason'] = '완료' if ALL_DONE else '이번 세션 분량 끝 — 다시 "모두 실행"'
print('1분 뒤 런타임 종료' if AUTO_SHUTDOWN else '')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT, len(cells), "cells")
