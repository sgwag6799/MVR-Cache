"""판정자 실험 E: 노트북(notebooks/mvr_colab_judge_cap_E.ipynb)의 코드 셀을 SSH 용 스크립트로 바꾼다.

바뀌는 것 (나머지는 노트북과 같은 코드):
  - 설정: 환경변수로 받는다 (P_LOW_OVERRIDE · RUN_OVERCONF · MD_OVERRIDE · DATA_OVERRIDE · BASE_META_U/JMED/JC1L)
  - Drive: 노트북의 drive.mount 대신, 맥 터미널에서 `colab drivemount -s <세션>`으로 붙인 Drive 를 찾는다
  - 자동 종료: 없다 (끝나면 맥 터미널에서 `colab stop -s <세션>`)
사용: python scripts/judge_cap/gen_run_E.py notebooks/mvr_colab_judge_cap_E.ipynb scripts/judge_cap/run_E.py
"""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:80])
    return s.replace(a, b)


c0 = codes[0]
i, j = c0.index("REQUIRE_L4 = True"), c0.index("import subprocess")
c0 = c0[:i] + '''import os
_env = os.environ.get
REQUIRE_L4 = _env('ALLOW_GPU', '0') != '1'      # 기존 기록(B · C · D · E 기준선)이 L4 다
MD_OVERRIDE = _env('MD_OVERRIDE', '')           # 자동 탐색이 실패할 때만 작업 폴더 경로
DATA_OVERRIDE = _env('DATA_OVERRIDE', '')       # test 파일을 못 찾을 때만 Drive 경로
BASE_META_OVERRIDE = {'U': _env('BASE_META_U', ''), 'Jmed': _env('BASE_META_JMED', ''), 'JC1L': _env('BASE_META_JC1L', '')}
P_LOW_OVERRIDE = float(_env('P_LOW_OVERRIDE')) if _env('P_LOW_OVERRIDE') else None   # 비우면 사전 점검 규칙
RUN_OVERCONF = _env('RUN_OVERCONF', '0') == '1'  # 선택 조건 Jover · JoverC35 (팀원 확인 후, 사전 기록 전에만)
RUN_FREE_AGG = _env('RUN_FREE_AGG', '1') == '1'
AUTO_SHUTDOWN = False                            # SSH: 끝나면 맥 터미널에서 colab stop -s <세션>
MAX_HOURS = float(_env('MAX_HOURS', '23.5'))
BRANCH = 'judge-cap-overconfident'

''' + c0[j:]
a = c0.index("from google.colab import drive")
b = c0.index("NEED = (")
c0 = c0[:a] + '''# SSH: Drive 는 맥 터미널에서 `colab drivemount -s <세션>` 으로 붙인다 (노트북의 drive.mount 는 SSH 에서 동작하지 않는다)
if not glob.glob('/content/drive/*'):
    raise SystemExit('Drive 가 안 붙어 있다 → 맥 터미널에서 `colab drivemount -s l4` 를 실행하고 인증한 뒤 다시 실행한다')
''' + c0[b:]
c0 = rep(c0, "    roots = ['/content/drive/MyDrive'] + sorted(glob.glob('/content/drive/Shareddrives/*'))",
         "    # 'My Drive' 는 'MyDrive' 와 같은 폴더라 둘 다 있으면 MyDrive 만 쓴다\n"
         "    roots = (glob.glob('/content/drive/MyDrive') or glob.glob('/content/drive/My Drive')) + sorted(\n"
         "        {*glob.glob('/content/drive/Shareddrives/*'), *glob.glob('/content/drive/Shared drives/*')})")
c1 = "_T0 = time.time()\nAUTO = {'reason': None}\nprint(f'시작 {datetime.datetime.now().isoformat(timespec=\"seconds\")} · 최대 {MAX_HOURS}시간 (자동 종료 없음)')\n"
last = rep(codes[-1], "print('1분 뒤 런타임 종료' if AUTO_SHUTDOWN else '')",
           "print('끝. 런타임은 맥 터미널에서 `colab stop -s <세션>` 으로 끈다 (다른 실험이 같은 세션에서 돌고 있으면 그 실험이 끝난 뒤)')")
body = [c0, c1] + codes[2:-1] + [last]
names = ["§0 설정 · GPU · Drive", "§0-1 시작", "§1 재사용 기록 · GPU", "§2 저장소 · 패치", "§2 환경", "§3 데이터",
         "§4 사전 기록", "§5 본 실행", "§6 결과 · 판정 기준", "§7 무료 집계", "§8 기록"]
assert len(body) == len(names), (len(body), len(names))
out = ['"""판정자 실험 E (SSH 판). gen_run_E.py 가 노트북에서 만든 파일이다. 직접 고치지 말고 노트북 생성기를 고친다.\n\n'
       '사용 (Colab VM, repo 루트, tmux 안): python3 -u scripts/judge_cap/run_E.py 2>&1 | tee -a /content/judge_e_run.log\n"""\n']
for nm, c in zip(names, body):
    out.append(f"\n# ===== {nm} =====\nprint('\\n===== {nm} =====', flush=True)\n{c}\n")
open(sys.argv[2], "w", encoding="utf-8").write("".join(out))
print("wrote", sys.argv[2])
