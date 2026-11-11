"""[cls-scale-decomp] 평가 스크립트를 그대로 실행하되, json이 numpy 숫자를 저장하게 한다.

사용: python ../scripts/decomp/json_safe_run.py benchmarks/eval_sembenchmark_verified.py <인자...>
바닐라 eval은 결과 JSON을 쓴 뒤 benchmark 형식 파일(observations에 numpy float32)을 쓰다가 TypeError로 끝난다.
팀 RunPod의 run_eval.py와 같은 처리다. 판정·결과에는 영향이 없다.
"""
import json
import os
import runpy
import sys

import numpy as np

_default = json.JSONEncoder.default


def _numpy_default(self, o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return _default(self, o)


json.JSONEncoder.default = _numpy_default
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(os.path.abspath(script))  # python <script> 로 실행할 때와 같은 sys.path
runpy.run_path(script, run_name="__main__")
