"""
run_eval.py
평가 스크립트를 실행하기 전에 json.dump가 numpy 숫자(float32 등)도 저장할 수 있게 고쳐준다.
(코랩에서 겪었던 'Object of type float32 is not JSON serializable' 오류 방지)

사용: python run_eval.py benchmarks.eval_sembenchmark_verified_splitter --인자들...
      (mvr-cache 폴더에서 실행)
      환경변수 DECISION_LOG=<파일.jsonl> 을 주면 요청마다 판정 기록을 남긴다 (평가 로직은 변경 없음)
"""
import json
import runpy
import sys

import numpy as np

_orig_dump, _orig_dumps = json.dump, json.dumps


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _dump(obj, fp, **kw):
    # 호출 쪽이 자체 변환기(cls)나 default를 지정했으면 건드리지 않음 (예: matplotlib 폰트 캐시)
    if kw.get("cls") is None and "default" not in kw:
        kw["default"] = _default
    return _orig_dump(obj, fp, **kw)


def _dumps(obj, **kw):
    if kw.get("cls") is None and "default" not in kw:
        kw["default"] = _default
    return _orig_dumps(obj, **kw)


json.dump, json.dumps = _dump, _dumps

def _install_decision_logger(path):
    """
    판정 기록(평가 로직은 그대로, 기록만 추가).
    요청마다 한 줄: 후보 유무, 점수 s, 그 항목의 기록 수(n_obs), 판정(action), 이유(reason),
    히트 여부, 1등 후보가 실제로 정답이었는지(nn_correct)
    reason: cold(기록 6개 미만이라 무조건 LLM) / explore(확신 부족) / exploit(재사용) / no_candidate
    """
    import os

    from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
    from vcache.main import VCache
    from vcache.vcache_policy.strategies.verified import _Action, _Algorithm

    try:
        from benchmarks.common.comparison import answers_have_same_meaning_static as _same
    except Exception:  # noqa: BLE001
        def _same(a, b):
            return str(a).strip() == str(b).strip()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fh = open(path, "w", encoding="utf-8")
    state = {"i": 0, "label": "", "last": None}

    orig_set = BenchmarkInferenceEngine.set_next_response

    def set_next_response(self, response, *a, **kw):
        state["label"] = response
        return orig_set(self, response, *a, **kw)

    orig_select = _Algorithm.select_action

    def select_action(self, similarity_score, metadata, *a, **kw):
        n_obs = len(getattr(metadata, "observations", []) or [])
        action = orig_select(self, similarity_score, metadata, *a, **kw)
        if n_obs < int(getattr(self, "min_observations", 6)):
            reason = "cold"
        elif action == _Action.EXPLORE:
            reason = "explore"
        else:
            reason = "exploit"
        state["last"] = {"s": float(similarity_score), "n_obs": n_obs,
                         "action": getattr(action, "name", str(action)), "reason": reason}
        return action

    orig_infer = VCache.infer_with_cache_info

    def infer_with_cache_info(self, *a, **kw):
        state["last"] = None
        out = orig_infer(self, *a, **kw)
        is_hit, _resp, _resp_meta, nn_meta = out
        id_set = kw.get("id_set", -1)
        nn_correct = None
        if nn_meta is not None:
            try:
                if id_set is not None and int(id_set) != -1:
                    nn_correct = int(getattr(nn_meta, "id_set", -999999)) == int(id_set)
                else:
                    nn_correct = bool(_same(state["label"], getattr(nn_meta, "response", "")))
            except Exception:  # noqa: BLE001
                nn_correct = None
        rec = {"i": state["i"], "hit": bool(is_hit), "has_nn": nn_meta is not None,
               "nn_correct": nn_correct}
        rec.update(state["last"] or {"reason": "no_candidate"})
        if nn_meta is not None:
            policy_action = getattr(nn_meta, "_consensus_action", None)
            policy_reason = getattr(nn_meta, "_consensus_reason", None)
            raw_score = getattr(nn_meta, "_consensus_raw_score", None)
            agreement = getattr(nn_meta, "_consensus_agreement_excl", None)
            candidate_count = getattr(nn_meta, "_consensus_candidate_count", None)
            probability = getattr(nn_meta, "_consensus_probability", None)
            decision_score = getattr(nn_meta, "_consensus_decision_score", None)
            if policy_action is not None:
                rec["action"] = policy_action
            if policy_reason is not None:
                rec["policy_reason"] = policy_reason
            if raw_score is not None:
                rec["decision_score"] = rec.get("s")
                rec["s"] = float(raw_score)
            if decision_score is not None:
                rec["decision_score"] = float(decision_score)
            if agreement is not None:
                rec["a_excl"] = float(agreement)
            if candidate_count is not None:
                rec["candidate_count"] = int(candidate_count)
            if probability is not None:
                rec["p_correct"] = float(probability)
        fh.write(_orig_dumps(rec) + "\n")
        state["i"] += 1
        if state["i"] % 1000 == 0:
            fh.flush()
        return out

    BenchmarkInferenceEngine.set_next_response = set_next_response
    _Algorithm.select_action = select_action
    VCache.infer_with_cache_info = infer_with_cache_info
    print(f"[run_eval] 판정 기록 켜짐: {path}")
    return fh


if __name__ == "__main__":
    import os

    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    _fh = None
    if os.environ.get("DECISION_LOG"):
        _fh = _install_decision_logger(os.environ["DECISION_LOG"])
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        if _fh is not None:
            _fh.close()
