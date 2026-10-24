from vcache.inference_engine.inference_engine import InferenceEngine
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.inference_engine.strategies.lang_chain import LangChainInferenceEngine
from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine
from vcache.inference_engine.strategies.silicon_flow import SiliconFlowInferenceEngine
# [수정: prefix-stress-test] vllm 없이도 import 되게 (평가는 BenchmarkInferenceEngine만 사용)
try:
    from vcache.inference_engine.strategies.vllm import VLLMInferenceEngine
except ImportError:
    VLLMInferenceEngine = None

__all__ = [
    "InferenceEngine",
    "LangChainInferenceEngine",
    "OpenAIInferenceEngine",
    "SiliconFlowInferenceEngine",
    "BenchmarkInferenceEngine",
    "VLLMInferenceEngine",
]
