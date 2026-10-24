"""Load prefix_split.py by file path.

Importing it as `vcache.vcache_core.splitter.prefix_split` would run the package
__init__ files, which pull in the RL splitter (rl4co, tensordict). The data and
offline scripts only need the pure-Python rules, so they load the file directly.
"""
# [새 파일] prefix_split.py를 파일 경로로 불러온다 (vcache 패키지 전체 import 회피)

import importlib.util
import os

_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "vcache", "vcache_core", "splitter", "prefix_split.py",
)
_spec = importlib.util.spec_from_file_location("prefix_split", _PATH)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

POOL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_pool.json")


def load_tokenizer(name: str = "BAAI/bge-base-en-v1.5"):
    """Fast tokenizer with offsets (tokenizers library; same files as transformers)."""
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(name)


def encode(tok, text: str, max_len: int = 512):
    """(ids, tokens, offsets, full_length) with [CLS]/[SEP], truncated to max_len like BGE."""
    e = tok.encode(text)
    full = len(e.ids)
    ids, tokens, offsets = list(e.ids), list(e.tokens), [tuple(o) for o in e.offsets]
    if full > max_len:  # 앞 max_len-1개 + [SEP] (transformers truncation과 같은 결과)
        ids = ids[: max_len - 1] + [ids[-1]]
        tokens = tokens[: max_len - 1] + [tokens[-1]]
        offsets = offsets[: max_len - 1] + [(0, 0)]
    return ids, tokens, offsets, full
