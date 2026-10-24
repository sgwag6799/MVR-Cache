"""Rule splitter for the shared-prefix stress test (stage 2, 41K cache stream).

Same interface as MaxSimSplitter, no RL checkpoint. Cuts with the rules in
prefix_split.py (the same function the offline diagnostics use), so stage 1 and
stage 2 split every prompt identically:

  - punctuation tokens . , ! ? ; : (not [CLS], not the mark right before [SEP], not [SEP])
  - at most `max_cuts` cut points inside the original prompt (cap_scope="original");
    the shared prefix is recognised by its pool sentences only to apply this cap
  - no full-sentence row, no full-cosine mix (run the evaluation without
    --include-full-embedding and --mix-fullcos)

Weights (condition M3): a head file from prefix_offline.py (Linear(768 -> 1), softmax over
the segments of a prompt). Without a head file the policy uses uniform weights (M2).
"""
# [새 파일] 2단계(41K 스트림)용 규칙 분할기. 1단계 오프라인 진단과 같은 함수로 자른다.

import hashlib
from types import SimpleNamespace
from typing import Optional

import torch

from . import prefix_split as ps
from .MaxSimSplitter import MaxSimSplitter


def _tensor_key(t: torch.Tensor) -> bytes:
    # 조각 텐서 내용으로 만든 키: 분할할 때 계산한 가중치를 정책이 다시 찾을 때 쓴다
    return hashlib.blake2b(t.detach().float().cpu().contiguous().numpy().tobytes(), digest_size=16).digest()


class PrefixRuleSplitter(MaxSimSplitter):
    def __init__(
        self,
        device="cpu",
        embedding_model=None,
        *,
        max_cuts: Optional[int] = 32,
        cap_scope: str = "original",
        pool_path: Optional[str] = None,
        head_path: Optional[str] = None,
    ):
        # MaxSimSplitter.__init__은 RL 체크포인트를 불러오므로 일부러 부르지 않는다
        if embedding_model is None:
            raise ValueError("PrefixRuleSplitter needs the shared EmbeddingModel")
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.max_cuts = max_cuts
        self.cap_scope = cap_scope
        self.overlap_tokens = 0
        self.include_full_embedding = False
        self.embedding_model = embedding_model
        self.embedding_model.model.to(self.device)
        self.generator = SimpleNamespace(tokenizer=embedding_model.tokenizer, lm=embedding_model.model)
        self.pool = ps.load_pool(pool_path) if pool_path else None
        if cap_scope == "original" and max_cuts is not None and max_cuts >= 0 and self.pool is None:
            raise ValueError("cap_scope='original' needs --prefix-pool to find where the prefix ends")
        self.head = None
        if head_path:
            ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
            self.head = torch.nn.Linear(int(ckpt["dim"]), 1)
            self.head.load_state_dict({k.replace("lin.", ""): v for k, v in ckpt["state"].items()})
            self.head.eval()
        self._weight_cache: dict = {}

    def encode_text(self, text: str) -> dict:
        # 부모 결과에 원문과 글자 위치(offset)를 붙인다 (공통 문단 끝 위치를 찾는 데 필요)
        enc = super().encode_text(text)
        t = self.generator.tokenizer(text, truncation=True, max_length=ps.MAX_LEN, return_offsets_mapping=True)
        enc["text"] = text
        enc["offsets"] = [tuple(o) for o in t["offset_mapping"]]
        return enc

    def cut_points(self, enc: dict) -> list:
        length = int(enc["length"])
        ids = enc["input_ids"][:length].tolist()
        tokens = self.generator.tokenizer.convert_ids_to_tokens(ids)
        n_chars = ps.detect_prefix_chars(enc["text"], self.pool) if self.pool else 0
        pend = ps.prefix_token_end(enc["offsets"][:length], n_chars)
        return ps.cut_points(tokens, length, max_cuts=self.max_cuts, cap_scope=self.cap_scope, prefix_end=pend)

    def split_text_return_maxsim_tensor_from_encoded(self, enc: dict):
        length = int(enc["length"])
        sent, _full = self._segment_embeds_from_pointers(enc["token_emb"], length, self.cut_points(enc))
        out = sent.to(dtype=torch.float32)
        if self.head is not None:
            with torch.no_grad():
                logits = self.head(out.detach().float().cpu()).squeeze(-1)
            self._weight_cache[_tensor_key(out)] = torch.softmax(logits, dim=-1)
        return out

    def segment_weights(self, tensor: torch.Tensor):
        """Per-segment weights for M3 (None-free: uniform when there is no head)."""
        if self.head is None:
            return torch.ones(tensor.shape[0])
        w = self._weight_cache.get(_tensor_key(tensor))
        if w is None:  # 다른 경로로 만들어진 텐서면 다시 계산
            with torch.no_grad():
                w = torch.softmax(self.head(tensor.detach().float().cpu()).squeeze(-1), dim=-1)
        return w

    def split_pair_return_maxsim_tensors_from_encoded(self, enc_a: dict, enc_b: dict) -> tuple:
        return (self.split_text_return_maxsim_tensor_from_encoded(enc_a),
                self.split_text_return_maxsim_tensor_from_encoded(enc_b))

    def split_pair_return_maxsim_tensors(self, text_a: str, text_b: str):
        return self.split_pair_return_maxsim_tensors_from_encoded(self.encode_text(text_a), self.encode_text(text_b))
