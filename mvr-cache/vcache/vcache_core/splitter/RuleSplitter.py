"""Checkpoint-free punctuation splitter with the same interface as MaxSimSplitter.

Used as the rule-based segmentation baseline: it cuts at punctuation tokens (the same
split characters the RL pointer policy may choose from), so swapping it in for
MaxSimSplitter changes only *where* prompts are split, not how they are scored.

Optional per-segment weighting (consumed by VerifiedSplitterDecisionPolicy via
`segment_weights`) turns the uniform MaxSim mean into a weighted mean:

  - uniform:  every segment weighs 1 (default, identical to the unweighted splitter)
  - length:   number of tokens in the segment
  - idf:      mean IDF of the segment's tokens (IDF table from --segment-weight-stats)
  - centroid: 1 - cos(segment, corpus centroid), i.e. how distinctive the segment is
  - mlp:      softplus(MLP(segment embedding)), trained offline with BCE

The full-embedding row (when included) gets the mean of the segment weights, so it keeps
the same relative influence it has under uniform weighting.
"""

import hashlib
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .embedding_model import EmbeddingModel
from .MaxSimSplitter import MaxSimSplitter

# Same split characters as AdaptedPointerNetworkPolicy._init_punctuation_ids.
PUNCT_CHARS = [",", ".", "!", "?", ":", ";", "，", "。", "！", "？", "：", "；"]
WEIGHTINGS = ("uniform", "length", "idf", "centroid", "mlp")
MIN_WEIGHT = 1e-3


def punctuation_positions(input_ids: torch.Tensor, length: int, punct_ids: set) -> list:
    """Punctuation token positions, excluding CLS and a trailing mark right before [SEP]."""
    ids = input_ids[:length].tolist()
    return [p for p in range(1, length - 2) if ids[p] in punct_ids]


def segment_spans(length: int, pointers: list) -> list:
    """[start, end) token spans produced by MaxSimSplitter._segment_embeds_from_pointers (no overlap)."""
    spans, prev = [], 0
    for p in pointers:
        start = (prev + 1) if prev > 0 else 1
        if p + 1 > start:
            spans.append((start, p + 1))
        prev = p
    tail_start = (prev + 1) if prev > 0 else 1
    if tail_start < length:
        spans.append((tail_start, length))
    return spans or [(1, length)]


class SegmentWeightMLP(torch.nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x)).squeeze(-1) + MIN_WEIGHT


def _tensor_key(t: torch.Tensor) -> bytes:
    return hashlib.blake2b(t.detach().float().cpu().contiguous().numpy().tobytes(), digest_size=16).digest()


class SegmentRowWeighter:
    """Per-row weights computed from segment embeddings alone (no token spans needed).

    Attach to any splitter (including the RL MaxSimSplitter) as `segment_weights` to make
    VerifiedSplitterDecisionPolicy score with a weighted MaxSim mean:

        splitter.segment_weights = SegmentRowWeighter("mlp", stats, include_full_embedding=True)

    Only the weightings that need no token information are supported ('centroid', 'mlp');
    'length' and 'idf' require the splitter itself to know the segment spans.
    """

    SUPPORTED = ("centroid", "mlp")

    def __init__(self, weighting: str, weight_stats: dict, *, include_full_embedding: bool = True):
        if weighting not in self.SUPPORTED:
            raise ValueError(f"SegmentRowWeighter supports {self.SUPPORTED}, got {weighting!r}")
        self.weighting = weighting
        self.include_full_embedding = bool(include_full_embedding)
        self._centroid = weight_stats.get("centroid")
        self._mlp = None
        if weighting == "centroid" and self._centroid is None:
            raise ValueError("weighting='centroid' needs a 'centroid' vector in weight_stats.")
        if weighting == "mlp":
            if "mlp_state" not in weight_stats:
                raise ValueError("weighting='mlp' needs 'mlp_state' in weight_stats.")
            self._mlp = SegmentWeightMLP(hidden=int(weight_stats.get("mlp_hidden", 128)))
            self._mlp.load_state_dict(weight_stats["mlp_state"])
            self._mlp.eval()

    def _rows(self, rows: torch.Tensor) -> torch.Tensor:
        if self.weighting == "centroid":
            cos = F.cosine_similarity(rows, self._centroid.float().unsqueeze(0), dim=-1)
            return (1.0 - cos).clamp_min(MIN_WEIGHT)
        with torch.inference_mode():
            return self._mlp(rows).clamp_min(MIN_WEIGHT)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        rows = tensor.detach().float().cpu()
        if self.include_full_embedding and rows.shape[0] > 1:
            seg_w = self._rows(rows[:-1])
            return torch.cat([seg_w, seg_w.mean().unsqueeze(0)])
        return self._rows(rows)


class RulePunctuationSplitter(MaxSimSplitter):
    def __init__(
        self,
        device="cpu",
        embedding_model=None,
        *,
        max_segments: int = 4,
        overlap_tokens: int = 0,
        include_full_embedding: bool = False,
        weighting: str = "uniform",
        weight_stats: dict = None,
    ):
        # Deliberately skip MaxSimSplitter.__init__: no checkpoint, env or policy is needed.
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.max_segments = int(max_segments)
        self.overlap_tokens = max(0, int(overlap_tokens))
        self.include_full_embedding = bool(include_full_embedding)
        self.embedding_model = embedding_model if embedding_model is not None else EmbeddingModel()
        self.embedding_model.model.to(self.device)
        # encode_text() only needs generator.tokenizer and generator.lm.
        self.generator = SimpleNamespace(
            tokenizer=self.embedding_model.tokenizer, lm=self.embedding_model.model
        )
        tok = self.embedding_model.tokenizer
        self._punct_ids = {i for i in tok.convert_tokens_to_ids(PUNCT_CHARS) if i != tok.unk_token_id}

        if weighting not in WEIGHTINGS:
            raise ValueError(f"Unknown weighting {weighting!r}; choose from {WEIGHTINGS}")
        if self.overlap_tokens and weighting != "uniform":
            raise ValueError("Segment weighting assumes non-overlapping segments (overlap_tokens=0).")
        self.weighting = weighting
        stats = weight_stats or {}
        self._idf = stats.get("idf")
        self._centroid = stats.get("centroid")
        self._mlp = None
        if weighting == "idf" and self._idf is None:
            raise ValueError("weighting='idf' needs an 'idf' table in weight_stats.")
        if weighting == "centroid" and self._centroid is None:
            raise ValueError("weighting='centroid' needs a 'centroid' vector in weight_stats.")
        if weighting == "mlp":
            if "mlp_state" not in stats:
                raise ValueError("weighting='mlp' needs 'mlp_state' in weight_stats.")
            self._mlp = SegmentWeightMLP(hidden=int(stats.get("mlp_hidden", 128)))
            self._mlp.load_state_dict(stats["mlp_state"])
            self._mlp.eval()
        self._weight_cache: dict = {}
        self.weight_cache_misses = 0

    def _weights_for_segments(self, sent: torch.Tensor, spans: list, input_ids: torch.Tensor) -> torch.Tensor:
        if self.weighting == "uniform":
            return torch.ones(sent.shape[0])
        if self.weighting == "length":
            return torch.tensor([float(e - s) for s, e in spans])
        if self.weighting == "idf":
            ids = input_ids.detach().cpu()
            return torch.stack([self._idf[ids[s:e]].float().mean() for s, e in spans])
        return self._weights_from_rows(sent.detach().float().cpu())

    def _weights_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if self.weighting == "centroid":
            cos = F.cosine_similarity(rows, self._centroid.float().unsqueeze(0), dim=-1)
            return (1.0 - cos).clamp_min(MIN_WEIGHT)
        with torch.inference_mode():
            return self._mlp(rows)

    def split_text_return_maxsim_tensor_from_encoded(self, enc: dict):
        length = int(enc["length"])
        pointers = punctuation_positions(enc["input_ids"], length, self._punct_ids)[: self.max_segments]
        sent, full = self._segment_embeds_from_pointers(
            enc["token_emb"], length, pointers, overlap_tokens=self.overlap_tokens
        )
        weights = self._weights_for_segments(sent, segment_spans(length, pointers), enc["input_ids"])
        weights = weights.float().clamp_min(MIN_WEIGHT)
        if self.include_full_embedding:
            out = torch.cat([sent, full], dim=0).to(dtype=torch.float32)
            weights = torch.cat([weights, weights.mean().unsqueeze(0)])
        else:
            out = sent.to(dtype=torch.float32)
        self._weight_cache[_tensor_key(out)] = weights
        return out

    def segment_weights(self, tensor: torch.Tensor) -> torch.Tensor:
        """Per-row weights for a tensor this splitter produced (looked up by content)."""
        w = self._weight_cache.get(_tensor_key(tensor))
        if w is not None:
            return w
        self.weight_cache_misses += 1
        if self.weighting in ("centroid", "mlp"):
            rows = tensor.detach().float().cpu()
            if self.include_full_embedding and rows.shape[0] > 1:
                seg_w = self._weights_from_rows(rows[:-1]).clamp_min(MIN_WEIGHT)
                return torch.cat([seg_w, seg_w.mean().unsqueeze(0)])
            return self._weights_from_rows(rows).clamp_min(MIN_WEIGHT)
        return torch.ones(tensor.shape[0])

    def split_pair_return_maxsim_tensors_from_encoded(self, enc_a: dict, enc_b: dict) -> tuple:
        # Rule segmentation is per text, so a pair is just two independent splits.
        return (
            self.split_text_return_maxsim_tensor_from_encoded(enc_a),
            self.split_text_return_maxsim_tensor_from_encoded(enc_b),
        )

    def split_pair_return_maxsim_tensors(self, text_a: str, text_b: str):
        return self.split_pair_return_maxsim_tensors_from_encoded(
            self.encode_text(text_a), self.encode_text(text_b)
        )
