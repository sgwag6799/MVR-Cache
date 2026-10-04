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
# [새 파일] 구두점 규칙 분할기 + 조각별 가중치 (Step 2, 3a, 3b, 5에서 사용)
#
# 전체 흐름:
#   1. 문장을 BGE로 한 번 임베딩해서 토큰마다 벡터를 얻는다.
#   2. 구두점(쉼표, 마침표 등)이 있는 곳마다 전부 자른다. (max_segments=0이면 자르지 않음 → 대조군)
#   3. 잘린 구간마다 토큰 벡터를 평균 내서 "조각 벡터"를 만든다. (원본 MaxSimSplitter 함수 재사용)
#   4. 조각마다 가중치를 계산해 둔다. vCache 정책이 MaxSim을 계산할 때 이 가중치로 가중 평균을 낸다.
#
# 원본 RL 분할기(MaxSimSplitter)를 상속하고 함수 이름·입출력을 똑같이 맞췄기 때문에,
# 평가 스크립트에서 분할기만 바꿔 끼우면 점수 계산·검색·캐시 정책은 그대로 둔 채 "자르는 위치"만 달라진다.

import hashlib  # 텐서 내용으로 고유 키(해시)를 만들 때 사용
from types import SimpleNamespace  # 속성만 담는 간단한 객체를 만들 때 사용

import torch
import torch.nn.functional as F  # 코사인 유사도, softplus 등 함수 모음

from .embedding_model import EmbeddingModel  # BGE 임베딩 모델 래퍼
from .MaxSimSplitter import MaxSimSplitter  # 원본 RL 분할기 (조각 벡터 만드는 함수를 물려받기 위해 상속)

# Same split characters as AdaptedPointerNetworkPolicy._init_punctuation_ids.
# 자를 수 있는 구두점 목록. RL 분할기가 고를 수 있는 후보 문자와 똑같이 맞췄다 (영문 + 전각 문자).
PUNCT_CHARS = [",", ".", "!", "?", ":", ";", "，", "。", "！", "？", "：", "；"]
# 지원하는 가중치 방식 5가지
WEIGHTINGS = ("uniform", "length", "idf", "centroid", "mlp")
# 가중치 하한. 가중치가 0이 되면 가중 평균의 분모가 0이 될 수 있어서 최소 0.001로 막는다.
MIN_WEIGHT = 1e-3


def punctuation_positions(input_ids: torch.Tensor, length: int, punct_ids: set) -> list:
    """Punctuation token positions, excluding CLS and a trailing mark right before [SEP]."""
    # 토큰 번호 목록에서 구두점 토큰이 있는 위치(인덱스)를 찾아 반환한다.
    # 범위 1 ~ length-3:
    #   - 0번은 [CLS] 토큰이라 제외
    #   - 맨 끝 [SEP] 바로 앞의 구두점(문장 끝 마침표 등)은 잘라도 의미 없는 빈 조각이 생기므로 제외
    ids = input_ids[:length].tolist()  # 텐서 → 파이썬 리스트 (패딩 제외, 실제 길이만큼)
    return [p for p in range(1, length - 2) if ids[p] in punct_ids]


def split_points(positions: list, max_segments) -> list:
    """Cut points for the rule splitter: every punctuation mark, or none when max_segments == 0."""
    # [수정] 규칙 분할기는 구두점이 있는 곳마다 전부 자른다 (조각 수 제한 없음).
    #        예전에는 앞에서부터 4개만 잘라서 4번째 구두점 이후가 마지막 조각 하나에 몰렸다.
    #        max_segments == 0 은 "분할하지 않음"(대조군, --splitter-max-segments 0)으로만 쓴다.
    if max_segments == 0:
        return []
    return list(positions)


def segment_spans(length: int, pointers: list) -> list:
    """[start, end) token spans produced by MaxSimSplitter._segment_embeds_from_pointers (no overlap)."""
    # 자르는 위치(pointers)로부터 각 조각의 토큰 범위 [시작, 끝)를 계산한다.
    # MaxSimSplitter가 조각 벡터를 만들 때 쓰는 범위와 똑같이 맞춘 것. (length, idf 가중치 계산에 필요)
    # 예: pointers=[3, 7], length=12 → [(1,4), (4,8), (8,12)]  (구두점은 앞 조각에 포함)
    spans, prev = [], 0
    for p in pointers:
        start = (prev + 1) if prev > 0 else 1  # 첫 조각은 1번([CLS] 다음)부터, 그 뒤는 이전 자른 위치 다음부터
        if p + 1 > start:  # 길이가 0인 조각은 만들지 않는다
            spans.append((start, p + 1))  # 구두점 토큰(p)까지 포함
        prev = p
    tail_start = (prev + 1) if prev > 0 else 1  # 마지막 자른 위치 다음부터 끝까지가 마지막 조각
    if tail_start < length:
        spans.append((tail_start, length))  # 주의: 끝(length)에 [SEP] 토큰도 포함된다
    return spans or [(1, length)]  # 자를 곳이 없으면 문장 전체가 한 조각


class SegmentWeightMLP(torch.nn.Module):
    # mlp 가중치용 작은 신경망: 조각 벡터(768차원) → 가중치 숫자 1개
    def __init__(self, dim: int = 768, hidden: int = 128):
        super().__init__()
        # 768 → 128 → ReLU → 1 구조
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # softplus로 항상 양수를 만들고, 하한(MIN_WEIGHT)을 더해 0이 되지 않게 한다.
        # squeeze(-1): [조각 수, 1] → [조각 수] 모양으로 바꿈
        return F.softplus(self.net(x)).squeeze(-1) + MIN_WEIGHT


def _tensor_key(t: torch.Tensor) -> bytes:
    # 텐서의 "내용"으로 16바이트 해시 키를 만든다.
    # 분할할 때 계산한 가중치를 저장해 두었다가, 나중에 같은 텐서가 들어오면 이 키로 찾아 쓰기 위함.
    # (vCache 정책은 조각 텐서만 넘겨받고 원래 문장 정보는 모르기 때문에, 내용으로 매칭한다)
    return hashlib.blake2b(t.detach().float().cpu().contiguous().numpy().tobytes(), digest_size=16).digest()


class SegmentRowWeighter:
    """Per-row weights computed from segment embeddings alone (no token spans needed).

    Attach to any splitter (including the RL MaxSimSplitter) as `segment_weights` to make
    VerifiedSplitterDecisionPolicy score with a weighted MaxSim mean:

        splitter.segment_weights = SegmentRowWeighter("mlp", stats, include_full_embedding=True)

    Only the weightings that need no token information are supported ('centroid', 'mlp');
    'length' and 'idf' require the splitter itself to know the segment spans.
    """
    # RL 분할기(Step 5)에 가중치를 붙이기 위한 클래스.
    # RL 분할기는 조각의 토큰 범위를 밖으로 알려주지 않으므로, 조각 벡터만으로 계산 가능한
    # centroid와 mlp만 지원한다. (length, idf는 토큰 정보가 필요해서 불가)

    SUPPORTED = ("centroid", "mlp")

    def __init__(self, weighting: str, weight_stats: dict, *, include_full_embedding: bool = True):
        if weighting not in self.SUPPORTED:  # 지원하지 않는 방식이면 에러
            raise ValueError(f"SegmentRowWeighter supports {self.SUPPORTED}, got {weighting!r}")
        self.weighting = weighting
        self.include_full_embedding = bool(include_full_embedding)  # 마지막 행이 전체 문장 벡터인지
        self._centroid = weight_stats.get("centroid")  # 통계 파일의 중심 벡터 (centroid용)
        self._mlp = None
        if weighting == "centroid" and self._centroid is None:
            raise ValueError("weighting='centroid' needs a 'centroid' vector in weight_stats.")
        if weighting == "mlp":
            if "mlp_state" not in weight_stats:
                raise ValueError("weighting='mlp' needs 'mlp_state' in weight_stats.")
            # 학습해 둔 MLP를 같은 구조로 만들고, 저장된 파라미터를 불러온다
            self._mlp = SegmentWeightMLP(hidden=int(weight_stats.get("mlp_hidden", 128)))
            self._mlp.load_state_dict(weight_stats["mlp_state"])
            self._mlp.eval()  # 추론 모드 (학습하지 않음)

    def _rows(self, rows: torch.Tensor) -> torch.Tensor:
        # 조각 벡터들 → 조각별 가중치
        if self.weighting == "centroid":
            # 1 − (조각과 코퍼스 중심의 코사인): 흔한(중심에 가까운) 조각일수록 작고, 특이한 조각일수록 크다
            cos = F.cosine_similarity(rows, self._centroid.float().unsqueeze(0), dim=-1)
            return (1.0 - cos).clamp_min(MIN_WEIGHT)
        with torch.inference_mode():  # 기울기 계산 없이 빠르게 실행
            return self._mlp(rows).clamp_min(MIN_WEIGHT)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # splitter.segment_weights(텐서) 처럼 함수로 호출된다
        rows = tensor.detach().float().cpu()
        if self.include_full_embedding and rows.shape[0] > 1:
            # 마지막 행은 전체 문장 벡터 → 조각들로 가중치를 구한 뒤, 전체 행에는 그 평균을 준다
            seg_w = self._rows(rows[:-1])
            return torch.cat([seg_w, seg_w.mean().unsqueeze(0)])
        return self._rows(rows)


class RulePunctuationSplitter(MaxSimSplitter):
    # 구두점 규칙 분할기 본체
    def __init__(
        self,
        device="cpu",
        embedding_model=None,
        *,
        max_segments=None,  # None = 구두점마다 전부 자름(기본), 0 = 자르지 않음(대조군)
        overlap_tokens: int = 0,  # 조각 경계에서 겹칠 토큰 수 (가중치를 쓰려면 0이어야 함)
        include_full_embedding: bool = False,  # 조각 뒤에 전체 문장 벡터를 한 행 더 붙일지
        weighting: str = "uniform",  # 가중치 방식
        weight_stats: dict = None,  # build_segment_weight_stats.py가 만든 통계 (idf, centroid, mlp)
    ):
        # Deliberately skip MaxSimSplitter.__init__: no checkpoint, env or policy is needed.
        # 부모(MaxSimSplitter)의 생성자는 RL 체크포인트를 불러오므로 일부러 호출하지 않고,
        # 필요한 속성만 직접 설정한다.
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        if max_segments not in (None, 0):
            raise ValueError("RulePunctuationSplitter cuts at every punctuation mark; max_segments must be None or 0 (no split).")
        self.max_segments = max_segments
        self.overlap_tokens = max(0, int(overlap_tokens))
        self.include_full_embedding = bool(include_full_embedding)
        # 임베딩 모델: 넘겨받은 게 있으면 공유(메모리 절약), 없으면 새로 만든다
        self.embedding_model = embedding_model if embedding_model is not None else EmbeddingModel()
        self.embedding_model.model.to(self.device)
        # encode_text() only needs generator.tokenizer and generator.lm.
        # 부모의 encode_text() 함수는 self.generator.tokenizer / self.generator.lm만 쓰므로,
        # 그 두 속성만 가진 가짜 객체를 만들어 넣는다.
        self.generator = SimpleNamespace(
            tokenizer=self.embedding_model.tokenizer, lm=self.embedding_model.model
        )
        tok = self.embedding_model.tokenizer
        # 구두점 문자 → 토크나이저의 토큰 번호로 변환 (어휘에 없는 문자 = unk 는 제외)
        self._punct_ids = {i for i in tok.convert_tokens_to_ids(PUNCT_CHARS) if i != tok.unk_token_id}

        # ---- 가중치 설정 검사 ----
        if weighting not in WEIGHTINGS:
            raise ValueError(f"Unknown weighting {weighting!r}; choose from {WEIGHTINGS}")
        if self.overlap_tokens and weighting != "uniform":
            # 조각이 겹치면 length/idf 계산 범위가 맞지 않으므로 금지
            raise ValueError("Segment weighting assumes non-overlapping segments (overlap_tokens=0).")
        self.weighting = weighting
        stats = weight_stats or {}  # 통계가 없으면 빈 딕셔너리
        self._idf = stats.get("idf")  # [어휘 크기] 토큰별 IDF 값 표
        self._centroid = stats.get("centroid")  # [768] 코퍼스 조각 벡터들의 평균
        self._mlp = None
        # 선택한 방식에 필요한 통계가 없으면 에러
        if weighting == "idf" and self._idf is None:
            raise ValueError("weighting='idf' needs an 'idf' table in weight_stats.")
        if weighting == "centroid" and self._centroid is None:
            raise ValueError("weighting='centroid' needs a 'centroid' vector in weight_stats.")
        if weighting == "mlp":
            if "mlp_state" not in stats:
                raise ValueError("weighting='mlp' needs 'mlp_state' in weight_stats.")
            # 학습된 MLP 불러오기
            self._mlp = SegmentWeightMLP(hidden=int(stats.get("mlp_hidden", 128)))
            self._mlp.load_state_dict(stats["mlp_state"])
            self._mlp.eval()
        # 텐서 내용 해시 → 가중치. 분할할 때 계산한 가중치를 저장해 두는 곳
        self._weight_cache: dict = {}
        # 저장된 가중치를 못 찾아 다시 계산한 횟수 (디버깅용)
        self.weight_cache_misses = 0

    def _weights_for_segments(self, sent: torch.Tensor, spans: list, input_ids: torch.Tensor) -> torch.Tensor:
        # 조각별 가중치 계산. sent = 조각 벡터들, spans = 조각별 토큰 범위, input_ids = 토큰 번호
        if self.weighting == "uniform":
            return torch.ones(sent.shape[0])  # 모두 1
        if self.weighting == "length":
            return torch.tensor([float(e - s) for s, e in spans])  # 조각의 토큰 수
        if self.weighting == "idf":
            ids = input_ids.detach().cpu()
            # 조각에 속한 토큰들의 IDF 값을 표에서 찾아 평균 → 흔한 단어 위주 조각은 작은 값
            return torch.stack([self._idf[ids[s:e]].float().mean() for s, e in spans])
        # centroid / mlp는 조각 벡터만으로 계산
        return self._weights_from_rows(sent.detach().float().cpu())

    def _weights_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        # 조각 벡터 → 가중치 (centroid, mlp 공통). SegmentRowWeighter._rows와 같은 계산
        if self.weighting == "centroid":
            cos = F.cosine_similarity(rows, self._centroid.float().unsqueeze(0), dim=-1)
            return (1.0 - cos).clamp_min(MIN_WEIGHT)
        with torch.inference_mode():
            return self._mlp(rows)

    def split_text_return_maxsim_tensor_from_encoded(self, enc: dict):
        # [핵심 함수] 임베딩된 문장(enc) 하나 → 조각 벡터 텐서. 가중치도 함께 계산해 저장해 둔다.
        # enc: encode_text()의 결과 (토큰 번호, 토큰 벡터, 길이 등)
        length = int(enc["length"])  # 실제 토큰 수
        # 구두점 위치를 모두 찾아 그 위치마다 자른다 (max_segments=0이면 자르지 않음)
        pointers = split_points(punctuation_positions(enc["input_ids"], length, self._punct_ids), self.max_segments)
        # 원본 MaxSimSplitter 함수로 자를 위치에서 조각 벡터(sent)와 전체 문장 벡터(full)를 만든다
        sent, full = self._segment_embeds_from_pointers(
            enc["token_emb"], length, pointers, overlap_tokens=self.overlap_tokens
        )
        # 조각별 가중치 계산 후 하한 적용
        weights = self._weights_for_segments(sent, segment_spans(length, pointers), enc["input_ids"])
        weights = weights.float().clamp_min(MIN_WEIGHT)
        if self.include_full_embedding:
            # 조각들 뒤에 전체 문장 벡터를 한 행 붙인다
            out = torch.cat([sent, full], dim=0).to(dtype=torch.float32)
            # 전체 행의 가중치 = 조각 가중치의 평균 → 가중치 방식이 바뀌어도 전체 행의 상대적 비중은 일정
            weights = torch.cat([weights, weights.mean().unsqueeze(0)])
        else:
            out = sent.to(dtype=torch.float32)
        # 결과 텐서의 내용 해시를 키로 가중치를 저장 → 나중에 segment_weights(out)으로 찾아 쓴다
        self._weight_cache[_tensor_key(out)] = weights
        return out

    def segment_weights(self, tensor: torch.Tensor) -> torch.Tensor:
        """Per-row weights for a tensor this splitter produced (looked up by content)."""
        # vCache 정책(_score_tensors)이 호출하는 함수: 조각 텐서 → 행별 가중치
        w = self._weight_cache.get(_tensor_key(tensor))  # 분할할 때 저장해 둔 가중치 찾기
        if w is not None:
            return w
        # 못 찾은 경우 (텐서가 다른 경로로 만들어졌거나 값이 미세하게 달라진 경우)
        self.weight_cache_misses += 1
        if self.weighting in ("centroid", "mlp"):
            # 이 두 방식은 텐서만으로 다시 계산할 수 있다
            rows = tensor.detach().float().cpu()
            if self.include_full_embedding and rows.shape[0] > 1:
                seg_w = self._weights_from_rows(rows[:-1]).clamp_min(MIN_WEIGHT)
                return torch.cat([seg_w, seg_w.mean().unsqueeze(0)])
            return self._weights_from_rows(rows).clamp_min(MIN_WEIGHT)
        # length/idf는 토큰 정보가 없어서 다시 계산할 수 없으므로 균등 가중치로 대체
        return torch.ones(tensor.shape[0])

    def split_pair_return_maxsim_tensors_from_encoded(self, enc_a: dict, enc_b: dict) -> tuple:
        # Rule segmentation is per text, so a pair is just two independent splits.
        # 두 문장을 각각 따로 분할해서 함께 반환 (RL 분할기와 같은 함수 이름을 맞추기 위해 존재)
        return (
            self.split_text_return_maxsim_tensor_from_encoded(enc_a),
            self.split_text_return_maxsim_tensor_from_encoded(enc_b),
        )

    def split_pair_return_maxsim_tensors(self, text_a: str, text_b: str):
        # 문자열 두 개를 받아 임베딩부터 한 뒤 위 함수 호출
        return self.split_pair_return_maxsim_tensors_from_encoded(
            self.encode_text(text_a), self.encode_text(text_b)
        )
