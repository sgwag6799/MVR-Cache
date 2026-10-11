"""Splitting rules for the shared-prefix stress test (branch prefix-stress-test).

One place for every rule the experiment fixes in advance, so that data checks,
offline pair diagnostics and the cache evaluation cut prompts in exactly the same way:

  - cut points: punctuation tokens . , ! ? ; :  (positions 1 .. length-3, i.e. not [CLS],
    not the mark right before [SEP], not [SEP])
  - cap: at most `max_cuts` cut points; with cap_scope="original" the cap counts only cut
    points inside the original prompt, so the prefix never changes how the original part
    is split (cap_scope="all" counts from the very first token instead)
  - prefix rows: a segment belongs to the shared prefix when it ends inside the prefix
    characters; condition M4 gives those segments weight 0

Pure Python (no torch) so it can run anywhere the tokenizer runs.
"""
# [새 파일] 공통 문단 스트레스 테스트의 분할 규칙.
# 데이터 점검·오프라인 쌍 진단·캐시 평가가 모두 이 함수를 써서 같은 방식으로 자른다.

from __future__ import annotations

import json
import random
from typing import Optional

PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}  # 자르는 구두점 토큰 (영문 6개)
MAX_LEN = 512  # bge-base-en-v1.5 최대 입력 길이 ([CLS], [SEP] 포함)


# ----------------------------------------------------------------------------- 공통 문단


def load_pool(path: str) -> list:
    """Pool file -> list of slots, each a list of variant sentences."""
    with open(path, encoding="utf-8") as f:
        return [slot["variants"] for slot in json.load(f)["slots"]]


def prefix_choice(row_key: str, n_slots: int, n_variants: int, seed: int, mode: str) -> list:
    """Variant index per slot for one prompt.

    variant: drawn per prompt with a seed derived from (seed, row_key). All slots are drawn
             every time, so level L2 uses the same sentences as the first two of L8.
    fixed:   variant 0 in every slot for every prompt.
    """
    if mode == "fixed":
        return [0] * n_slots
    rng = random.Random(f"{seed}|{row_key}")  # 행마다 고정된 난수 → 단계가 달라도 같은 선택
    return [rng.randrange(n_variants) for _ in range(n_slots)]


def build_prefix(pool: list, choice: list, level: int) -> str:
    """The first `level` slots, joined with spaces (empty for level 0)."""
    return " ".join(pool[i][choice[i]] for i in range(level))


def join_prompt(prefix: str, prompt: str) -> tuple:
    """(full prompt, number of prefix characters). The original prompt is left untouched."""
    if not prefix:
        return prompt, 0
    return prefix + " " + prompt, len(prefix)


def detect_prefix_chars(text: str, pool: list) -> int:
    """Length of the leading run of pool sentences in `text` (0 if none).

    Used where only the final prompt text is available (the cache evaluation), to apply
    cap_scope="original". It recognises pool sentences only, never other text.
    """
    sentences = {s for slot in pool for s in slot}
    pos = 0
    while True:
        rest = text[pos:]
        hit = next((s for s in sentences if rest.startswith(s)), None)
        if hit is None:
            break
        pos += len(hit)
        if text[pos:pos + 1] == " ":
            nxt = text[pos + 1:]
            if any(nxt.startswith(s) for s in sentences):
                pos += 1
                continue
        break
    return pos


# ----------------------------------------------------------------------------- 자르기


def prefix_token_end(offsets: list, prefix_chars: int) -> int:
    """Index one past the last prefix token (1 when there is no prefix).

    offsets: tokenizer character offsets for the (truncated) sequence incl. [CLS]/[SEP],
    whose special tokens have (0, 0).
    """
    end = 1
    if prefix_chars <= 0:
        return end
    for i in range(1, len(offsets)):
        a, b = offsets[i]
        if b == 0 and a == 0:  # [SEP]
            break
        if b <= prefix_chars:
            end = i + 1
        else:
            break
    return end


def cut_points(tokens: list, length: int, *, max_cuts: Optional[int] = 32,
               cap_scope: str = "original", prefix_end: int = 1) -> list:
    """Token positions to cut at (inclusive segment ends).

    tokens: token strings of the sequence (incl. [CLS] and [SEP]), length: real length.
    max_cuts None or < 0 = no cap.
    """
    if length <= 3:
        return []
    cuts = [i for i in range(1, length - 2) if tokens[i] in PUNCT_TOKENS]
    if max_cuts is None or max_cuts < 0:
        return cuts
    if cap_scope == "all":
        return cuts[:max_cuts]
    if cap_scope != "original":
        raise ValueError(f"cap_scope must be 'original' or 'all', got {cap_scope!r}")
    # 공통 문단 안의 자르는 위치는 모두 두고, 원본 부분의 자르는 위치만 앞에서 max_cuts개
    head = [c for c in cuts if c < prefix_end]
    body = [c for c in cuts if c >= prefix_end]
    return head + body[:max_cuts]


def segment_spans(length: int, cuts: list) -> list:
    """[start, end) token spans, same as MaxSimSplitter._segment_embeds_from_pointers
    (no overlap; [CLS] excluded; the last span runs to `length` and includes [SEP])."""
    spans, prev = [], 0
    for p in sorted(set(cuts)):
        start = (prev + 1) if prev > 0 else 1
        if p + 1 > start:
            spans.append((start, p + 1))
        prev = p
    tail = (prev + 1) if prev > 0 else 1
    if tail < length:
        spans.append((tail, length))
    return spans or [(1, length)]


def prefix_segment_flags(spans: list, prefix_end: int) -> list:
    """True for segments that lie entirely in the shared prefix."""
    return [e <= prefix_end for _, e in spans]


# ----------------------------------------------------------------------------- 정답 비교


def normalize_answer(text) -> str:
    """Same normalisation as StringComparisonSimilarityEvaluator (cache decisions)."""
    return (
        str(text).strip().rstrip(".").lower()
        .replace('"', "").replace("'", "").replace("[", "").replace("]", "")
    )
