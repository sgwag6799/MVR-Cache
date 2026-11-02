"""Helpers for top-K answer agreement and calibrated correctness scores."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

import numpy as np


_SHORT_ANSWER_MAX_CHARS = 80
_SHORT_PUNCTUATION = re.compile(r"""[.,\"'\[\]]""")


def _normalize_short_answer(answer: str) -> str:
    return _SHORT_PUNCTUATION.sub("", str(answer)).strip().lower()


def _answer_agreement_flags(
    best_answer: str,
    candidate_answers: Sequence[str],
    *,
    embed_texts: Callable[[list[str]], np.ndarray],
    embedding_threshold: float = 0.85,
) -> list[bool]:
    answers = [str(answer or "") for answer in candidate_answers]
    if not answers:
        return []

    best = str(best_answer or "")
    long_pairs = [
        answer
        for answer in answers
        if len(best) > _SHORT_ANSWER_MAX_CHARS
        or len(answer) > _SHORT_ANSWER_MAX_CHARS
    ]
    embeddings: dict[str, np.ndarray] = {}
    if long_pairs:
        texts = list(dict.fromkeys([best, *long_pairs]))
        vectors = np.asarray(embed_texts(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("embed_texts must return one vector per input text")
        embeddings = dict(zip(texts, vectors, strict=True))

    best_norm = _normalize_short_answer(best)
    flags = []
    for answer in answers:
        if (
            len(best) <= _SHORT_ANSWER_MAX_CHARS
            and len(answer) <= _SHORT_ANSWER_MAX_CHARS
        ):
            flags.append(_normalize_short_answer(answer) == best_norm)
            continue

        best_vector = embeddings[best]
        answer_vector = embeddings[answer]
        denominator = float(
            np.linalg.norm(best_vector) * np.linalg.norm(answer_vector)
        )
        cosine = (
            float(np.dot(best_vector, answer_vector)) / denominator
            if denominator > 1e-12
            else 0.0
        )
        flags.append(cosine >= embedding_threshold)

    return flags


def answer_agreement_ratio(
    best_answer: str,
    candidate_answers: Sequence[str],
    *,
    embed_texts: Callable[[list[str]], np.ndarray],
    embedding_threshold: float = 0.85,
) -> float:
    """Return the share of candidate answers agreeing with the best candidate."""
    flags = _answer_agreement_flags(
        best_answer,
        candidate_answers,
        embed_texts=embed_texts,
        embedding_threshold=embedding_threshold,
    )
    return float(sum(flags) / len(flags)) if flags else 0.0


def top1_excluded_agreement(
    best_answer: str,
    candidate_answers: Sequence[str],
    *,
    embed_texts: Callable[[list[str]], np.ndarray],
    embedding_threshold: float = 0.85,
) -> tuple[int, float | None]:
    """Return candidate count and agreement among peers, excluding top1 itself."""
    flags = _answer_agreement_flags(
        best_answer,
        candidate_answers,
        embed_texts=embed_texts,
        embedding_threshold=embedding_threshold,
    )
    if len(flags) < 2:
        return len(flags), None
    return len(flags), float(max(0, sum(flags) - 1) / (len(flags) - 1))


def predict_correct_probability(
    similarity: float,
    agreement: float,
    candidate_count: int,
    model: dict,
) -> float:
    """Evaluate the saved logistic model over its named features."""
    coefficients = model["coefficients"]
    feature_names = model.get(
        "features", ["similarity", "agreement", "candidate_count"]
    )
    if len(coefficients) != len(feature_names):
        raise ValueError("model coefficient count does not match its features")
    feature_values = {
        "similarity": float(similarity),
        "agreement": float(agreement),
        "candidate_count": float(candidate_count),
    }
    means = model.get("feature_means", [0.0] * len(feature_names))
    scales = model.get("feature_scales", [1.0] * len(feature_names))
    standardized = [
        (feature_values[name] - float(mean)) / float(scale)
        for name, mean, scale in zip(
            feature_names, means, scales, strict=True
        )
    ]
    logit = float(model["intercept"]) + sum(
        float(coefficient) * value
        for coefficient, value in zip(coefficients, standardized, strict=True)
    )
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-min(logit, 700.0)))
    exp_logit = math.exp(max(logit, -700.0))
    return exp_logit / (1.0 + exp_logit)