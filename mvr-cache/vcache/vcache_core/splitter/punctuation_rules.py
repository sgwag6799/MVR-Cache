"""Where the rule splitter cuts a prompt: every punctuation mark, minus the ones that do
not end a phrase.

Decisions are made on the original text through the tokenizer's character offsets, so
"3.5" (no spaces) and "3. 5" are told apart even though both tokenize to 3 / . / 5.

  1. punctuation inside numbers           3.5, 1,000, $1,299.99, 10:30, v2.1.3, 2026.10.04
  2. abbreviations and titles             e.g., i.e., etc., vs., Mr., Dr., Ph.D., U.S., a.m., No. 5
  3. URLs, e-mail addresses, file names   www.amazon.com, https://..., abc@gmail.com, data.csv
  4. runs of punctuation                  ..., !!!, ?!, ., ;; and ". . ."  -> cut once, after the run
  5. emoticons                            :) ;) :-( :D ^^;
  6. cuts that leave a segment with no word (trailing ".", a list number "1.", a
     punctuation-only piece) are dropped, so empty segments never appear
"""
# [새 파일] 규칙 분할기의 "어디서 자를지" 규칙.
# 기본은 구두점마다 자르되, 문장을 끝내는 구두점이 아닌 경우(숫자, 약어, URL, 이모티콘 등)는 자르지 않는다.
# 토크나이저는 "3.5"와 "3. 5"를 똑같이 3 / . / 5 로 쪼개므로, 원문 글자 위치(offset)를 보고 판단한다.

from __future__ import annotations

import re

# 2. 약어 (소문자, 끝의 마침표 제외). 이 단어 안/끝의 마침표는 자르지 않는다
ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "esp",
    "mr", "mrs", "ms", "dr", "prof", "ph.d", "jr", "sr", "st", "mt", "rev", "gen", "capt", "sgt",
    "u.s", "u.k", "u.s.a", "a.m", "p.m", "d.c",
    "inc", "ltd", "co", "corp", "dept", "est", "vol", "fig", "ave", "blvd",
    "ft", "oz", "lb", "lbs", "hr", "hrs", "min", "mins", "sec", "secs", "pt", "qt",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
# "No." / "Nos."는 뒤에 숫자가 올 때만 약어 ("I said no."는 문장 끝이므로 자른다)
NUMBER_ABBREVIATIONS = {"no", "nos"}
# a.m., u.s.a. 같은 "한 글자 + 마침표" 반복은 목록에 없어도 약어로 본다
_ACRONYM = re.compile(r"(?:[a-z]\.){2,}[a-z]?")

# 3. URL·이메일·파일 이름으로 끝나는 단어
_WEB_SUFFIX = re.compile(
    r"\.(?:com|org|net|edu|gov|io|co|uk|kr|jp|de|fr|ca|au|us|ly|me|tv|info|biz"
    r"|png|jpe?g|gif|bmp|svg|webp|csv|tsv|txt|pdf|docx?|xlsx?|pptx?|json|xml|html?"
    r"|zip|rar|gz|mp3|mp4|wav|avi|mov|mkv|exe|apk|py|js)\b"
)

# 5. 이모티콘: 눈(:;=) + 코(-^') 선택 + 입
_EMOTICON = re.compile(r"[:;=][-^'o]?[)(\]\[DdPpOo/\\|*3]")

# 단어 앞뒤에 붙는 따옴표·괄호 (약어/URL 판단 전에 떼어 냄)
_WRAP = "\"'“”‘’()[]{}<>"
_TRAILING = ",;:!?"


def _word_at(text: str, a: int) -> tuple[str, int]:
    """Whitespace-delimited word containing character `a`, and its start index."""
    start = a
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = a
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end], start


def is_protected(text: str, a: int) -> bool:
    """True when the punctuation character text[a] must not be a cut point."""
    return protection_rule(text, a) != 0


def protection_rule(text: str, a: int) -> int:
    """Which rule keeps text[a] from being a cut point: 1, 2, 3, 5 or 6 (list number); 0 = cut."""
    c = text[a]
    prev = text[a - 1] if a > 0 else ""
    nxt = text[a + 1] if a + 1 < len(text) else ""

    # 1. 숫자 안: 앞뒤가 바로 숫자 (3.5 / 1,000 / 10:30 / 192.168.0.1 / v2.1.3)
    if c in ".,:" and prev.isdigit() and nxt.isdigit():
        return 1

    # 5. 이모티콘 (:) ;) :-( :D) — 구두점이 이모티콘의 눈 부분이고, 뒤에 글자가 바로 붙지 않은 경우
    if c in ":;":
        m = _EMOTICON.match(text, a)
        if m and (m.end() >= len(text) or not text[m.end()].isalnum()):
            return 5
    # ^^; ^_^; 같은 이모티콘의 세미콜론
    if c == ";" and "^" in text[max(0, a - 3):a]:
        return 5

    word, start = _word_at(text, a)
    # 단어 앞의 따옴표·괄호를 떼어 내고, 그만큼 위치(start)도 옮긴다
    stripped = word.lstrip(_WRAP)
    start += len(word) - len(stripped)
    core = stripped.rstrip(_WRAP).rstrip(_TRAILING)  # 끝의 따옴표·괄호·쉼표 등 제거 (마침표는 유지)
    idx = a - start  # 단어 안에서 이 구두점의 위치
    low = core.lower()

    # 3. URL·이메일·파일 이름 안의 구두점 (단어 끝에 붙은 문장 부호는 제외)
    if 0 <= idx < len(core.rstrip(".")):
        if "://" in low or low.startswith("www.") or ("@" in low and "." in low):
            return 3
        if c == "." and _WEB_SUFFIX.search(low.rstrip(".")):
            return 3

    # 6. 맨 앞 목록 번호 "1." "2." "a." — 글 맨 앞이거나 앞 문장이 끝난 직후에 오는 번호의 마침표
    if c == "." and re.fullmatch(r"(?:\d{1,2}|[a-z])\.", low) and idx == len(core) - 1:
        before = text[:start].rstrip()
        if not before or before[-1] in ".!?:;":
            return 6

    # 2. 약어·호칭 안이나 끝의 마침표
    if c == "." and 0 <= idx < len(core):
        key = low.rstrip(".")
        if key in ABBREVIATIONS or _ACRONYM.fullmatch(low) or _ACRONYM.fullmatch(key):
            return 2
        if key in NUMBER_ABBREVIATIONS and text[a + 1:].lstrip()[:1].isdigit():
            return 2
    return 0


# rule_cut_points(stats=...)가 채우는 규칙별 발동 횟수 키
RULE_COUNT_KEYS = (
    "rule1_number", "rule2_abbrev", "rule3_web", "rule4_run_merged",
    "rule5_emoticon", "rule6_list_number", "rule6_empty_merged",
)
_RULE_KEY = {1: "rule1_number", 2: "rule2_abbrev", 3: "rule3_web", 5: "rule5_emoticon", 6: "rule6_list_number"}


def rule_cut_points(text: str, offsets: list, input_ids: list, punct_ids: set, stats: dict = None) -> list:
    """Token positions to cut at (inclusive segment ends), following rules 1-6.

    offsets / input_ids: from the tokenizer with return_offsets_mapping=True; position 0
    is [CLS] and the last position is [SEP]. When `stats` is a dict, the number of times
    each rule fired is added to it (keys in RULE_COUNT_KEYS).
    """
    # [로그] stats를 넘기면 규칙별로 몇 번 발동했는지 센다 (실험 로그 B2)
    def bump(key):
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    length = len(input_ids)
    # 토큰마다 글자(문자·숫자)를 포함하는지. [CLS]/[SEP]는 offset이 (0, 0)이라 False
    has_word = [any(ch.isalnum() for ch in text[a:b]) for a, b in offsets]

    # 1·2·3·5: 구두점 토큰 중 보호 대상이 아닌 것만 후보 ([CLS], [SEP] 제외)
    cuts: list[int] = []
    for p in range(1, length - 1):
        if input_ids[p] not in punct_ids:
            continue
        a, b = offsets[p]
        if b <= a:
            continue
        rule = protection_rule(text, a)
        if rule:
            bump(_RULE_KEY[rule])
            continue
        # 4. 연속·반복 구두점: 직전 후보와 사이에 글자(문자·숫자)가 없으면 같은 묶음 → 마지막 위치에서 한 번만 자름
        if cuts and not any(ch.isalnum() for ch in text[offsets[cuts[-1]][1]:a]):
            cuts[-1] = p
            bump("rule4_run_merged")
            continue
        cuts.append(p)

    # 6. 글자가 하나도 없는 조각이 생기면 그 경계를 없애 옆 조각에 붙인다.
    #    문장 끝 마침표, 맨 앞 목록 번호 "1.", 구두점만 있는 조각이 여기서 사라진다.
    while cuts:
        bounds = [0] + cuts + [length - 1]  # 조각 i = (bounds[i], bounds[i+1]] 구간의 토큰
        empty = next(
            (i for i in range(len(bounds) - 1) if not any(has_word[bounds[i] + 1:bounds[i + 1] + 1])),
            None,
        )
        if empty is None:
            break
        bump("rule6_empty_merged")
        if empty == len(bounds) - 2:
            cuts.pop()  # 마지막 조각이 비면 → 앞 조각에 붙임
        elif empty == 0:
            cuts.pop(0)  # 첫 조각이 비면 → 뒤 조각에 붙임
        else:
            cuts.pop(empty - 1)  # 가운데 조각이 비면 → 앞 조각에 붙임
    return cuts
