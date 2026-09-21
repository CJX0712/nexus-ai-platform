"""中英文混排文本处理工具（零依赖）。

中文没有空格分词，所以这里统一采用「字符级 + 拉丁词」的混合切分：
- 连续拉丁字母/数字序列作为一个词
- 每个 CJK 汉字作为单字词，并额外产出相邻二字词（bigram）
这种切分对中英混排的关键词检索效果稳定，且完全不需要分词器依赖。
"""

from __future__ import annotations

import re
import unicodedata

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
_LATIN = r"A-Za-z0-9_"
_TOKEN_RE = re.compile(rf"[{_LATIN}]+|[{_CJK}]")
_CJK_RE = re.compile(rf"[{_CJK}]")
_STRIP_RE = re.compile(r"[ \t\r\n]+")


def normalize(text: str) -> str:
    """全角转半角、统一空白。"""
    text = unicodedata.normalize("NFKC", text)
    return _STRIP_RE.sub(" ", text).strip()


def _pieces(chunk: str) -> list[str]:
    """对一段不含空格的文本片切词。"""
    out: list[str] = []
    i = 0
    n = len(chunk)
    while i < n:
        m = _TOKEN_RE.match(chunk, i)
        if m is None:
            i += 1
            continue
        tok = m.group(0)
        if _CJK_RE.match(tok):
            # 汉字流：逐字 + 相邻二字词
            start = i
            end = start
            while end < n and _CJK_RE.match(chunk[end]):
                end += 1
            han = chunk[start:end]
            out.extend(han)
            out.extend(han[j : j + 2] for j in range(len(han) - 1))
            i = end
        else:
            out.append(tok.lower())
            i = m.end()
    return out


def tokenize(text: str) -> list[str]:
    """切词，返回小写词列表。"""
    if not text:
        return []
    parts = _STRIP_RE.split(normalize(text))
    out: list[str] = []
    for p in parts:
        out.extend(_pieces(p))
    return out


def char_ngrams(text: str, n: int = 2) -> list[str]:
    """字符级 n-gram 特征，用于向量化时的子词鲁棒性。"""
    s = normalize(text).replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def is_cjk_dominant(text: str, threshold: float = 0.3) -> bool:
    """判断文本是否以中文为主，用于重排器的语言守卫决策。"""
    if not text:
        return False
    cjk = len(_CJK_RE.findall(text))
    return cjk / max(len(text), 1) >= threshold


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1.5 字/token，英文约 4 字符/token。

    在没有 tokenizer 依赖的前提下给出稳定且可复现的估计值。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return max(1, int(cjk / 1.5 + rest / 4.0))


def truncate(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[:limit] + "..."
