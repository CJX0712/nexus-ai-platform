"""文档切片。

策略：结构优先递归切分 —— 先按空行/标点切到段落，过长再切句子，最后按字符硬切。
带 overlap，避免关键句被边界截断导致两半都检索不到。

不变量：
    1. 切片顺序拼接后必须能覆盖原文的所有非空白内容
    2. 相邻切片之间存在 overlap_len 个字符的重叠
    3. 空白文档返回空列表，不产生垃圾切片
"""

from __future__ import annotations

import re
import uuid

from nexus.types import Chunk, Document

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_SENT_SPLIT = re.compile(r"(?<=[。！？；.!?;])\s*")


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]


class RecursiveChunker:
    """递归字符切片器。

    Args:
        chunk_size: 目标切片字符数
        overlap: 相邻切片重叠字符数，建议为 chunk_size 的 15%~25%
        min_size: 小于该长度的切片会与下一片合并，避免碎片
    """

    name = "recursive"

    def __init__(self, chunk_size: int = 420, overlap: int = 80, min_size: int = 40) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        pieces: list[str] = []
        for para in _PARA_SPLIT.split(text):
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                pieces.append(para)
                continue
            buf = ""
            for sent in _SENT_SPLIT.split(para):
                if not sent:
                    continue
                if len(buf) + len(sent) <= self.chunk_size:
                    buf += sent
                else:
                    if buf:
                        pieces.append(buf)
                    buf = sent
                    while len(buf) > self.chunk_size:
                        pieces.append(buf[: self.chunk_size])
                        buf = buf[self.chunk_size - self.overlap :]
            if buf:
                pieces.append(buf)
        merged = self._merge_small(pieces)
        return self._apply_overlap(merged)

    def _merge_small(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []
        out: list[str] = []
        for p in pieces:
            if (out and len(p) < self.min_size) or (out and len(out[-1]) < self.min_size):
                out[-1] = out[-1] + p
            else:
                out.append(p)
        return out

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        if len(pieces) <= 1 or self.overlap <= 0:
            return pieces
        out: list[str] = [pieces[0]]
        for i in range(1, len(pieces)):
            tail = out[-1][-self.overlap :]
            out.append(tail + pieces[i])
        return out

    def split(self, doc: Document) -> list[Chunk]:
        parts = self._split_text(doc.text)
        chunks: list[Chunk] = []
        for idx, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{idx}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=part,
                    index=idx,
                    metadata=dict(doc.metadata),
                )
            )
        return chunks
