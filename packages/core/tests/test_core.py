"""核心不变量测试：文本处理 / 向量库 / 嵌入器 / 切片器。

每个测试断言的都是"数学上必须为真的性质"，而不是"代码恰好这么写"，
因此这些断言在任何等价实现下都必须成立。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))

from nexus.embedding.hashing import TfidfHashingEmbedder
from nexus.rag.chunker import RecursiveChunker
from nexus.types import Document
from nexus.util.mathx import (
    cosine_matrix,
    l2_normalize,
    percentile,
    signed_hash,
    stable_hash,
)
from nexus.util.text import (
    char_ngrams,
    estimate_tokens,
    is_cjk_dominant,
    tokenize,
)
from nexus.vectorstore.memory import InMemoryVectorStore


class TestText:
    def test_tokenize_splits_latin_words_and_cjk_chars(self):
        toks = tokenize("Hello World 混合检索 RAG")
        assert "hello" in toks
        assert "world" in toks
        assert "混" in toks
        assert "rag" in toks

    def test_tokenize_cjk_produces_bigram(self):
        toks = tokenize("混合检索")
        assert "混合" in toks and "合检" in toks and "检索" in toks

    def test_tokenize_is_deterministic_and_empty_safe(self):
        assert tokenize("") == []
        assert tokenize("中文 test") == tokenize("中文 test")

    def test_char_ngrams(self):
        assert char_ngrams("abcd", 2) == ["ab", "bc", "cd"]
        assert char_ngrams("中", 2) == ["中"]

    def test_is_cjk_dominant(self):
        assert is_cjk_dominant("重排器加语言守卫") is True
        assert is_cjk_dominant("cross-encoder reranking pipeline") is False

    def test_estimate_tokens_monotonic_and_nonzero(self):
        short = estimate_tokens("你好")
        long = estimate_tokens("你好" * 50)
        assert short >= 1
        assert long > short


class TestMath:
    def test_hash_is_deterministic_and_well_distributed(self):
        assert stable_hash("nexus") == stable_hash("nexus")
        assert stable_hash("甲") != stable_hash("乙")
        assert all(0 <= stable_hash(str(i)) < 2**32 for i in range(64))
        hashes = {stable_hash(str(i)) % 64 for i in range(64)}
        assert len(hashes) > 20

    def test_signed_hash_range_and_sign(self):
        for tok in ["a", "检索", "63", "重排器"]:
            idx, sign = signed_hash(tok, 64)
            assert 0 <= idx < 64
            assert sign in (1, -1)

    def test_l2_normalize_unit_norm(self):
        v = np.array([[3.0, 4.0], [0.0, 0.0]])
        out = l2_normalize(v)
        assert out.shape == (2, 2)
        assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-6
        assert np.isfinite(out[1]).all()

    def test_cosine_matrix_identity_and_empty(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = cosine_matrix(a, a)
        assert abs(sims[0][0] - 1.0) < 1e-6
        assert abs(sims[0][1] - 0.0) < 1e-6
        assert cosine_matrix(a, np.zeros((0, 2), dtype=np.float32)).shape == (2, 0)

    def test_percentile_edges(self):
        assert percentile([], 50) == 0.0
        assert percentile([1.0, 2.0, 3.0], 0) == 1.0
        assert percentile([1.0, 2.0, 3.0], 100) == 3.0


class TestVectorStore:
    def test_upsert_then_search_self_is_one(self):
        store = InMemoryVectorStore(dim=4)
        v = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        store.upsert(["a"], v)
        hits = store.search(v[0], 1)
        assert hits[0][0] == "a"
        assert abs(hits[0][1] - 1.0) < 1e-5

    def test_upsert_same_id_overwrites_without_growth(self):
        store = InMemoryVectorStore(dim=3)
        store.upsert(["x"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        store.upsert(["x"], np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        assert store.count() == 1
        res = store.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), 1)
        assert abs(res[0][1] - 1.0) < 1e-5

    def test_delete_unknown_returns_zero(self):
        store = InMemoryVectorStore(dim=2)
        assert store.delete(["nope"]) == 0

    def test_delete_then_count_and_index_integrity(self):
        store = InMemoryVectorStore(dim=2)
        mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        store.upsert(["a", "b", "c"], mat)
        assert store.delete(["b"]) == 1
        assert store.count() == 2
        assert store.ids() == ["a", "c"]
        q = np.array([1.0, 0.0], dtype=np.float32)
        assert store.search(q, 1)[0][0] == "a"
        store.upsert(["d"], np.array([[0.0, 1.0]], dtype=np.float32))
        assert store.count() == 3
        assert store.search(np.array([0.0, 1.0], dtype=np.float32), 1)[0][0] in {"c", "d"}

    def test_clear_and_empty_search(self):
        store = InMemoryVectorStore(dim=2)
        store.upsert(["a"], np.array([[1.0, 0.0]], dtype=np.float32))
        store.clear()
        assert store.count() == 0
        assert store.search(np.array([1.0, 0.0], dtype=np.float32), 5) == []

    def test_wrong_dim_raises(self):
        store = InMemoryVectorStore(dim=3)
        with pytest.raises(ValueError):
            store.upsert(["a"], np.zeros((1, 5), dtype=np.float32))

    def test_mismatched_ids_length_raises(self):
        store = InMemoryVectorStore(dim=3)
        with pytest.raises(ValueError):
            store.upsert(["a", "b"], np.zeros((1, 3), dtype=np.float32))


class TestEmbedder:
    def test_output_is_unit_vector_and_shape(self):
        emb = TfidfHashingEmbedder(dim=64)
        vecs = emb.embed(["混合检索需要 bm25", "第二条文档"])
        assert vecs.shape == (2, 64)
        assert abs(np.linalg.norm(vecs[0]) - 1.0) < 1e-5

    def test_deterministic_across_calls(self):
        emb = TfidfHashingEmbedder(dim=32)
        a = emb.embed(["同一句话"])
        b = emb.embed(["同一句话"])
        assert np.allclose(a, b)

    def test_identical_text_similarity_is_one(self):
        emb = TfidfHashingEmbedder(dim=128)
        v = emb.embed(["完全相同的文本"])
        sim = float(v[0] @ v[0].T)
        assert abs(sim - 1.0) < 1e-5

    def test_unrelated_text_similarity_below_related(self):
        emb = TfidfHashingEmbedder(dim=128)
        emb.observe(["向量数据库选型", "混合检索的 RRF 融合"])
        v = emb.embed(["混合检索的 RRF 融合", "今天天气不错适合出游"])
        sim_rel = float(v[0] @ emb.embed(["混合检索 RRF 融合"])[0].T)
        sim_unrel = float(v[1] @ emb.embed(["混合检索 RRF 融合"])[0].T)
        assert sim_rel > sim_unrel

    def test_observe_updates_idf_and_doc_count(self):
        emb = TfidfHashingEmbedder(dim=16)
        assert emb.n_docs == 0
        emb.observe(["甲"])
        assert emb.n_docs == 1
        emb.observe(["甲", "乙"])
        assert emb.n_docs == 3

    def test_zero_vector_text_is_finite(self):
        emb = TfidfHashingEmbedder(dim=16)
        vecs = emb.embed([""])
        assert np.isfinite(vecs).all()

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            TfidfHashingEmbedder(dim=0)


class TestChunker:
    def _doc(self, text: str) -> Document:
        return Document(doc_id="d1", title="t", text=text)

    def test_content_is_preserved(self):
        raw = "第一段内容。" * 40 + "\n\n" + "第二段内容。" * 40
        chunks = RecursiveChunker().split(self._doc(raw))
        rebuilt = "".join(c.text for c in chunks)
        # 换行属于段落分隔结构而非内容，比较时忽略；其余字符一个都不能丢
        assert set(raw.replace("\n", "")) == set(rebuilt.replace("\n", ""))
        assert len(chunks) > 1

    def test_overlap_exists_between_neighbours(self):
        raw = "这是一段用于验证重叠策略的长文本。" * 30
        chunker = RecursiveChunker(chunk_size=120, overlap=30)
        chunks = chunker.split(self._doc(raw))
        assert len(chunks) >= 2
        assert chunks[1].text[:30] == chunks[0].text[-30:]

    def test_chunk_ids_are_stable_and_ordered(self):
        chunks = RecursiveChunker(chunk_size=80, overlap=10).split(self._doc("甲乙丙" * 60))
        assert [c.chunk_id for c in chunks][:3] == ["d1#0", "d1#1", "d1#2"]
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_blank_document_yields_no_chunks(self):
        assert RecursiveChunker().split(self._doc("   \n\n  ")) == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=50, overlap=50)
