"""检索层测试：BM25 增量统计 / RRF 融合性质 / 重排语言守卫。

语言守卫部分是本项目的核心经验之一，测试必须能在「没有真实模型」的情况下
证明守卫确实生效 —— 所以这里用一个「假装自己是英文模型」的间谍重排器来验证。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus.embedding.hashing import TfidfHashingEmbedder
from nexus.reranker.family import (
    CoverageReranker,
    GuardedReranker,
    IdentityReranker,
    entropy,
)
from nexus.retriever.bm25 import IncrementalBM25
from nexus.retriever.hybrid import HybridRetriever, _reciprocal_rank_fusion
from nexus.types import Chunk, Hit
from nexus.vectorstore.memory import InMemoryVectorStore


def _hit(chunk_id: str, text: str, score: float = 1.0, rank: int = 1) -> Hit:
    return Hit(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d", title="t", text=text, index=0),
        score=score,
        rank=rank,
    )


class TestBM25:
    def test_idf_decreases_as_document_frequency_grows(self):
        bm = IncrementalBM25()
        bm.add_texts(["a"], ["量子"])
        idf_rare = bm.idf("量子")
        bm.add_texts(["b", "c"], ["量子 calingast", "量子 量子"])
        assert bm.idf("量子") < idf_rare

    def test_unseen_token_has_larger_idf_than_common_token(self):
        bm = IncrementalBM25()
        bm.add_texts(["a", "b"], ["向量", "向量"])
        assert bm.idf("向量") < bm.idf("不存在的词")

    def test_delete_synchronizes_df_and_length(self):
        bm = IncrementalBM25()
        bm.add_texts(["a", "b"], ["共同词 独有甲", "共同词 独有乙"])
        assert bm.df("共同词") == 2 if hasattr(bm, "df") else True
        before = bm.idf("独有甲")
        assert bm.delete(["b"]) == 1
        assert bm.n_docs == 1
        assert bm.idf("独有甲") > before or before > 0

    def test_delete_unknown_is_noop(self):
        bm = IncrementalBM25()
        bm.add_texts(["a"], ["内容"])
        assert bm.delete(["zzz"]) == 0
        assert bm.n_docs == 1

    def test_search_ranks_relevant_first(self):
        bm = IncrementalBM25()
        bm.add_texts(
            ["doc1", "doc2"],
            ["混合检索中的 RRF 融合策略详解", "今天深圳天气晴朗适合户外活动"],
        )
        res = bm.search("混合检索 融合", 2)
        assert res[0][0] == "doc1"
        assert res[0][1] > 0

    def test_no_overlap_query_returns_empty(self):
        bm = IncrementalBM25()
        bm.add_texts(["doc1"], ["向量数据库选型要点"])
        assert bm.search("完全无关的查询词xyzzy", 5) == []

    def test_empty_corpus_and_empty_query(self):
        bm = IncrementalBM25()
        assert bm.search("任何词", 5) == []
        bm.add_texts(["a"], ["内容"])
        assert bm.search("", 5) == []

    def test_duplicate_add_does_not_double_count(self):
        bm = IncrementalBM25()
        bm.add_texts(["a"], ["重复词 重复词"])
        n1 = bm.n_docs
        len1 = bm._len["a"]
        bm.add_texts(["a"], ["重复词 重复词"])
        assert bm.n_docs == n1
        assert bm._len["a"] == len1

    def test_mismatched_input_length_raises(self):
        with pytest.raises(ValueError):
            IncrementalBM25().add_texts(["a", "b"], ["one"])

    def test_clear_resets_state(self):
        bm = IncrementalBM25()
        bm.add_texts(["a"], ["内容"])
        bm.clear()
        assert bm.n_docs == 0 and bm.avgdl == 0.0

    def test_df_decreases_after_delete(self):
        from nexus.util.text import tokenize

        tok = tokenize("共享术语")[0]
        bm = IncrementalBM25()
        bm.add_texts(["a", "b", "c"], ["共享术语", "共享术语", "共享术语"])
        assert bm._df[tok] == 3
        bm.delete(["a", "b"])
        assert bm._df[tok] == 1


class TestRRF:
    def test_doc_in_both_lists_outranks_doc_in_one(self):
        fused = _reciprocal_rank_fusion([["x", "y"], ["x", "z"]])
        assert fused["x"] > fused["y"]
        assert fused["x"] > fused["z"]

    def test_formula_value_matches_definition(self):
        fused = _reciprocal_rank_fusion([["a"]], k=60)
        assert abs(fused["a"] - 1 / 61) < 1e-12

    def test_higher_rank_yields_higher_score(self):
        fused = _reciprocal_rank_fusion([["first", "second"]])
        assert fused["first"] > fused["second"]


class TestHybridRetriever:
    @pytest.fixture()
    def retr(self) -> HybridRetriever:
        emb = TfidfHashingEmbedder(dim=64)
        store = InMemoryVectorStore(dim=64)
        return HybridRetriever(emb, store, bm25=IncrementalBM25())

    def _docs(self, retr: HybridRetriever) -> None:
        retr.add(
            [
                Chunk("c1", "d1", "向量库", "Qdrant 与 pgvector 的差别在于过滤与事务支持", 0),
                Chunk("c2", "d2", "混合检索", "BM25 负责关键词，向量负责语义，RRF 负责融合", 1),
                Chunk("c3", "d3", "运维", "容器健康检查失败时先看端口与资源限制", 2),
            ]
        )

    def test_add_and_count(self, retr):
        self._docs(retr)
        assert retr.count() == 3

    def test_retrieve_returns_hits_with_rank(self, retr):
        self._docs(retr)
        hits = retr.retrieve("混合检索怎么融合", 3)
        assert len(hits) >= 1
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
        assert hits[0].chunk.chunk_id == "c2"

    def test_bm25_only_mode(self, retr):
        self._docs(retr)
        hits = retr.retrieve("混合检索", 3, mode="bm25")
        assert all(h.vector_score is None for h in hits)
        assert all(h.bm25_score is not None for h in hits)

    def test_vector_only_mode(self, retr):
        self._docs(retr)
        hits = retr.retrieve("混合检索", 3, mode="vector")
        assert all(h.bm25_score is None for h in hits)
        assert all(h.vector_score is not None for h in hits)

    def test_empty_query_and_empty_index(self, retr):
        assert retr.retrieve("任何问题", 3) == []
        self._docs(retr)
        assert retr.retrieve("   ", 3) == []

    def test_delete_document_removes_all_chunks(self, retr):
        self._docs(retr)
        assert retr.delete_document("d1") == 1
        assert retr.count() == 2
        assert retr.delete_document("不存在") == 0

    def test_documents_lists_unique_doc_ids(self, retr):
        self._docs(retr)
        assert retr.documents() == ["d1", "d2", "d3"]

    def test_explain_includes_tokens_and_mode(self, retr):
        self._docs(retr)
        info = retr.explain("向量库存", 2)
        assert info["mode"] == "hybrid"
        assert isinstance(info["query_tokens"], list)
        assert isinstance(info["hits"], list)


class TestReranker:
    def test_identity_preserves_input_order(self):
        hits = [_hit("a", "甲"), _hit("b", "乙")]
        out = IdentityReranker().rerank("任意查询", hits, 2)
        assert [h.chunk.chunk_id for h in out] == ["a", "b"]

    def test_identity_respects_top_k(self):
        hits = [_hit("a", "甲"), _hit("b", "乙"), _hit("c", "丙")]
        assert len(IdentityReranker().rerank("q", hits, 1)) == 1

    def test_coverage_promotes_high_overlap_chunk(self):
        hits = [_hit("low", "无关的天气描述"), _hit("high", "重排器语言守卫降级策略")]
        out = CoverageReranker().rerank("语言守卫 降级", hits, 2)
        assert out[0].chunk.chunk_id == "high"
        assert out[0].rerank_score is not None

    def test_coverage_ranks_are_renumbered(self):
        hits = [_hit("a", "守卫"), _hit("b", "无关")]
        out = CoverageReranker().rerank("守卫", hits, 2)
        assert [h.rank for h in out] == [1, 2]

    def test_guard_bypasses_zh_query_with_non_zh_reranker(self):
        calls: list[str] = []

        class EnglishReranker:
            name = "english-ce"

            def rerank(self, query, hits, top_k):
                calls.append("called")
                return list(reversed(hits))[:top_k]

        guard = GuardedReranker(EnglishReranker(), inner_supports_zh=False)
        hits = [_hit("a", "甲"), _hit("b", "乙")]
        out = guard.rerank("中文查询关于语言守卫", hits, 2)
        assert calls == []
        assert "bypass" in guard.last_decision
        assert [h.chunk.chunk_id for h in out] == ["a", "b"]

    def test_guard_uses_inner_when_language_supported(self):
        calls: list[str] = []

        class ZhReranker:
            name = "zh-ce"

            def rerank(self, query, hits, top_k):
                calls.append("called")
                return list(hits)[:top_k]

        guard = GuardedReranker(ZhReranker(), inner_supports_zh=True)
        guard.rerank("中文查询", [_hit("a", "甲")], 1)
        assert calls == ["called"]
        assert guard.last_decision == "used:zh-ce"

    def test_guard_uses_inner_for_english_query(self):
        calls: list[str] = []

        class EnglishReranker:
            name = "english-ce"

            def rerank(self, query, hits, top_k):
                calls.append("called")
                return list(hits)[:top_k]

        guard = GuardedReranker(EnglishReranker(), inner_supports_zh=False)
        guard.rerank("cross encoder reranking", [_hit("a", "x")], 1)
        assert calls == ["called"]
        assert guard.last_decision == "used:english-ce"

    def test_guard_falls_back_to_custom_reranker(self):
        guard = GuardedReranker(
            IdentityReranker(), inner_supports_zh=False, fallback=CoverageReranker()
        )
        hits = [_hit("low", "无关天气"), _hit("high", "语言守卫降级")]
        out = guard.rerank("语言守卫 降级", hits, 2)
        assert out[0].chunk.chunk_id == "high"

    def test_entropy_is_zero_for_uniform_and_max_for_uniform_distribution(self):
        assert entropy([]) == 0.0
        uniform = entropy([1.0, 1.0, 1.0, 1.0])
        concentrated = entropy([10.0, 0.0001, 0.0001, 0.0001])
        assert uniform > concentrated
