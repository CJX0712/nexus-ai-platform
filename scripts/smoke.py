"""零依赖冒烟自检。

在没有任何外部服务（无 Ollama、无网络、无 API Key、无数据库）的前提下，
验证「摄入 -> 检索 -> 精排 -> 生成 -> 引用」整条链路真的能跑通。

用法：
    python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))

from nexus.config import Settings  # noqa: E402
from nexus.container import build_system  # noqa: E402

CORPUS = [
    (
        "混合检索白皮书",
        "混合检索把稀疏检索与稠密检索结合起来。稀疏检索以 BM25 为代表，"
        "它对关键词精确匹配非常敏感，适合型号、参数、专有名词。"
        "稠密检索用向量表达语义，能召回没有字面重叠但含义相近的内容。"
        "两种信号通过 RRF 融合，即用名次的倒数和作为最终分数，"
        "这样可以绕开两路分数量纲不一致的问题。RRF 公式中的 k 取 60 是常见经验值。\n\n"
        "为什么不能直接加权求和？因为 BM25 的分数没有上界，而余弦相似度被限制在负一到一之间。"
        "如果强行把两路分数线性加权，就必须先做归一化，而归一化所需的极值会随语料不断变化，"
        "导致同一套参数在不同知识库上表现漂移。RRF 只使用名次信息，天然免疫量纲差异，"
        "这让它在换语料、换嵌入模型之后依然保持稳定。\n\n"
        "工程上还要注意候选集宽度。如果只给向量通道和关键词通道各取十个候选，"
        "那么融合的候选池最多二十条，一旦正确答案排在两条通道的第十五名，融合后就彻底丢失了。"
        "实践中通常把候选宽度设为最终结果数量的四倍左右，再做融合截断。"
        "代价是多算一些相似度，但换来的是召回率的稳定提升，这笔交换通常是划算的。\n\n"
        "最后是文本预处理。中英文混排时，英文需要按空格和标点切词并转小写，"
        "中文则没有天然的空格边界。常见的做法是把连续汉字拆成单字，再补充相邻二字词作为特征，"
        "这样既避免引入分词器依赖，又保留了一定的词组信息。",
    ),
    (
        "重排的实践经验",
        "交叉编码器重排能显著提升排序质量，但有一个代价高昂的陷阱："
        "当重排模型的训练语言与查询语言不一致时，例如用英文重排器处理中文查询，"
        "它会给出接近随机的分数，并且因为它的输出直接覆盖其他所有信号，"
        "最终排序反而比不重排更差。实测中文查询的 top-1 命中会从十一次掉到四次。"
        "因此必须为重排器加上语言守卫，不匹配时自动降级为恒等重排。\n\n"
        "这个坑之所以难以发现，是因为它在英文评测集上完全看不出来。"
        "团队通常在英文基准上验证重排收益，得到正向结论后直接上线到中文业务，"
        "线上指标却悄悄劣化。等到业务方反馈答案变差时，往往已经过去了数周。"
        "排查时还容易误判成嵌入模型不好或切片不合理，从而白白重构整个检索链路。\n\n"
        "识别方法其实很简单：拿一小批中文查询，分别统计重排前后的 top-1 命中率。"
        "如果重排后的命中率明显低于重排前，几乎可以确定是语言错配。"
        "更稳妥的做法是把这个检查做进流水线，让重排器自己判断查询语言，"
        "在不匹配时自动旁路，并把降级原因记录到追踪数据里，避免它变成一个沉默的黑盒。\n\n"
        "另外一个常见误区是认为重排一定值得那点延迟。"
        "交叉编码器要对每个候选做一次完整的前向计算，候选数增加时会线性变慢。"
        "在低延迟要求的场景里，更经济的做法是先用融合检索把候选缩到二十条以内，"
        "再对这二十条做精排，最终取前五条送入生成环节。",
    ),
    (
        "向量数据库选型",
        "向量库负责存储嵌入并提供近似最近邻查询。"
        "内存实现适合单机与小数据量，优点是不需要外部服务、依赖最少、启动最快；"
        "生产环境通常选择 Qdrant 或 pgvector，前者专注向量检索且支持复杂过滤，"
        "后者能与现有 PostgreSQL 的事务和权限体系复用。选型时应先看是否需要"
        "混合过滤、多租户隔离与水平扩展，而不是单纯比较召回速度。\n\n"
        "索引结构决定了召回与精度的权衡。暴力检索给出百分之百准确的召回，"
        "在小数据量下往往比任何近似索引都快，因为省去了索引维护和图遍历的开销。"
        "HNSW 通过构建分层近邻图实现亚线性查询，代价是内存占用更高、构建更慢。"
        "IVF 先聚类再检索倒排桶，内存友好但需要合理设定桶数以平衡速度与召回。\n\n"
        "真正的决策往往不在技术层，而在组织层：团队是否已经有 PostgreSQL 的运维能力，"
        "是否需要向量数据与业务数据在同一事务里保持一致，是否有多租户隔离的合规要求。"
        "把这些问题回答清楚，选型结果通常就自然浮现了，不需要纠结基准测试里几毫秒的差异。\n\n"
        "一个务实的路径是先做接口抽象，让业务代码只依赖检索协议，"
        "初期用内存实现把端到端链路跑通并积累评测数据，"
        "等到数据量或并发真正成为瓶颈时，再替换成专用的向量数据库。"
        "只要协议边界清晰，这类替换通常只需要改一处装配代码。",
    ),
]

QUERY = "为什么要给重排器加语言守卫？"


async def main() -> int:
    t0 = time.perf_counter()
    s = Settings(provider="mock", data_dir=str(ROOT / "data"), top_k=3)
    sysm = build_system(s)

    total_chunks = 0
    for title, text in CORPUS:
        doc_id, n = sysm.pipeline.add_document(title, text)
        total_chunks += n
        print(f"[ingest] {title:<12} doc_id={doc_id} chunks={n}")
    assert total_chunks > 3, "切片数量异常，说明长文档未被正确切分"

    hits = sysm.pipeline.retrieve(QUERY, 3)
    print(f"\n[retrieve] query={QUERY}")
    for h in hits:
        v = "None" if h.vector_score is None else f"{h.vector_score:.3f}"
        b = "None" if h.bm25_score is None else f"{h.bm25_score:.3f}"
        print(f"  #{h.rank} score={h.score:.4f} bm25={b} vec={v} :: {h.chunk.text[:40]}...")
    assert hits, "检索没有返回任何结果"
    top_title = hits[0].chunk.title
    print(f"\n[expect] 期望命中《重排的实践经验》，实际命中《{top_title}》")

    ans = await sysm.pipeline.answer(QUERY, use_rag=True)
    print(f"\n[answer] {ans.answer}")
    print(f"[citations] {len(ans.citations)} 条 -> {[c.chunk.doc_id for c in ans.citations]}")

    ok = True
    if top_title != "重排的实践经验":
        print("FAIL: top-1 未命中期望文档")
        ok = False
    if not ans.citations:
        print("FAIL: 没有产生引用")
        ok = False
    if "语言守卫" not in ans.answer:
        print("FAIL: 答案没有真正用到检索到的资料")
        ok = False

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n{'PASS' if ok else 'FAIL'}  耗时 {elapsed:.0f} ms  切片总数 {total_chunks}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
