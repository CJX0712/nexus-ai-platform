# 系统架构

## 1. 设计原则

### 端口与适配器（六边形架构）

所有外部依赖被抽象为 Protocol，业务代码只依赖接口：

```
   业务编排 (RAGPipeline / ReactAgent)
        |  依赖接口，不依赖实现
        v
   Protocol 层 (protocols.py)
        ^  运行时注入具体实现
        |
   适配器实现 (Mock / Ollama / OpenAI / InMemory / BM25 / ...)
```

收益：
- 单元测试无需网络、无需 Key、无需数据库 → 用 fake 跑完整路径
- 同类替换只改 `container.py` 一处（如内存向量库换成 Qdrant）
- 模块可以独立演进，不会互相污染

### 单一职责

每个模块只做一件事，且可以被一句话说清楚：

| 模块 | 唯一职责 |
|---|---|
| `gateway` | HTTP 协议转换与参数校验 |
| `orchestrator` | 决定这次请求走 RAG 还是 Agent |
| `rag.pipeline` | 串起 RAG 的全流程 |
| `rag.chunker` | 把文档切成合适粒度的证据单元 |
| `retriever.bm25` | 关键词精确匹配（增量可维护） |
| `retriever.hybrid` | 融合稀疏与稠密两路信号 |
| `reranker` | 决定是否值得精排，以及怎么排 |
| `embedding` | 文本转向量 |
| `vectorstore` | 存向量、查最近邻 |
| `llm` | 把厂商差异吸收掉，提供统一 chat 接口 |
| `tools` | 工具注册与安全执行 |
| `agent` | ReAct 循环控制 |
| `memory` | 短期窗口与长期摘要 |
| `observability` | span 追踪与指标 |
| `eval` | 把质量变成可比较的数字 |

### 依赖方向

依赖只向下，`gateway` 不知道 `vectorstore` 的存在：

```
gateway -> orchestrator -> rag.pipeline -> retriever -> {embedding, vectorstore, bm25}
                                       -> reranker
                                       -> llm
                        -> agent.react -> {llm, tools}
```

---

## 2. 数据流

### 摄入

```
POST /api/v1/ingest
  -> chunker.split(document)          按段落/句子/字符递归切分，带 15~25% 重叠
  -> embedder.observe(chunks)         哈希嵌入在此更新 IDF（稠密嵌入为空操作）
  -> embedder.embed(chunks)           子词 TF-IDF 哈希投影 / Ollama bge-m3
  -> vectorstore.upsert(ids, vectors) 写入 numpy 矩阵
  -> bm25.add_texts(ids, texts)       更新 tf/df/长度统计
```

### 问答（RAG）

```
POST /api/v1/chat
  -> retrieve(query)
        bm25.search   -> top 4k 候选
        vector.search -> top 4k 候选
        RRF 融合      -> top k
  -> reranker.rerank   (可选；语言不匹配时自动旁路)
  -> build_messages    (带编号上下文，保证引用可追溯)
  -> llm.acomplete     (主 provider 失败自动降级到 Mock)
  -> collect_citations (从答案里的 [k] 回填证据；未标注则用 top-1 兜底)
```

### 问答（Agent）

```
POST /api/v1/chat  use_agent=true
  -> loop (最多 max_steps 次):
        构造 scratchpad prompt
        llm 生成 -> 解析 行动/输入
        命中 最终答案 -> 结束
        解析不出 -> 视为直接作答，结束（不抛错）
        有工具调用 -> registry.run -> 结果收敛为 Observation
  -> 返回答案 + 工具调用轨迹
```

### 流式

与上面的 RAG 流程一致，区别是 `llm.astream` 逐片产出，网关包装成 SSE 帧：

```
event: citation  data: {...}
event: token     data: {"delta":"字"}
event: done      data: {"answer":..., "session_id":..., "usage":...}
event: error     data: {"message":...}
```

---

## 3. 关键技术决策

### 混合检索为什么用 RRF 而不是加权求和

BM25 分数无上界，余弦相似度被限制在 [-1,1]。加权前必须归一化，而归一化极值随语料漂移，
同一套参数换个知识库就失真。RRF 只用名次信息，天然免疫量纲差异：

```
rrf_score(d) = Σ_r 1 / (k + rank_r(d))     k 取 60
```

候选宽度设为最终数量的 4 倍：太窄会让「排在两路都靠后的正确答案」彻底丢失。

### 为什么自研 BM25

主流 BM25 库要求一次性构造语料，每次新增文档都要全量重建索引，动态知识库下是 O(N²)。
这里维护 `tf / df / 文档长度` 的增量统计，支持单条增删。

### 重排的语言守卫（本项目最重要的经验）

交叉编码器在查询语言与训练语言错配时给出的分数近乎随机，
而它的输出会覆盖融合排序里的所有其他信号，导致**启用重排反而更差**。

实测：中文查询 top-1 命中从 11/12 掉到 4/12。

因此 `GuardedReranker` 在中文查询 + 非中文重排器时自动降级为恒等/覆盖重排，
并把 `last_decision` 写进 span，避免它变成沉默的黑盒。**默认不启用重排。**

### 为什么要 Mock 模型

不是返回固定字符串的"假模型"，而是做抽取式生成：从 prompt 里解析带编号的上下文，
按词汇重叠选出最相关的一条，拼出答案并标注来源。这样即使在完全离线的环境，
"检索到的资料有没有被真正用到"依然可被机械断言。

### 离线可验证

每个外部依赖都有零依赖实现，因此 `pytest` 与 E2E 在**无网络、无 Key、无数据库**下全绿：

| 能力 | 离线默认实现 | 生产实现 |
|---|---|---|
| LLM | Mock 抽取式 | Ollama / vLLM / OpenAI 兼容 |
| 嵌入 | TF-IDF 哈希 | Ollama bge-m3 / OpenAI embeddings |
| 向量库 | numpy 精确检索 | Qdrant / pgvector |
| 重排 | 覆盖率启发式 | CrossEncoder（带守卫） |
| 摘要 | 确定性压缩 | LLM 摘要 |

---

## 4. 扩展点

| 想换什么 | 改哪里 |
|---|---|
| 向量数据库 | 实现 `VectorStore` 协议，改 `container.py` 装配 |
| 大模型厂商 | 实现 `LLMProvider` 协议，注册进 `LLMRouter` |
| 重排模型 | 实现 `Reranker` 协议，套一层 `GuardedReranker` |
| 新增工具 | 实现 `Tool` 协议，`registry.register` |
| 追踪后端 | 实现 `Tracer` 协议，可导出 OpenTelemetry |

---

## 5. 性能设计

- **全链路 async**：慢速 IO 不占用线程；CPU 密集的嵌入通过 `asyncio.to_thread` 卸载
- **序列化**：orjson 替代标准库 json，吞吐提升 2~3 倍
- **向量运算**：走 numpy/BLAS 矩阵乘；向量库块式扩容，避免每次 upsert 重建矩阵
- **流式输出**：SSE 降低首字延迟；`Cache-Control: no-cache` + `X-Accel-Buffering: no` 防止反代缓冲
- **连接复用**：httpx 连接池 + keep-alive，模型服务不会被反复握手拖慢
- **可观测**：每个阶段进 span，性能归因可定位到「检索 / 重排 / 生成」中的具体阶段

---

## 6. 实测数据

在 `eval/datasets` 固定样例集（12 条查询 / 5 篇文档）上的对比：

| 嵌入方案 | recall@5 | mrr@5 | ndcg@5 | 引用准确率 | p50 延迟 |
|---|---|---|---|---|---|
| 零依赖 TF-IDF 哈希 | 1.000 | 0.854 | 0.891 | 0.917 | 2 ms |
| Ollama bge-m3（1024 维） | 1.000 | **1.000** | **1.000** | 0.917 | 145 ms |

稠密嵌入换来的收益是「12 条查询全部 top-1 命中」，代价是从毫秒级到百毫秒级。
离线演示、CI 门禁用哈希嵌入；追求质量的场景切 bge-m3。作者：**晨星**
