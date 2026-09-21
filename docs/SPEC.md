# Spec · nexus-ai-platform v0.1.0

> 基于：ARCHITECTURE v0.1.0
> 状态：已确认
> 作者：晨星
> 本文是开发与验收的唯一依据。

---

## 1. 产品定义

- **一句话**：一套端到端可运行的开源 AI 系统，覆盖检索增强生成与工具调用 Agent，每个模块可独立验证、可插拔替换。
- **目标用户**：需要把 RAG / Agent 能力落地到自己业务的工程团队；以及希望学习现代 AI 系统分层与评测方法的开发者。
- **核心问题**：市面上的 AI 示例要么依赖一堆外部服务导致跑不起来，要么是单体脚本无法验证与替换。

## 2. MVP 范围（锁定）

| 优先级 | 功能 | 验收摘要 |
|---|---|---|
| P0 | 文档摄入与切片 | 长文档必须切成多片，重叠率可调 |
| P0 | 混合检索 | BM25 + 向量 + RRF，三种模式可切换 |
| P0 | 带引用的生成 | 答案标注来源编号，引用可回溯到具体分片 |
| P0 | 流式输出 | SSE 输出 token / 引用 / 完成三类事件 |
| P0 | 真实本地模型 | 通过 Ollama 调用本地 GGUF 模型完成真实生成 |
| P0 | 稠密嵌入 | 可选接入 bge-m3，显著提升语义召回 |
| P1 | 工具调用 Agent | ReAct 循环，步数上限，解析失败优雅收敛 |
| P1 | 会话记忆 | 滑动窗口 + 概要压缩 |
| P1 | 链路追踪 | span 树，可导出给控制台做火焰图 |
| P1 | 回归评测 | 固定样例集 + 指标 + 阈值门禁 |
| P2 | Web 控制台 | 对话 / 知识库 / Trace 三面板 |

## 3. 明确不做（Out-of-Scope）

| 不做的功能 | 原因 | 何时考虑 |
|---|---|---|
| 用户注册登录与多租户隔离 | 定位是能力内核，鉴权属业务层，应由接入方实现 | v2.0 提供中间件与可扩展点 |
| 分布式的向量集群 | 单机与小数据量场景不需要，协议已预留 | 数据量达到百万级时 |
| 模型微调与训练 | 与推理服务职责不同，应独立成服务 | 有明确训练需求时 |
| 图形化工作流编排 | MVP 阶段收益不足 | 多步骤业务场景出现后 |
| 多模态（图片/语音） | 检索与生成链路尚未稳定 | 单模态质量达标后 |

## 4. 技术架构（版本锚定）

| 层 | 技术 | 版本 | 选择理由 |
|---|---|---|---|
| 语言 | Python | 3.13.14 | 生态与 AI 库支持最好 |
| Web 框架 | FastAPI | 0.141.1 | 原生 async + 自动生成 OpenAPI |
| ASGI 服务器 | uvicorn | 0.53.0 | 标准实现，无额外编译依赖 |
| 校验 | pydantic | 2.13.5 | v2 内核用 Rust，性能远好于 v1 |
| 配置 | pydantic-settings | 2.15.0 | 环境变量与类型校验合一 |
| HTTP 客户端 | httpx | 0.28.1 | 支持连接复用与流式响应 |
| 数值 | numpy | 2.5.3 | 向量运算走 BLAS |
| 序列化 | orjson | 3.12.0 | 比标准库 json 快 2~3 倍 |
| 配置格式 | PyYAML | 6.0.3 | 配置文件解析 |
| 前端 | React + TypeScript | 19.3.0 / 5.x | React 19 + Vite 构建 |
| 构建 | Vite | 8.3.0 | 冷启动快、HMR 稳定 |
| 图标 | lucide-react | 1.47.0 | 统一描边 SVG，禁用 emoji 图标 |
| 本地推理 | Ollama（llama.cpp 内核） | 当前版本 | 官方分发、跨平台、OpenAI 兼容端点 |
| 嵌入模型 | bge-m3 | 最新版 | 中英双语强，1024 维 |

**禁止引入需要源码编译的依赖**：目标环境可能缺少 Rust / MSVC 工具链，干净环境的"一键复现"必须依赖纯 Wheel。

## 5. API 端点清单

Base：`http://<host>:<port>`

| Method | Path | 功能 | 请求体 | 响应体 |
|---|---|---|---|---|
| GET | `/api/health` | 健康检查 | - | status / version / provider / uptime_s |
| POST | `/api/v1/ingest` | 摄入文档 | title, text, doc_id?, metadata? | doc_id, n_chunks, n_chars |
| GET | `/api/v1/documents` | 文档列表 | - | items[], total |
| GET | `/api/v1/documents/{doc_id}/chunks` | 分片预览 | limit | items[], total |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 | - | deleted, doc_id |
| POST | `/api/v1/search` | 检索 | query, top_k?, mode? | hits[], latency_ms, trace_id |
| POST | `/api/v1/chat` | 问答（非流式） | query, session_id?, use_rag?, use_agent?, top_k?, provider? | answer, citations[], tool_calls[], trace_id, usage |
| POST | `/api/v1/chat/stream` | 问答（SSE） | 同上 | token / citation / tool / done / error 事件流 |
| GET | `/api/v1/traces` | 追踪列表 | limit | items[], total |
| GET | `/api/v1/traces/{trace_id}` | 追踪详情 | - | spans[]（含父子关系与耗时） |
| GET | `/api/v1/metrics` | 运行指标 | - | requests, errors, p50/p95, tokens |

完整 OpenAPI 模式由服务自动生成，见 `/docs` 与 `/openapi.json`。

## 6. 数据契约

| 模型 | 关键字段 |
|---|---|
| `Message` | role（system/user/assistant/tool）, content, name? |
| `Document` | doc_id, title, text, metadata, created_at |
| `Chunk` | chunk_id（`{doc_id}#{index}`）, doc_id, title, text, index |
| `Hit` | chunk, score, rank, vector_score?, bm25_score?, rerank_score?, stage |
| `LLMResponse` | text, usage（prompt/completion/total_ms/ttft_ms）, model |
| `ToolResult` | ok, output, error |
| `Answer` | session_id, answer, citations[], tool_calls[], trace_id, usage |

## 7. 接口契约（Protocol）

实现必须满足 `nexus/protocols.py` 中定义的签名：

| Protocol | 核心方法 |
|---|---|
| `LLMProvider` | `acomplete(messages, temperature, max_tokens)`, `astream(...)`, `count_tokens(text)` |
| `Embedder` | `embed(texts) -> (n, dim) 单位向量`, `aembed`, `observe(texts)` |
| `VectorStore` | `upsert`, `delete`, `search -> [(id, score)]`, `count`, `clear` |
| `Retriever` | `add(chunks)`, `retrieve(query, top_k) -> [Hit]` |
| `Reranker` | `rerank(query, hits, top_k) -> [Hit]` |
| `Tool` | `parameters() -> JSON Schema`, `run(**kwargs) -> ToolResult` |
| `Tracer` | `start_trace`, `span(ctx)`, `finish_trace`, `get`, `list_traces` |

## 8. 设计 Token（前端）

- 图标统一 `lucide-react`，尺寸 16 / 20 / 24 px，禁止 emoji 充当功能图标
- 颜色必须走 CSS 变量，禁止硬编码（`#fff` / `#000` 除外）
- 深色专业界面，单一强调色，8px 圆角体系，间距 4 的倍数
- 响应式最低支持到 1024px

## 9. 验收标准（EARS 格式）

| 编号 | 功能 | 验收标准 | 优先级 |
|---|---|---|---|
| AC-01 | 摄入 | When 提交长度超过切片阈值的文档，系统**必须**把它切成多于一个分片，且相邻分片存在重叠 | P0 |
| AC-02 | 检索 | While 知识库非空且查询非空，系统**必须**在 3 秒内返回带 rank 的命中列表 | P0 |
| AC-03 | 混合检索 | If 指定 `mode=bm25`，系统**必须**只返回稀疏通道分数，`vector_score` 为空 | P0 |
| AC-04 | 生成 | While 模型返回带 `[k]` 标注的答案，系统**必须**把标注映射到对应的第 k 条命中作为引用 | P0 |
| AC-05 | 流式 | When 请求流式问答，系统**必须**按 `event:`/`data:` 格式输出 token 与 done 帧，且 done 帧含 session_id | P0 |
| AC-06 | 降级 | If 主模型不可用，系统**必须**自动降级到备用 provider 并在答案前标记降级 | P0 |
| AC-07 | 重排守卫 | While 查询以中文为主且重排器不支持中文，系统**必须**旁路重排并记录降级原因 | P1 |
| AC-08 | Agent | While 模型输出不含可解析的工具调用，系统**必须**把该输出作为最终答案返回而不抛异常 | P1 |
| AC-09 | Agent | When 循环步数达到上限，系统**必须**停止并返回当前最佳结论 | P1 |
| AC-10 | 工具安全 | If 表达式含黑名单 AST 节点，系统**必须**拒绝求值并返回失败原因 | P1 |
| AC-11 | 追踪 | While 完成一次问答，系统**必须**记录包含检索与生成 span 的追踪树 | P1 |
| AC-12 | 评测 | When 运行回归评测，recall@5、mrr@5、ndcg@5 **必须**不低于 `thresholds.json` 中的阈值 | P1 |
| AC-13 | 错误流 | If 请求不存在的文档删除，系统**必须**返回 404 而非 500 | P0 |
| AC-14 | 可复现 | While 在干净环境执行 `pip install -r requirements.txt`，安装**必须**无需编译工具链 | P0 |

## 10. 边界与约束

- 不支持 IE；浏览器最低支持现代 Chromium / Firefox / Safari 当前版本
- 单机部署，向量索引常驻内存（可选落盘）
- 单文档单次摄入上限 20 万字符，单次查询上限 4000 字符
- 默认不启用重排；Agent 默认不开启，需显式参数打开

## 11. 已知坑与规避

| 坑 | 规避方式 |
|---|---|
| 跨语言错配的 cross-encoder 让排序变差 | `GuardedReranker` 语言守卫自动降级 |
| 测试数据短于切片阈值，导致多分片路径从未被覆盖 | 语料与 E2E 数据显式超过阈值 |
| 同一文档多个分片在 nDCG 中被重复计分 | 指标计算前按文档 id 保序去重 |
| 归一化分数随语料漂移，加权融合不稳定 | 改用 RRF，只用名次信息 |
| 重解析工具调用失败导致整轮崩溃 | 解析失败即视为最终答案，优雅退化 |
| embedding 是 CPU 密集，阻塞事件循环 | `asyncio.to_thread` 卸载 |

## 12. 端到端验证步骤（权威）

```bash
pip install -r requirements.txt

# 1. 离线冒烟（不需要任何外部服务）
python scripts/smoke.py

# 2. 单元测试
pytest -q

# 3. 真实进程端到端
python scripts/e2e.py --provider mock

# 4. 真实本地模型端到端
ollama serve
ollama pull qwen2.5:1.5b-instruct
python scripts/e2e.py --provider ollama

# 5. 回归评测门禁
python scripts/eval.py

# 6. 一键全量
python scripts/verify.py
```

## 13. 变更记录

| 日期 | 变更 | 原因 | 影响范围 |
|---|---|---|---|
| 2026-09-22 | 初版 Spec | 项目建立 | 全部 |
| 2026-09-22 | 新增 Ollama 稠密嵌入通道 | 哈希嵌入对语义改写召回不足 | `embed_provider` 配置与装配 |
