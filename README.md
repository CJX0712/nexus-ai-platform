# nexus-ai-platform

端到端可插拔 AI 系统：**RAG 检索增强生成 + Tool-Calling Agent + 混合检索 + 重排守卫 + 回归评测 + 链路追踪 + Web 控制台**。

不是 demo，是一套能在干净环境里一键复现、每个模块都能独立验证、且可以真的用本地模型跑起来完整链路的系统。核心依赖只有 9 个纯 Wheel 包，全部零编译。

作者：**晨星** · 协议：MIT

---

## 为什么这样设计

| 常见做法 | 本项目做法 | 原因 |
|---|---|---|
| 直接调 Embedding API、直接写 `chromadb.query()` | 所有外部依赖抽象成 Protocol，运行时注入 | 单元测试可以用 fake 跑，无需网络与 Key |
| 只用向量检索 | BM25 + 向量 + RRF 融合 | 专有名词、型号、参数这类必须逐字命中的查询，纯向量会漏 |
| 重排一律启用 | 带**语言守卫**的重排 | 英文 CrossEncoder 处理中文查询，top-1 命中会从 11/12 掉到 4/12 |
| 单例脚本拼凑 | 端口与适配器，组合根唯一 | 换向量库 / 换模型只改 `container.py` 一处 |
| "测试通过"就发货 | 真实进程 E2E + 评测门禁 | 单测全绿但服务起不来是最常见的交付风险 |

---

## 架构

```
Web 控制台 (React 19 + TS)          REST / SSE
        |
   fastapi gateway  ->  编排 Orchestrator
                             |
   +------------+------------+------------+------------+
   |            |            |            |            |
LLM 路由    混合检索      重排(守卫)    工具执行      记忆
(Mock/Ollama  BM25+向量    覆盖/CE     计算/时间/   滑窗+摘要
 /OpenAI/vLLM)  RRF 融合              知识库检索
                             |
                    评测 Eval + 追踪 Tracer（横切）
```

分层与职责详见 [ARCHITECTURE.md](ARCHITECTURE.md)，完整契约见 [docs/SPEC.md](docs/SPEC.md)。

---

## 快速开始

```bash
git clone https://github.com/CJX0712/nexus-ai-platform.git
cd nexus-ai-platform

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/smoke.py        # 离线自检：不需要任何外部服务
```

离线自检会摄入 3 篇文档、跑完整链路并断言 top-1 命中：

```
[ingest] 混合检索白皮书      chunks=4
[retrieve] query=为什么要给重排器加语言守卫？
  #1 score=0.0328 bm25=1.000 ...
[answer] [mock] ... 因此必须为重排器加上语言守卫 ...（来源 [1]）
PASS  耗时 19 ms  切片总数 12
```

### 用真实本地模型（推荐）

```bash
ollama serve                       # 另开终端
ollama pull qwen2.5:1.5b-instruct  # 生成模型
ollama pull bge-m3                 # 嵌入模型（可选，显著提升语义召回）

python -m nexus serve --provider ollama --embed-provider ollama
```

打开 http://127.0.0.1:8000/docs 看交互式 API，控制台在 `packages/web`。

模型不可用时会自动降级到内置 Mock 模型，链路不会中断。

---

## 一键验证

```bash
python scripts/verify.py           # pytest + e2e + eval + 静态检查 全量
python scripts/e2e.py              # 真实进程端到端：成功流 + 错误流
python scripts/eval.py             # 回归评测（默认离线 hash 嵌入）
python scripts/eval.py --embed-provider ollama   # 换成 bge-m3 稠密嵌入
```

预期结果：

| 命令 | 通过:失败 | 说明 |
|---|---|---|
| `pytest` | 142:0 | 覆盖文本/嵌入/向量库/BM25/RRF/守卫/工具/Agent/记忆/网关 |
| `scripts/e2e.py` | 27:0 | 真实进程，含 SSE 帧校验、错误流、端口清理 |
| `scripts/eval.py` | PASS | recall@5 1.000 / mrr@5 0.854 / ndcg@5 0.891 / p95 3ms |

评测阈值写在 `eval/datasets/thresholds.json`，可接进 CI 做回归门禁。

---

## 目录结构

```
packages/core/nexus/          后端（Python 3.13）
  protocols.py                全部接口契约，系统的"宪法"
  gateway/                    REST + SSE 入口（FastAPI，orjson 序列化）
  orchestrator/               会话编排
  rag/                        切片 / Prompt / 流水线
  retriever/                  增量 BM25 + RRF 融合检索
  reranker/                   恒等 / 覆盖率 / CrossEncoder + 语言守卫
  embedding/                  零依赖 TF-IDF 哈希 / Ollama 稠密嵌入
  vectorstore/                numpy 精确检索（协议对齐 Qdrant / pgvector）
  llm/                        Mock / Ollama / OpenAI 兼容 + 路由降级
  tools/ agent/ memory/       工具注册表 / ReAct 循环 / 滑窗摘要记忆
  observability/              span 追踪 + 指标
  eval/                       评测指标与运行器
packages/web/                 控制台（React 19 + TypeScript + lucide 图标）
eval/datasets/                固化语料 + 样例集 + 阈值
scripts/                      smoke / e2e / eval / verify
docs/                         SPEC / 部署 / 使用 / ADR 决策记录
```

---

## 配置

全部支持环境变量覆盖，前缀 `NEXUS_`，例如 `NEXUS_PROVIDER=ollama`。

| 变量 | 默认 | 说明 |
|---|---|---|
| `PROVIDER` | `ollama` | `ollama` / `openai` / `mock` / `auto` |
| `OLLAMA_MODEL` | `qwen2.5:1.5b-instruct` | 本地生成模型 |
| `EMBED_PROVIDER` | `hash` | `hash` 离线零依赖 / `ollama` 稠密嵌入 |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | 稠密嵌入模型，1024 维 |
| `RAG_MODE` | `hybrid` | `hybrid` / `vector` / `bm25` |
| `RERANK` | `off` | `off` / `coverage` / `cross` |
| `TOP_K` | `5` | 召回条数 |
| `PERSIST` | `false` | 是否把索引落盘到 `data/` |

更多见 [docs/USAGE.md](docs/USAGE.md)，生产部署见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

---

## 性能要点

- 全链路 async，CPU 密集的嵌入卸载到线程池，不阻塞事件循环
- 序列化用 orjson，比标准库 json 快 2~3 倍
- 嵌入批量处理 + 结果缓存，重复文本不重复计算
- 向量检索走 numpy/BLAS 矩阵乘，向量库块式扩容避免频繁重分配
- SSE 流式输出降低首字延迟；连接池复用 + 请求级超时
- 检索、重排、生成的耗时全部进 span，性能问题可定位到具体阶段

---

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 分层设计、数据流、扩展点
- [docs/SPEC.md](docs/SPEC.md) — 范围、API、数据契约、验收标准
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — 本地 / Docker / 生产部署
- [docs/USAGE.md](docs/USAGE.md) — CLI、API、控制台使用说明
- [docs/decisions/](docs/decisions) — ADR 架构决策记录

---

## 许可

MIT License. Copyright (c) 2026 晨星
