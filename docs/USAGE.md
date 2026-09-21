# 使用指南

## 1. 命令行

```bash
python -m nexus serve [选项]
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 监听地址，容器里用 `0.0.0.0` |
| `--port` | `8000` | 端口 |
| `--provider` | `ollama` | `ollama` / `openai` / `mock` / `auto` |
| `--model` | 由 provider 决定 | 生成模型名 |
| `--embed-provider` | `hash` | `hash` 离线；`ollama` 稠密嵌入 |
| `--embed-model` | `bge-m3` | 嵌入模型名 |
| `--rag-mode` | `hybrid` | `hybrid` / `vector` / `bm25` |
| `--rerank` | `off` | `off` / `coverage` / `cross` |
| `--top-k` | `5` | 召回条数 |
| `--persist` | 关闭 | 开启后索引落盘到 `data/` |

常用组合：

```bash
# 离线演示（不连任何外部服务）
python -m nexus serve --provider mock

# 本地真实模型 + 真实稠密嵌入（推荐）
python -m nexus serve --provider ollama --embed-provider ollama

# 只用关键词通道，排查语义漂移时用
python -m nexus serve --rag-mode bm25

# 云端模型（OpenAI 兼容协议）
NEXUS_OPENAI_API_KEY=sk-xxx \
python -m nexus serve --provider openai --model deepseek-chat
```

环境变量与选项等价，前缀 `NEXUS_`，例如 `NEXUS_TOP_K=10`。完整清单见 [README](../README.md#配置)。

---

## 2. HTTP API

### 摄入文档

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"title":"混合检索","text":"BM25 负责关键词，向量负责语义，RRF 负责融合。"}'
```

```json
{"doc_id":"a1b2c3","n_chunks":1,"n_chars":38}
```

### 检索

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"混合检索怎么融合","top_k":3,"mode":"hybrid"}'
```

返回的每条命中都带 `bm25_score` 与 `vector_score`，可以直接看出两路通道各自的贡献。

### 问答（非流式）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"为什么要做混合检索","top_k":3}'
```

```json
{
  "session_id": "9f2a...",
  "answer": "混合检索把稀疏与稠密结合...",
  "citations": [{"chunk_id":"a1b2c3#0","doc_id":"a1b2c3","text":"...","score":0.98,"rank":1}],
  "tool_calls": [],
  "trace_id": "3c1d...",
  "usage": {"prompt_tokens":120,"completion_tokens":64,"total_ms":820.5,"ttft_ms":310.2}
}
```

### 问答（流式 SSE）

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"为什么要做混合检索"}'
```

```
event: citation
data: {"chunk_id":"a1b2c3#0","text":"...","score":0.98,"rank":1}

event: token
data: {"delta":"混"}

event: done
data: {"answer":"...","session_id":"9f2a...","trace_id":"3c1d...","usage":{...}}
```

前端用 EventSource 消费即可；注意不要被反向代理缓冲（服务已设置 `X-Accel-Buffering: no`）。

### Agent（工具调用）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"现在是几点，顺便算一下 128*37","use_agent":true}'
```

内置工具：`calculator`（受限表达式求值）、`now`、`text_stats`、`kb_search`（把检索作为工具）。

### 追踪

```bash
curl http://127.0.0.1:8000/api/v1/traces
curl http://127.0.0.1:8000/api/v1/traces/<trace_id>
```

返回 span 树，含父子关系、相对起点的毫秒偏移、耗时与自定义属性。

### 指标

```bash
curl http://127.0.0.1:8000/api/v1/metrics
```

---

## 3. Web 控制台

```bash
cd packages/web
npm install
npm run dev        # http://localhost:5173，已配置 /api 代理到 8000
```

三个面板：

1. **对话**：流式输出、引用来源可展开、工具调用折叠展示、每条回答显示延迟与 token
2. **知识库**：粘贴文本摄入、文档列表与删除、检索测试可直观对比 bm25 与向量两路原始分
3. **Trace**：追踪列表 + 火焰图风格 span 瀑布，用于定位性能瓶颈

---

## 4. 评测

```bash
python scripts/eval.py                                   # 离线（哈希嵌入 + Mock）
python scripts/eval.py --embed-provider ollama           # 真实嵌入
python scripts/eval.py --provider ollama --embed-provider ollama   # 真实生成 + 真实嵌入
python scripts/eval.py --mode bm25                       # 只测稀疏通道
python scripts/eval.py --json                            # 机器可读输出
```

指标含义：

| 指标 | 回答的问题 |
|---|---|
| `recall@5` | 前 5 条里有没有找回必须的文档 |
| `mrr@5` | 第一个正确答案排在第几位 |
| `ndcg@5` | 名次质量（考虑排序位置） |
| `faithfulness` | 答案里的字符二元组有多少能在证据中找到（越高越不容易胡说） |
| `citation_precision` | 引用的证据里有多少是真正相关的 |

自定义评测：替换 `eval/datasets/corpus.jsonl` 与 `queries.jsonl`，改 `thresholds.json` 调门槛。

---

## 5. 常见问题

**模型不可用 / 连接被拒绝**：服务会自动降级到 Mock 并在答案前标注 `[模型降级]`。检查 `ollama serve` 是否已启动、`ollama list` 里有没有对应模型。

**检索结果不理想**：先看是不是该走 `mode=bm25`（专有名词多时稀疏通道更强）；
语义改写多的场景切 `--embed-provider ollama`（bge-m3）；
最后才考虑开 `--rerank coverage`（务必确认查询语言与重排模型匹配）。

**改端口**：`python -m nexus serve --port 9000`，控制台同步改 vite 代理配置。

作者：晨星
