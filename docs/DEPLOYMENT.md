# 部署指南

作者：晨星

## 1. 本地部署

### 1.1 仅用内置能力（零外部服务）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nexus serve --provider mock
python scripts/smoke.py
```

适合：CI、演示、功能验证。不需要网络与 GPU。

### 1.2 本地真实模型

```bash
# 安装 Ollama：https://ollama.com/download
ollama serve                       # 保持运行
ollama pull qwen2.5:1.5b-instruct  # 生成
ollama pull bge-m3                 # 嵌入（可选）

python -m nexus serve --provider ollama --embed-provider ollama
```

模型规模建议：

| 机器内存 | 推荐生成模型 | 说明 |
|---|---|---|
| ≤ 8 GB | qwen2.5:1.5b-instruct | 约 1 GB，响应快 |
| 16 GB | qwen2.5:7b-instruct-q4_K_M | 约 4.7 GB，质量明显更好 |
| ≥ 32 GB | qwen3:14b 或更大 | 复杂推理场景 |

---

## 2. Docker 部署

### 2.1 只跑应用（外接已有 Ollama）

```bash
docker build -t nexus-ai-platform .
docker run --rm -p 8000:8000 \
  -e NEXUS_PROVIDER=ollama \
  -e NEXUS_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  nexus-ai-platform
```

### 2.2 应用 + Ollama 一起起

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:1.5b-instruct
docker compose exec ollama ollama pull bge-m3
curl http://127.0.0.1:8000/api/health
```

`compose` 文件里 `nexus` 依赖 `ollama`，健康检查通过后才会开始服务。

---

## 3. 生产部署要点

### 3.1 进程与并发

```bash
python -m nexus serve --host 0.0.0.0 --port 8000
# 多实例时用进程管理器或容器编排横向扩容
```

- 单实例即可处理好中小流量；瓶颈通常在模型推理而非网关
- 向量索引常驻内存，多实例之间**不共享索引**：
  生产环境请开启 `--persist` 并把 `data/` 挂到共享存储，或替换 `VectorStore` 实现为 Qdrant / pgvector
- 建议前置反向代理做 TLS 与限流

### 3.2 健康检查与回滚

- 健康检查端点：`GET /api/health`
- Helm / K8s 探针直接指向该端点
- 回滚时只需把镜像 tag 退回上一个版本；配置全部走环境变量，改配置不需要重新构建

### 3.3 必须配置的反向代理行为

SSE 必须关闭代理缓冲，否则流式输出会变成一次性吐出：

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
}
```

### 3.4 安全

- 网关默认开放 CORS（`NEXUS_CORS_ORIGINS`），生产应限定来源
- 本仓库不含鉴权，鉴权应在网关或反向代理层实现
- 工具层已做沙箱：表达式求值走 AST 白名单，网络类工具默认不启用
- 不要在生产直接暴露 `/api/v1/metrics` 给公网

### 3.5 持久化与备份

```bash
python -m nexus serve --persist --data-dir /var/lib/nexus
```

索引以 npz + json 保存；备份这两个文件即可恢复知识库。
注意嵌入模型升级后向量维度可能变化，此时需要重新摄入。

---

## 4. 故障排查

| 现象 | 排查 |
|---|---|
| `/api/health` 里 provider 是 `mock` | 主模型不可用已降级；检查 Ollama 是否运行、模型是否拉取 |
| 答案变慢 | 看 `/api/v1/traces/<id>`，定位是检索、嵌入还是生成阶段 |
| 摄入后检索不到 | 确认 `embed_provider` 与索引构建时一致；维度不匹配会直接查不到 |
| 端口被占用 | E2E 脚本会强制 kill；手工场景下换 `--port` |
| Docker 里连不上宿主机 Ollama | 用 `host.docker.internal` 或 `network_mode: host` |

---

## 5. 环境变量速查

| 变量 | 默认 |
|---|---|
| `NEXUS_HOST` / `NEXUS_PORT` | `127.0.0.1` / `8000` |
| `NEXUS_PROVIDER` | `ollama` |
| `NEXUS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `NEXUS_OLLAMA_MODEL` | `qwen2.5:1.5b-instruct` |
| `NEXUS_EMBED_PROVIDER` | `hash` |
| `NEXUS_OLLAMA_EMBED_MODEL` | `bge-m3` |
| `NEXUS_RAG_MODE` | `hybrid` |
| `NEXUS_RERANK` | `off` |
| `NEXUS_TOP_K` | `5` |
| `NEXUS_PERSIST` | `false` |
| `NEXUS_DATA_DIR` | `./data` |
| `NEXUS_LOG_LEVEL` | `INFO` |
