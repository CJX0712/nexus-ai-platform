"""命令行入口：把整个系统作为一个进程启动。

用法：
    python -m nexus serve --port 8000
    python -m nexus serve --provider ollama --model qwen2.5:1.5b-instruct
    python -m nexus serve --provider mock          # 离线演示
"""

from __future__ import annotations

import argparse
import sys

from nexus.config import Settings
from nexus.container import build_system
from nexus.gateway.app import create_app


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nexus", description="nexus-ai-platform 服务")
    sub = ap.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--provider", default="ollama", choices=["ollama", "openai", "mock", "auto"])
    serve.add_argument("--model", default="")
    serve.add_argument("--rag-mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    serve.add_argument(
        "--embed-provider",
        default="hash",
        choices=["hash", "ollama", "openai"],
        help="hash=零依赖离线嵌入；ollama=本地稠密嵌入（推荐 bge-m3）",
    )
    serve.add_argument("--embed-model", default="", help="嵌入模型名，如 bge-m3")
    serve.add_argument("--rerank", default="off", choices=["off", "coverage", "cross"])
    serve.add_argument("--top-k", type=int, default=5)
    serve.add_argument("--data-dir", default="./data")
    serve.add_argument("--persist", action="store_true", help="把索引持久化到 data 目录")
    serve.add_argument("--reload", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd != "serve":
        build_parser().print_help()
        return 2

    s = Settings(
        host=args.host,
        port=args.port,
        provider=args.provider,
        rag_mode=args.rag_mode,
        rerank=args.rerank,
        top_k=args.top_k,
        data_dir=args.data_dir,
        persist=args.persist,
    )
    if args.model:
        if args.provider == "ollama":
            s.ollama_model = args.model
        else:
            s.openai_model = args.model
    if args.embed_provider:
        s.embed_provider = args.embed_provider
    if args.embed_model:
        s.ollama_embed_model = args.embed_model

    import uvicorn

    system = build_system(s)
    app = create_app(s, system)
    uvicorn.run(app, host=s.host, port=s.port, log_level=s.log_level.lower(), reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
