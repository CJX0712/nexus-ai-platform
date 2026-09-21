"""FastAPI 应用工厂。

入口文件只做装配，不含业务逻辑。
"""

from __future__ import annotations

import logging

import orjson
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from nexus.config import Settings
from nexus.container import System, build_system
from nexus.gateway.routes import register_routes
from nexus.gateway.sse import sse_frame
from nexus.version import __author__, __version__


class ORJSONResponse(Response):
    """用 orjson 替代标准库 json，序列化吞吐提升约 2~3 倍。"""

    media_type = "application/json"

    def render(self, content: object) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)


def create_app(settings: Settings | None = None, system: System | None = None) -> FastAPI:
    s = settings or Settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    app = FastAPI(
        title="nexus-ai-platform",
        version=__version__,
        description="端到端可插拔 AI 系统：RAG + Agent + 混合检索 + 评测 + 可观测性",
        contact={"name": __author__},
        license_info={"name": "MIT"},
        default_response_class=ORJSONResponse,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    sys_ = system or build_system(s)
    register_routes(app, sys_)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logging.exception("unhandled error: %s", exc)
        return ORJSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}", "path": request.url.path},
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        close = getattr(sys_.llm, "aclose", None)
        if close:
            await close()

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "name": "nexus-ai-platform",
            "version": __version__,
            "author": __author__,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


async def _error_sse(message: str):  # pragma: no cover - 供调试使用
    yield sse_frame("error", {"message": message})
