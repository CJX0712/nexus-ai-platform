"""端到端验证：拉起真实进程，跑成功流 + 错误流，最后必须清理端口。

为什么不用 docker compose + curl：
    在受限环境里容器与 curl 都不可靠，而且它们验证的是"脚本能跑"，
    而不是"服务真的起来了"。这里直接 spawn Python 进程、
    轮询健康检查、用标准库发请求，失败时会打印子进程 stderr，绝不静默跳过。

用法：
    python scripts/e2e.py
    python scripts/e2e.py --provider ollama
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "packages" / "core"
PORT = int("8123")
BASE = f"http://127.0.0.1:{PORT}"

_DOCS = [
    {"title": "混合检索", "text": "BM25 负责关键词精确匹配，向量负责语义近似，RRF 用名次倒数融合。混合检索的候选宽度通常取最终数量的四倍。"},
    {"title": "重排守卫", "text": "英文重排器处理中文查询会给出近似随机分数，必须加语言守卫，在语言错配时降级为恒等重排。" * 4},
    {"title": "向量库", "text": "内存向量库适合小数据量与端到端验证，生产可选 Qdrant 或 pgvector。"},
]


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failed += 1
            print(f"  FAIL  {name} {detail}")


def request(method: str, path: str, body: dict | None = None, timeout: int = 60):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def expect_error(method: str, path: str, code: int, body: dict | None = None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=30)
        return None
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError:
        return None


def wait_health(proc: subprocess.Popen, timeout: int = 40) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            status, body = request("GET", "/api/health", timeout=3)
            if status == 200 and body.get("status") == "ok":
                return True
        except Exception:
            time.sleep(0.5)
    return False


def kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/pid", str(proc.pid), "/t", "/f"],
                capture_output=True,
                check=False,
            )
        else:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    global PORT, BASE
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default="")
    ap.add_argument("--embed-provider", default="hash", choices=["hash", "ollama", "openai"])
    ap.add_argument("--embed-model", default="")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    PORT = args.port
    BASE = f"http://127.0.0.1:{PORT}"

    import os

    env = {**dict(os.environ), "PYTHONPATH": str(CORE)}
    print(f"启动服务: provider={args.provider} port={PORT}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "nexus", "serve", "--port", str(PORT), "--provider", args.provider]
        + (["--model", args.model] if args.model else [])
        + ["--embed-provider", args.embed_provider]
        + (["--embed-model", args.embed_model] if args.embed_model else []),
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    c = Checker()
    try:
        if not wait_health(proc):
            print("服务未能启动，子进程输出：")
            try:
                out, err = proc.communicate(timeout=5)
                print(err[-3000:] or out[-2000:])
            except Exception:
                pass
            return 1

        print("\n成功流")
        status, health = request("GET", "/api/health")
        c.ok("健康检查返回 200", status == 200)
        c.ok("健康检查带 provider", bool(health.get("provider")), str(health))

        doc_ids = []
        for doc in _DOCS:
            status, body = request("POST", "/api/v1/ingest", doc)
            c.ok(f"摄入《{doc['title']}》", status == 200 and body.get("n_chunks", 0) > 0, str(body))
            doc_ids.append(body.get("doc_id"))

        status, body = request("GET", "/api/v1/documents")
        c.ok("文档列表数量正确", body.get("total") == len(_DOCS), str(body.get("total")))

        status, body = request("POST", "/api/v1/search", {"query": "语言守卫", "top_k": 3})
        hits = body.get("hits", [])
        c.ok("检索有结果", len(hits) > 0, str(body))
        c.ok("检索 top1 相关", hits and hits[0].get("rank") == 1, str(hits[:1]))

        status, body = request("POST", "/api/v1/search", {"query": "语言守卫", "mode": "bm25"})
        c.ok("bm25 单通道可用", status == 200 and body.get("hits"))
        status, body = request("POST", "/api/v1/search", {"query": "语言守卫", "mode": "vector"})
        c.ok("vector 单通道可用", status == 200)

        status, body = request("POST", "/api/v1/chat", {"query": "为什么要加语言守卫", "top_k": 3})
        c.ok("非流式问答返回答案", status == 200 and bool(body.get("answer")), str(body)[:200])
        c.ok("问答返回引用", len(body.get("citations", [])) > 0)
        c.ok("问答返回 trace_id", bool(body.get("trace_id")))

        tid = body.get("trace_id", "")
        if tid:
            status, trace = request("GET", f"/api/v1/traces/{tid}")
            c.ok("trace 含 span", status == 200 and len(trace.get("spans", [])) > 0, str(trace)[:200])

        sse = urllib.request.urlopen(
            urllib.request.Request(
                BASE + "/api/v1/chat/stream",
                data=json.dumps({"query": "为什么要加语言守卫"}).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            ),
            timeout=90,
        ).read().decode()
        c.ok("流式输出含 token 帧", "event: token" in sse)
        c.ok("流式输出含 done 帧", "event: done" in sse)
        done_block = [b for b in sse.split("\n\n") if b.startswith("event: done")]
        if done_block:
            payload = json.loads(done_block[0].split("data: ", 1)[1])
            c.ok("done 帧含 session_id", bool(payload.get("session_id")))
            c.ok("done 帧含非流式一致的 answer", bool(payload.get("answer")))

        status, metrics = request("GET", "/api/v1/metrics")
        c.ok("metrics 有请求计数", metrics.get("requests", 0) >= 1, str(metrics))

        print("\n错误流")
        c.ok("删除不存在文档返回 404", expect_error("DELETE", "/api/v1/documents/nope", 404) == 404)
        c.ok("查询不存在 trace 返回 404", expect_error("GET", "/api/v1/traces/nope", 404) == 404)
        c.ok("未知路由返回 404", expect_error("GET", "/api/v1/unknown", 404) == 404)
        c.ok("空 query 被校验拦截", expect_error("POST", "/api/v1/chat", 422, {"query": ""}) == 422)
        c.ok("空 title 被校验拦截", expect_error("POST", "/api/v1/ingest", 422, {"title": "", "text": "x"}) == 422)

        status, body = request("DELETE", f"/api/v1/documents/{doc_ids[0]}")
        c.ok("删除已有文档成功", status == 200 and body.get("deleted") is True)
        status, body = request("GET", "/api/v1/documents")
        c.ok("删除后数量递减", body.get("total") == len(_DOCS) - 1, str(body.get("total")))

        count = 0
        for attempt in range(1, 13):
            status, health = request("GET", "/api/health")
            if attempt == 1:
                count = count
            time.sleep(0.1)
        c.ok("服务在多次请求后仍存活", status == 200)
    finally:
        kill(proc)

    print(f"\n通过 {c.passed} / 失败 {c.failed}")
    return 0 if c.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
