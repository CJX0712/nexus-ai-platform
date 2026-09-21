# ADR-001: 采用端口与适配器架构，所有外部依赖抽象为 Protocol

## Status

Accepted (2026-09-22)

## Background

AI 系统的外部依赖非常多且易变：LLM 厂商每周出新模型、向量库选型频繁变更、
嵌入服务可能被替换。如果业务代码直接依赖具体 SDK，任何一次替换都涉及大面积改动，
而且单元测试必须真的连上外部服务，导致 CI 里根本跑不了。

## Decision

1. 所有外部能力定义为 `Protocol`，集中在 `nexus/protocols.py`
2. 业务模块只依赖接口，不 import 具体实现
3. 具体实现（Mock / Ollama / OpenAI / InMemory / BM25 / CrossEncoder）在 `container.py` 装配注入
4. 每个外部依赖都提供一个零依赖实现，保证离线可跑

## Consequences

正面：
- 单测与 E2E 在无网络、无 Key、无数据库下全绿
- 替换任意一层只需改组合根一处
- 模块边界清晰，改动影响面可预测

负面：
- 多了一层接口定义，初次阅读需要理解协议
- 需要显式装配，比直接 `import chromadb` 多几行代码

## Related ADRs

ADR-005, ADR-006
