"""nexus-ai-platform - 端到端可插拔 AI 系统核心库。

设计原则：
    端口与适配器。所有外部依赖（大模型 / 向量库 / 重排模型 / 工具 / 存储）
    都被抽象为 Protocol，运行时由容器注入具体实现。
    每个模块可以脱离网络、脱离 API Key、单独跑测试。

作者: 晨星
"""

from nexus.version import __author__, __version__

__all__ = ["__author__", "__version__"]
