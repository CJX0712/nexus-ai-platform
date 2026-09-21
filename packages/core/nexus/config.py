"""集中式配置。所有可调参数都在这里，全部可用环境变量覆盖（前缀 NEXUS_）。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from nexus.types import RetrieveMode


class Settings(BaseSettings):
    """运行配置。

    优先级：环境变量 NEXUS_*  > .env 文件 > 默认值
    """

    model_config = SettingsConfigDict(env_prefix="NEXUS_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1

    provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b-instruct"
    ollama_keep_alive: str = "10m"
    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_model: str = "deepseek-chat"
    openai_api_key: str = ""

    embed_provider: str = "hash"
    ollama_embed_model: str = "bge-m3"
    openai_embed_model: str = "text-embedding-3-small"
    embed_dim: int = 384
    top_k: int = 5
    chunk_size: int = 420
    chunk_overlap: int = 80
    rag_mode: RetrieveMode = "hybrid"
    rerank: str = "off"
    agent_max_steps: int = 4
    auto_agent: bool = False

    data_dir: str = "./data"
    persist: bool = False
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"


DEFAULT_SETTINGS = Settings()
