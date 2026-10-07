"""Settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

# The repository root (knowledge/, eval/, data/). Override with ASKDATA_HOME if the package is installed elsewhere.
PROJECT_ROOT = Path(os.getenv("ASKDATA_HOME") or Path(__file__).resolve().parents[2])
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
EVAL_DIR = PROJECT_ROOT / "eval"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    "anthropic": "claude-opus-5-5",
    "oracle": "gold-sql",
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    llm_provider: str  # ollama (default, local) | anthropic | oracle (gold SQL, for testing)
    llm_model: str
    ollama_base_url: str
    anthropic_effort: str
    embeddings: str  # tfidf | bge-m3 (via Ollama)
    embedding_model: str
    few_shot_k: int
    max_repairs: int
    query_timeout_s: float
    max_rows: int
    min_group_size: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clinic.duckdb"

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / "parquet"

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env("LLM_PROVIDER", "ollama").lower()
        data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        return cls(
            data_dir=data_dir,
            llm_provider=provider,
            llm_model=_env("LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            anthropic_effort=_env("ANTHROPIC_EFFORT", "medium"),
            embeddings=_env("EMBEDDINGS", "tfidf").lower(),
            embedding_model=_env("EMBEDDING_MODEL", "bge-m3"),
            few_shot_k=int(_env("FEW_SHOT_K", "4")),
            max_repairs=int(_env("MAX_REPAIRS", "2")),
            query_timeout_s=float(_env("QUERY_TIMEOUT_S", "10")),
            max_rows=int(_env("MAX_ROWS", "500")),
            min_group_size=int(_env("MIN_GROUP_SIZE", "10")),
        )

    def with_overrides(self, **changes) -> "Settings":
        if "llm_provider" in changes and "llm_model" not in changes:
            changes["llm_model"] = DEFAULT_MODELS.get(changes["llm_provider"], self.llm_model)
        return replace(self, **changes)
