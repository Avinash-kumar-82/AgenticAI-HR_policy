from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/src/backend/core/config.py
# parents[0] -> backend/src/backend/core
# parents[1] -> backend/src/backend
# parents[2] -> backend/src
# parents[3] -> backend
BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_ADMIN_KEY = "change-me-in-production"


class Settings(BaseSettings):
    # Application
    app_name: str = "Enterprise HR Policy Agentic RAG Copilot"
    app_env: str = "development"

    # LLM and web search providers
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # Vector database
    pinecone_api_key: str = ""
    pinecone_index_name: str = "fde-hr-policy-rag"
    pinecone_namespace: str = "company-hr-kb"

    # Embeddings and LLM
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # If you see structured-output warnings or odd grades, try
    # "llama-3.3-70b-versatile" instead.
    groq_model: str = "openai/gpt-oss-20b"

    # RAG and retry configuration
    top_k: int = 5
    max_retries: int = 1

    # Admin access
    admin_api_key: str = DEFAULT_ADMIN_KEY

    # Application directories
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    if settings.app_env != "development" and settings.admin_api_key == DEFAULT_ADMIN_KEY:
        raise RuntimeError(
            "Set ADMIN_API_KEY in the environment before running outside development."
        )

    return settings