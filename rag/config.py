from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


def _safe_choice(value: Optional[str], allowed: set[str], default: str) -> str:
    if not value:
        return default
    value = str(value).strip().lower()
    return value if value in allowed else default


def _resolve_path(value: Optional[str], default: str) -> str:
    raw = Path(value or default).expanduser()
    return str(raw if raw.is_absolute() else (Path.cwd() / raw).resolve())


class RAGConfig(BaseSettings):
    app_name: str = "RAG Project"
    app_host: str = os.environ.get("APP_HOST", "127.0.0.1")
    app_port: int = int(os.environ.get("APP_PORT", 8010))

    vectorstore_type: str = _safe_choice(os.environ.get("VECTORSTORE_TYPE"), {"milvus", "chroma"}, "chroma")

    milvus_host: str = os.environ.get("MILVUS_HOST", "127.0.0.1")
    milvus_port: int = int(os.environ.get("MILVUS_PORT", 19530))
    milvus_db_name: str = os.environ.get("MILVUS_DB_NAME", "default")
    milvus_collection: str = os.environ.get("MILVUS_COLLECTION", "rag_docs")
    milvus_user: Optional[str] = os.environ.get("MILVUS_USER")
    milvus_password: Optional[str] = os.environ.get("MILVUS_PASSWORD")

    chroma_path: str = _resolve_path(os.environ.get("CHROMA_PATH"), "./chroma_db_rag")
    chroma_collection: str = os.environ.get("CHROMA_COLLECTION", "rag_docs")

    embedding_model_path: str = _resolve_path(os.environ.get("EMBEDDING_MODEL_PATH"), "./models/bge-small-zh-v1.5")
    embedding_device: str = os.environ.get("EMBEDDING_DEVICE", "cpu")

    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    openai_chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "deepseekv3_1")

    max_upload_files: int = int(os.environ.get("MAX_UPLOAD_FILES", 5))
    chunk_size: int = max(100, int(os.environ.get("CHUNK_SIZE", 900)))
    chunk_overlap: int = max(0, int(os.environ.get("CHUNK_OVERLAP", 180)))
    coarse_chunk_size: int = max(100, int(os.environ.get("COARSE_CHUNK_SIZE", 1500)))
    coarse_chunk_overlap: int = max(0, int(os.environ.get("COARSE_CHUNK_OVERLAP", 100)))
    max_upload_file_mb: int = max(1, int(os.environ.get("MAX_UPLOAD_FILE_MB", 30)))

    top_k: int = max(1, int(os.environ.get("TOP_K", 10)))
    hybrid_fetch_k: int = max(1, int(os.environ.get("HYBRID_FETCH_K", 40)))
    hybrid_alpha: float = max(0.0, min(1.0, float(os.environ.get("HYBRID_ALPHA", 0.7))))
    freshness_weight: float = max(0.0, min(1.0, float(os.environ.get("FRESHNESS_WEIGHT", 0.25))))
    freshness_trigger_similarity: float = max(0.0, min(1.0, float(os.environ.get("FRESHNESS_TRIGGER_SIMILARITY", 0.55))))
    rewrite_enabled: bool = os.environ.get("REWRITE_ENABLED", "true").lower() == "true"

    auth_mode: str = _safe_choice(os.environ.get("AUTH_MODE"), {"none", "api_key"}, "none")
    api_keys: str = os.environ.get("API_KEYS", "")

    chat_history_dir: str = _resolve_path(os.environ.get("CHAT_HISTORY_DIR"), "./chat_history_rag")
    chat_history_max_turns: int = max(1, int(os.environ.get("CHAT_HISTORY_MAX_TURNS", 20)))

    upload_dir: str = _resolve_path(os.environ.get("UPLOAD_DIR"), "./rag_uploads")
    metadata_db_path: str = _resolve_path(os.environ.get("METADATA_DB_PATH"), "./rag_metadata")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
