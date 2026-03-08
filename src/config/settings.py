import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GITHUB_WEBHOOK_SECRET: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    APP_ID: Optional[str] = None
    APP_PRIVATE_KEY: Optional[str] = None  # PEM string; alternatively set path and load it in code
    APP_PRIVATE_KEY_path: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None
    OPENAI_API_KEY_RAG: Optional[str] = None
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    RAG_TOPK_PER_NAMESPACE: int = 8
    RAG_MIN_HITS_MAIN: int = 3
    RAG_MIN_SCORE_MAIN: float = 0.6
    RAG_MAX_FALLBACK_BASES: int = 99
    RAG_FINAL_CONTEXT_SIZE: int = 10
    HEURISTIC_ENSEMBLE_RUNS: int = 4
    HEURISTIC_ENSEMBLE_TEMPERATURE: float = 0.3
    # Max number of parallel LLM calls for the heuristic ensemble.
    # If <= 0, it will default to HEURISTIC_ENSEMBLE_RUNS.
    HEURISTIC_ENSEMBLE_MAX_CONCURRENCY: int = 3
    # Sprint / planning settings
    WORK_HOURS_PER_DAY: int = 8
    SPRINT_DEFAULT_DAYS: int = 14
    PLANNING_TRIGGER_LABEL: str = "planning"
    SPRINT_BACKLOG_LABEL: str = "Backlog"

settings = Settings()

# Backward compatibility for existing env naming.
if not settings.PINECONE_API_KEY:
    settings.PINECONE_API_KEY = os.getenv("PINECONE_SECRET")
