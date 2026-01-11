from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    WEBHOOK_SECRET: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    APP_ID: Optional[str] = None
    APP_PRIVATE_KEY: Optional[str] = None  # PEM string; alternatively set path and load it in code
    APP_PRIVATE_KEY_path: Optional[str] = None
    # Sprint / planning settings
    WORK_HOURS_PER_DAY: int = 8
    SPRINT_DEFAULT_DAYS: int = 14
    SPRINT_CAPACITY_HOURS: Optional[int] = None
    PLANNING_TRIGGER_LABEL: str = "planning"
    SPRINT_BACKLOG_LABEL: str = "Backlog"

    class Config:
        env_file = ".env"


settings = Settings()

