from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    WEBHOOK_SECRET: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    github_app_id: Optional[str] = None
    github_app_private_key: Optional[str] = None  # PEM string; alternatively set path and load it in code
    github_app_private_key_path: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()

