from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    WEBHOOK_SECRET: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    APP_ID: Optional[str] = None
    APP_PRIVATE_KEY: Optional[str] = None  # PEM string; alternatively set path and load it in code
    APP_PRIVATE_KEY_path: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()

