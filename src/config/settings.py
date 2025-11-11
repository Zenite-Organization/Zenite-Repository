from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    github_webhook_secret: str
    ia_api_url: str
    class Config:
        env_file = ".env"

settings = Settings()

