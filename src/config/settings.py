from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # github_webhook_secret: str
    # ia_api_url: str
    GEMINI_API_KEY: Optional[str] = None
    # GitHub GraphQL / Projects configuration
    github_api_token: Optional[str] = None
    github_api_url: str = "https://api.github.com/graphql"
    github_project_v2_id: Optional[str] = None
    github_project_field_estimate_id: Optional[str] = None
    # GitHub App credentials (optional). If present use GitHub App flow to obtain installation tokens.
    github_app_id: Optional[str] = None
    github_app_private_key: Optional[str] = None  # PEM string; alternatively set path and load it in code
    github_app_private_key_path: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()

