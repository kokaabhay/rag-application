from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # For the Application
    app_name: str = "Customer Support RAG"

    # This is the LLM configuration
    llm_api_key: str
    llm_base_url: str
    llm_model: str

    model_config = SettingsConfigDict(
        env_file=".env",        
        extra="ignore",
        env_file_encoding="utf-8",
    )


settings = Settings()