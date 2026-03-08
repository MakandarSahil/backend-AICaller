"""
app/core/config.py
All environment variables are read here via pydantic-settings.
Never import os.environ directly elsewhere — always use settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    ENV: str = "development"           # development | production
    SECRET_KEY: str = "changeme"       # override in .env / GitHub secret

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"
    # In Docker Compose the hostname is the service name "redis"
    # For local dev without Docker: redis://localhost:6379/0

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins
    # e.g. "https://dashboard.aicaller.in,https://aicaller.in"
    CORS_ORIGINS: List[str] = ["*"]

    # ── Supabase (server-side only — never expose to frontend) ────────────────
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # ── Twilio ────────────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""

    # ── Azure Speech ──────────────────────────────────────────────────────────
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = "eastus"

    # ── Groq LLM ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""

    # pydantic-settings will read from a .env file if present
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Single shared instance — import this everywhere
settings = Settings()