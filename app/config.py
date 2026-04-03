from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # environemt
    env: str = "development"

    # server
    port: str = "8000"
    public_url: str = "https://determination-barbara-strongly-adjustments.trycloudflare.com"
    secret_key: str = "secret"
    cors_origins: str = "*"
    log_file: str = "/app/logs/backend.log"
    log_max_bytes: int = 10_485_760  # 10 MB
    log_backup_count: int = 5

    # supabase
    supabase_url: str
    supabase_service_role_key: str

    # redis
    redis_url: str = "redis://redis:6379/0"

    # twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # azure
    azure_speech_key: str
    azure_speech_region: str

    # groq
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.7

    # STT
    stt_language: str = "en-US"

    # TTS
    tts_default_voice: str = "en-IN-PrabhatNeural"

    # barge-in
    barge_in_threshold: int = 500
    barge_in_min_chunks: int = 15

    # Caching (Redis TTLs in seconds)
    cache_ttl_agent: int = 300             # 5 min — agent config + KB
    cache_ttl_kb: int = 300               # 5 min
    cache_ttl_caller_history: int = 1800  # 30 min
    cache_ttl_auth: int = 300             # 5 min — API key + JWT workspace lookups

    # LLM promt limits
    kb_max_chars: int = 60_000
    caller_history_max_turns: int = 10

    # Celery
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

    # Helpers

    @property
    def is_production(self) -> bool:
        return self.env == "production"
 
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
 
    @property
    def ws_call_url(self) -> str:
        """Full WSS URL for Twilio stream — used in TwiML response."""
        base = self.public_url.rstrip("/")
        return base.replace("https://", "wss://").replace("http://", "ws://") + "/call"
 
 
@lru_cache
def get_settings() -> Settings:
    return Settings()