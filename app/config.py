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
    public_url: str = "https://api.iamspiderman.me"
    ws_call_url_override: str | None = None
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
    
    # encryption for credential storage (BYO Twilio, etc.)
    # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: str = ""

    # azure
    azure_speech_key: str
    azure_speech_region: str

    # groq
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 512
    groq_temperature: float = 0.7

    # STT
    stt_language: str = "en-US"

    # TTS
    tts_default_voice: str = "en-IN-PrabhatNeural"

    # barge-in
    # Threshold is on PCM16 amplitude scale (0-32767).
    # TTS echo through phone lines typically measures 1000-3000.
    # Lower threshold for more responsive interruption detection.
    barge_in_threshold: int = 3500
    barge_in_min_chunks: int = 3

    # Caching (Redis TTLs in seconds)
    cache_ttl_agent: int = 3600           # 1 hour — agent config rarely changes
    cache_ttl_kb: int = 3600             # 1 hour
    cache_ttl_caller_history: int = 3600  # 1 hour
    cache_ttl_auth: int = 3600           # 1 hour — API key + JWT workspace lookups

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
        if self.ws_call_url_override:
            return self.ws_call_url_override.strip()
        base = self.public_url.rstrip("/")
        return base.replace("https://", "wss://").replace("http://", "ws://") + "/call"
 
 
@lru_cache
def get_settings() -> Settings:
    return Settings()