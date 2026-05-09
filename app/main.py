import logging
import os
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.clients.redis import init_redis, close_redis
from app.clients.supabase import init_supabase, close_supabase
from app.config import get_settings
from app.routers import voice, query, api_keys, telephony
from app.ws.call_handler import handle_call

settings = get_settings()
START_TIME = time.time()

log_level = logging.DEBUG if not settings.is_production else logging.INFO
log_format = "%(asctime)s %(levelname)s %(name)s — %(message)s"

stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)
stream_handler.setFormatter(logging.Formatter(log_format))

handlers: list[logging.Handler] = [stream_handler]
try:
    log_dir = os.path.dirname(settings.log_file) or "."
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)
except Exception as exc:
    # Keep app booting even if file logger cannot be created.
    logging.getLogger(__name__).warning("File logger disabled: %s", exc)

logging.basicConfig(level=log_level, handlers=handlers, force=True)
logger = logging.getLogger(__name__)


# ── Tag metadata — shown as section headers + descriptions in Swagger UI ───────
_OPENAPI_TAGS = [
    {
        "name": "query",
        "description": (
            "**Core pipeline endpoint.** Send a text message to an agent and receive "
            "a streamed or single-response answer. Used by the dashboard chat UI and "
            "all external chatbot integrations. Auth required."
        ),
    },
    {
        "name": "api-keys",
        "description": (
            "**API key management.** Create, list, and revoke API keys for external "
            "chatbot integrations. JWT auth only — only workspace owners can manage keys."
        ),
    },
    {
        "name": "telephony",
        "description": (
            "**Twilio webhooks.** Called by Twilio automatically when a call arrives "
            "on a configured number. Not called by the dashboard or external developers."
        ),
    },
    {
        "name": "ops",
        "description": "Health check and service info endpoints.",
    },
]

## Authentication

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting aicaller-backend (env=%s)", settings.env)
    await init_redis()
    logger.info("Redis connected: %s", settings.redis_url)
    await init_supabase()
    logger.info("Supabase connected: %s", settings.supabase_url)
    yield
    logger.info("Shutting down...")
    await close_redis()
    await close_supabase()
    logger.info("Shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AICaller / CallMind API",
    version="1.0.0",
    description="""
Real-time AI voice agent and text pipeline backend.

## Authentication

Two auth schemes are supported. Use the **Authorize** button (top right) to set one.

### Supabase JWT — dashboard users
Copy your session token from the dashboard (browser devtools → Application → Local Storage → `sb-*-auth-token` → `access_token`).
Paste it into the **BearerAuth** field in Authorize.

### API Key — external integrations
Create a key via `POST /api-keys`. Keys look like `cm_live_xxxxxxxx...`.
Paste into the **ApiKeyAuth** field **or** the BearerAuth field — both work.

## Base URL
- **Production:** `https://api.callmind.com`
- **Development:** `http://localhost:8000`
""",
    # Always-on docs — endpoints are protected by auth anyway
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=_OPENAPI_TAGS,
    swagger_ui_parameters={
        "persistAuthorization": True,       # auth survives page reload
        "displayRequestDuration": True,     # shows latency of each request
        "filter": True,                     # search box at top of Swagger UI
        "syntaxHighlight.theme": "obsidian",
    },
    lifespan=lifespan,
)


# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(voice.router, tags=["telephony"])
app.include_router(telephony.router)
app.include_router(query.router, tags=["query"])
app.include_router(api_keys.router, tags=["api-keys"])


@app.get(
    "/health",
    tags=["ops"],
    summary="Health check",
    response_description="Service status and Redis connectivity",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "env": "production", "redis": "ok"}
                }
            },
        }
    },
)
async def health():
    """
    Returns service status and Redis connectivity.
    Used by Traefik, Docker HEALTHCHECK, and the CI/CD deploy gate.
    """
    from app.clients.redis import get_redis
    try:
        await get_redis().ping()
        redis_status = "ok"
    except Exception as exc:
        redis_status = f"error: {exc}"
    return {"status": "ok", "env": settings.env, "redis": redis_status, "message": "Hi i am sahil makandar you can reach me at sahilmakandar15@gmail.com"}

@app.get(
    "/",
    tags=["ops"],
    summary="Service info",
    response_description="Service name and version",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"service": "aicaller-backend", "version": "1.0.0"}
                }
            }
        }
    },
)
async def root():
    """Returns service name and version."""
    return {"service": "aicaller-backend", "version": "1.0.0", "status": "running"}



@app.get(
    "/health/info",
    tags=["ops"],
    summary="Service info (extended)",
    response_description="Service information and uptime",
)
async def health_info():
    """Returns extended service information used by diagnostics and tests."""
    return {
        "service": "aicaller-backend",
        "version": "1.0.0",
        "uptime_seconds": int(time.time() - START_TIME),
        "environment": settings.env,
    }


@app.get(
    "/health/ready",
    tags=["ops"],
    summary="Readiness check",
    response_description="Checks ready-state of downstream deps (Redis)",
)
async def health_ready():
    """Readiness endpoint that checks Redis connectivity.

    Tests inject `app.state.redis` with an AsyncMock; prefer that when present.
    """
    checks: dict[str, str] = {}

    redis_client = getattr(app.state, "redis", None)
    if redis_client is None:
        # Fall back to module-level Redis client if app.state.redis not present
        try:
            from app.clients.redis import get_redis

            redis_client = get_redis()
        except Exception as exc:  # not initialised
            checks["redis"] = f"error: {exc}"

    if redis_client:
        try:
            await redis_client.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"

    status = "ready" if all(v == "ok" for v in checks.values()) else "not_ready"
    return {"status": status, "checks": checks}


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/call")
async def ws_call(websocket: WebSocket, agent_id: str | None = None):
    """
    Twilio Media Stream WebSocket.
    URL: wss://api.callmind.com/call?agent_id={uuid}
    Opened automatically by Twilio after POST /voice returns TwiML.
    """
    await handle_call(websocket, agent_id)


# ── Custom OpenAPI schema — inject both security schemes ──────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        tags=_OPENAPI_TAGS,
        routes=app.routes,
    )

    # ── Security scheme definitions ───────────────────────────────────────
    schema.setdefault("components", {}).setdefault("securitySchemes", {})

    # Scheme 1 — Supabase JWT or API key via Authorization: Bearer header
    schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT or cm_live_xxx",
        "description": (
            "**Dashboard users:** paste your Supabase `access_token` (JWT).\n\n"
            "**External developers:** paste your API key (`cm_live_xxxxxxxx`).\n\n"
            "Both are accepted in this field."
        ),
    }

    # Scheme 2 — API key via X-API-Key header (alternative for external devs)
    schema["components"]["securitySchemes"]["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": (
            "Alternative to BearerAuth for external developers.\n\n"
            "Paste your API key (`cm_live_xxxxxxxx`) here.\n\n"
            "Use either BearerAuth **or** ApiKeyAuth — not both."
        ),
    }

    # ── Apply security to every path that needs auth ──────────────────────
    # Both schemes are listed as OR alternatives (Swagger shows either works)
    _AUTH_SECURITY = [{"BearerAuth": []}, {"ApiKeyAuth": []}]
    _JWT_ONLY_SECURITY = [{"BearerAuth": []}]

    protected_paths = {
        "/query": ["post"],
        "/query/tts-preview": ["post"],
        "/api-keys": ["get", "post"],
        "/api-keys/{key_id}": ["delete", "patch"],
        "/telephony/providers": ["get", "post"],
        "/telephony/providers/{provider_id}/verify": ["post"],
        "/telephony/providers/{provider_id}": ["delete"],
    }
    jwt_only_paths = {
        "/api-keys": ["post"],
        "/api-keys/{key_id}": ["delete", "patch"],
        "/telephony/providers": ["post"],
        "/telephony/providers/{provider_id}/verify": ["post"],
        "/telephony/providers/{provider_id}": ["delete"],
    }

    for path, methods in (schema.get("paths") or {}).items():
        for method, operation in methods.items():
            if method.upper() == "HEAD":
                continue
            # JWT-only endpoints
            if path in jwt_only_paths and method in jwt_only_paths.get(path, []):
                operation["security"] = _JWT_ONLY_SECURITY
            # All other auth-required endpoints
            elif path in protected_paths and method in protected_paths.get(path, []):
                operation["security"] = _AUTH_SECURITY

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
