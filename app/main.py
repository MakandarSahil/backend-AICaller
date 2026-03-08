"""
aicaller-backend
Main FastAPI application entry point.
"""

import time
import redis.asyncio as aioredis
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.health import router as health_router


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup and once on shutdown.
    Use this to open / close connections (Redis, DB clients, etc.)
    """
    # Startup
    app.state.redis = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    print(f"[startup] Redis connected at {settings.REDIS_URL}")
    print(f"[startup] Environment: {settings.ENV}")

    yield  # <-- app is running while we're here

    # Shutdown
    await app.state.redis.aclose()
    print("[shutdown] Redis connection closed")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AICaller Backend",
    description="STT → LLM → TTS pipeline server for AICaller platform",
    version="0.1.0",
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router, tags=["Health"])


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "aicaller-backend",
        "version": "0.1.0",
        "status": "running",
    }