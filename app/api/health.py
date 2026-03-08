"""
app/api/health.py

Health check endpoints.

Why do we need more than one?
  /health      — basic liveness probe (is the process alive?)
  /health/ready — readiness probe (is it ready to serve traffic? Redis connected?)
  /health/info  — human-readable info about the running instance

Docker Compose, Traefik, and later K8s all use these to decide
whether to route traffic to this container.
"""

import time
import platform
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

# Track startup time so we can report uptime
_START_TIME = time.time()


# ── Liveness ──────────────────────────────────────────────────────────────────
@router.get("/health")
async def health_live():
    """
    Liveness probe.
    Returns 200 as long as the process is running.
    Traefik / Docker healthcheck hits this every 10s.
    If this returns non-200, the container is restarted.
    """
    return {"status": "ok"}


# ── Readiness ─────────────────────────────────────────────────────────────────
@router.get("/health/ready")
async def health_ready(request: Request):
    """
    Readiness probe.
    Checks that all dependencies (Redis) are reachable.
    Returns 200 only when the app is truly ready to handle requests.
    Returns 503 if any dependency is down.
    """
    checks = {}

    # Check Redis
    try:
        redis = request.app.state.redis
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())

    if not all_ok:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "checks": checks},
        )

    return {"status": "ready", "checks": checks}


# ── Info ──────────────────────────────────────────────────────────────────────
@router.get("/health/info")
async def health_info(request: Request):
    """
    Human-readable info endpoint.
    Useful during deployments to confirm which version is running.
    Not a liveness/readiness probe — just for debugging.
    """
    from app.core.config import settings

    uptime_seconds = round(time.time() - _START_TIME, 2)

    return {
        "service": "aicaller-backend",
        "version": "0.1.0",
        "environment": settings.ENV,
        "uptime_seconds": uptime_seconds,
        "python": platform.python_version(),
        "platform": platform.system(),
    }