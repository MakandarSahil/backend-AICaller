"""
tests/test_health.py

Basic health endpoint tests.
These run in GitHub Actions (Job 1: test) on every push.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from app.main import app


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "aicaller-backend"
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_health_live():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_info():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "aicaller-backend"
    assert "uptime_seconds" in data
    assert "environment" in data


@pytest.mark.asyncio
async def test_health_ready_redis_ok():
    """
    Mock Redis ping so we can test the ready endpoint
    without a real Redis connection in unit tests.
    """
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Inject mock redis into app state
        app.state.redis = mock_redis
        response = await client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["redis"] == "ok"