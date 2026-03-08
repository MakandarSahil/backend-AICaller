"""
app/workers/celery_app.py

Celery application definition.
Phase 1: just the setup, no tasks yet.
Phase 3: will add tasks for async logging, RAG ingestion, etc.
"""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "aicaller",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # How long to keep task results in Redis
    result_expires=3600,
)