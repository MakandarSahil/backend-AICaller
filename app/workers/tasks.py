"""
app/workers/tasks.py

Celery task definitions.
Phase 1: placeholder only.
Phase 3: will add log_conversation, ingest_document, etc.
"""

from app.workers.celery_app import celery_app


@celery_app.task(name="tasks.ping")
def ping():
    """
    Test task — confirms Celery worker is running.
    Call from API: from app.workers.tasks import ping; ping.delay()
    """
    return "pong"