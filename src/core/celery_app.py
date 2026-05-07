from celery import Celery
from src.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "dataguard",
    broker=f"redis://{settings.redis.host}:{settings.redis.port}/{settings.redis.db}",
    backend=f"redis://{settings.redis.host}:{settings.redis.port}/{settings.redis.db}",
    include=[
        "src.tasks.eda_tasks",
        "src.tasks.leakage_tasks"
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600, # 1 hour max
)
