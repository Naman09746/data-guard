from fastapi import APIRouter, HTTPException
from src.core.celery_app import celery_app
from celery.result import AsyncResult
from typing import Any
from src.api.schemas.tasks import TaskStatusResponse

router = APIRouter(tags=["System"])

@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status and progress of a background task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None,
        "progress": 0,
        "detail": None
    }
    
    if task_result.status == 'PROGRESS':
        response["progress"] = task_result.info.get("progress", 0)
        response["detail"] = task_result.info.get("status", "")
    elif task_result.status == 'SUCCESS':
        response["progress"] = 100
        response["result"] = task_result.get()
    elif task_result.status == 'FAILURE':
        response["detail"] = str(task_result.info)
        
    return response
