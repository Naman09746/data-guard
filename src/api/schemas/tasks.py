from pydantic import BaseModel
from typing import Optional, Any
from uuid import UUID

class TaskResponse(BaseModel):
    task_id: str
    scan_id: UUID
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    result: Optional[Any] = None
    detail: Optional[str] = None
