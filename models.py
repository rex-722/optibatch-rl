from pydantic import BaseModel
from typing import List

class Assignment(BaseModel):
    rider_id: int
    order_id: int
    action: str

class OptiBatchAction(BaseModel):
    assignments: List[Assignment] = []
    thought_process: str = "Waiting for data..."

class ResetRequest(BaseModel):
    mode: str = "normal"
