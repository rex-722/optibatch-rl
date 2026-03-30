from pydantic import BaseModel
from typing import List, Dict, Any

class OptiBatchAction(BaseModel):
    action_type: int

class ResetRequest(BaseModel):
    task_id: str = "medium"

class OptiBatchState(BaseModel):
    time: int
    pending_orders_count: int
    available_riders: int
    oldest_order_time: int
    sla_limit: int
    max_capacity: int
    live_phase: str = "NORMAL"
    just_delivered: int = 0
    just_dispatched: bool = False

class StepResponse(BaseModel):
    state: OptiBatchState
    reward: float
    done: bool
    info: Dict[str, Any]