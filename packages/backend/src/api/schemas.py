from pydantic import BaseModel


class StatusResponse(BaseModel):
    status: str
    version: str


class WorldStateResponse(BaseModel):
    available: bool
    turn: int
    current_time: str
    history_size: int
    agent_names: list[str]


class WorldStepResponse(BaseModel):
    turn: int
    speaker_name: str
    reply: str
    silent_reason: str
    trace: dict[str, object]
    parse_failure_rate: float
    silent_rate: float
    semantic_repeat_rate: float
    topic_progress_rate: float
