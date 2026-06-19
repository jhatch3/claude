"""
File schemas.py

Pydantic request/response models for the HTTP API (app.py). These define the
shapes FastAPI validates against and documents in the OpenAPI schema.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


class HealthResponse(BaseModel):
    status: str = "ok"
    redis: bool
    agent: bool


class ResetResponse(BaseModel):
    status: str = "ok"
    message: str
