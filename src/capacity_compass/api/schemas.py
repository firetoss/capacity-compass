"""API request/response schemas (FastAPI/Pydantic)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EvaluationRequestModel(BaseModel):
    model: str
    param_count_b: Optional[float] = None
    max_context_len: Optional[int] = None
    precision: Optional[str] = None
    vendor_scope: Optional[List[str]] = None
    generate_summary: Optional[bool] = Field(
        default=None,
        description="Override default LLM summary switch; None uses server setting.",
    )


class EvaluationResponseModel(BaseModel):
    quick_answer: Optional[dict]
    scenes: dict
    raw_evaluation: dict
    llm_summary: Optional[str] = None
    disclaimers: List[str] = Field(default_factory=list)
