"""FastAPI server exposing capacity evaluation endpoint."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException

from ..pipeline.normalizer import EvaluationRequest
from ..pipeline.service import build_service_context, evaluate_capacity
from .schemas import EvaluationRequestModel, EvaluationResponseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="CapacityCompass API")


def get_context():
    if not hasattr(get_context, "_cache"):
        get_context._cache = build_service_context()
    return get_context._cache


@app.post("/api/llm/capacity/evaluate", response_model=EvaluationResponseModel)
async def evaluate(request: EvaluationRequestModel, context=Depends(get_context)):
    try:
        evaluation = evaluate_capacity(
            EvaluationRequest(**request.model_dump()),
            context,
            generate_summary=request.generate_summary,
        )
        return EvaluationResponseModel(**evaluation)
    except Exception as exc:  # pragma: no cover - FastAPI handles
        logger.exception("evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
