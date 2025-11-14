"""FastAPI server exposing capacity evaluation endpoint."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException

from ..pipeline.normalizer import EvaluationRequest
from ..pipeline.service import build_service_context, evaluate_capacity
from ..settings import get_runtime_settings
from .schemas import EvaluationRequestModel, EvaluationResponseModel

logger = logging.getLogger(__name__)
settings = get_runtime_settings()
logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
app = FastAPI(title="CapacityCompass API")


def get_context():
    if not hasattr(get_context, "_cache"):
        get_context._cache = build_service_context()
    return get_context._cache


@app.post("/api/llm/capacity/evaluate", response_model=EvaluationResponseModel)
async def evaluate(request: EvaluationRequestModel, context=Depends(get_context)):
    try:
        generate_summary = (
            request.generate_summary
            if request.generate_summary is not None
            else settings.enable_llm_summary
        )
        normalized_request_payload = request.model_dump(exclude={"generate_summary"})
        evaluation = evaluate_capacity(
            EvaluationRequest(**normalized_request_payload),
            context,
            generate_summary=generate_summary,
        )
        return EvaluationResponseModel(**evaluation)
    except Exception as exc:  # pragma: no cover - FastAPI handles
        logger.exception("evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
