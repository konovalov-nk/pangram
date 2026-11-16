from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .detector import DetectorEnsemble

app = FastAPI(title="Local AI Detector", version="0.1.0")
_detector: DetectorEnsemble | None = None
_load_lock = Lock()


class DetectionRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=500_000)]
    threshold: Annotated[float, Field(gt=0, lt=1)] = 0.5


def get_detector() -> DetectorEnsemble:
    global _detector
    if _detector is None:
        with _load_lock:
            if _detector is None:
                _detector = DetectorEnsemble()
    return _detector


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("static") / "index.html")


@app.get("/api/status")
async def status() -> dict:
    if _detector is None:
        return {"ready": False, "message": "Models load on the first detection request."}
    return _detector.status()


@app.post("/api/detect")
async def detect(request: DetectionRequest) -> dict:
    try:
        detector = await asyncio.to_thread(get_detector)
        return await asyncio.to_thread(detector.detect, request.text, request.threshold)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
