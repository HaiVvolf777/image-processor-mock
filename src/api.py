"""
FastAPI reference implementation.

This is a working prototype endpoint — it shows the .NET developer exactly
what request/response shape to expect when they call the Gemini analyzer.
They can either:
  (a) Keep this Python service running and call it from .NET via HTTP, or
  (b) Port the prompt + logic to C# using the Gemini REST API directly.

Run locally:
    uvicorn src.api:app --reload --port 8000

Then POST to http://localhost:8000/analyze
"""

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from .analyzer import analyze_gym_images

load_dotenv()

app = FastAPI(
    title="Hotel Gym Image Analyzer",
    description="Analyze hotel gym images with Gemini Vision API",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    imageUrls: list[HttpUrl] = Field(
        min_length=1,
        max_length=20,
        description="1–20 publicly accessible gym image URLs",
        examples=[
            [
                "https://cdn-media.hotelgyms.com/images/gym/optimized/5ce4cb5b-ba0e-460f-aa0d-cec49520d64e.jpg",
                "https://cdn-media.hotelgyms.com/images/gym/optimized/e79f6b45-bad2-4125-928a-4a6ae8396c49.jpg",
            ]
        ],
    )
    gymId: str | None = Field(
        default=None,
        description="Optional reference ID returned in the response",
    )


class AnalyzeResponse(BaseModel):
    gymId: str | None
    analysis: dict
    processingTimeMs: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze 1–20 hotel gym images and return structured JSON.

    Each detected attribute includes a confidence score (0.0 – 1.0).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    urls = [str(u) for u in request.imageUrls]

    t0 = time.time()
    try:
        analysis = await analyze_gym_images(urls, api_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {exc}")

    return AnalyzeResponse(
        gymId=request.gymId,
        analysis=analysis,
        processingTimeMs=int((time.time() - t0) * 1000),
    )
