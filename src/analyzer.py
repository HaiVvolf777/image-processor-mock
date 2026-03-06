"""
Core gym image analyzer.

Takes a list of publicly accessible image URLs, fetches them, sends them
to Gemini Vision API, and returns a validated GymAnalysis dict.
"""

import asyncio
import json
import time
from typing import Optional

import httpx
from google import genai
from google.genai import types
from pydantic import ValidationError

from .prompt import ANALYSIS_PROMPT, GymAnalysis

MAX_IMAGES = 20
_SUPPORTED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_INSTRUCTION = "Analyze the gym image(s) above and return the JSON."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_image(client: httpx.AsyncClient, url: str) -> tuple[bytes, str]:
    """Fetch one image and return (bytes, mime_type)."""
    response = await client.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    if content_type not in _SUPPORTED_MIME:
        content_type = "image/jpeg"
    return response.content, content_type


def _build_contents(image_data: list[tuple[bytes, str]]) -> list[types.Part]:
    """Assemble Gemini content parts: [prompt, image1, ..., imageN, instruction]."""
    parts: list[types.Part] = [types.Part.from_text(text=ANALYSIS_PROMPT)]
    for image_bytes, mime_type in image_data:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    parts.append(types.Part.from_text(text=_INSTRUCTION))
    return parts


def _parse_response(raw_text: str, url_count: int) -> dict:
    """Parse Gemini JSON output and validate against GymAnalysis schema."""
    data = json.loads(raw_text)
    # Ensure imageCount reflects actual images sent
    data.setdefault("imageCount", url_count)
    validated = GymAnalysis(**data)
    return validated.model_dump()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def analyze_gym_images(
    image_urls: list[str],
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict:
    """
    Analyze hotel gym images with Gemini Vision.

    Args:
        image_urls: 1–20 publicly accessible image URLs.
        api_key:    Google Gemini API key.
        model:      Gemini model name (default: gemini-2.0-flash).

    Returns:
        GymAnalysis dict with detected attributes and confidence scores.

    Raises:
        ValueError:  If image_urls is empty or exceeds MAX_IMAGES.
        httpx.HTTPError: If an image URL is unreachable.
        json.JSONDecodeError / ValidationError: If Gemini returns bad output
            after a retry.
    """
    if not image_urls:
        raise ValueError("At least one image URL is required.")
    if len(image_urls) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images allowed, got {len(image_urls)}.")

    # Fetch all images in parallel
    async with httpx.AsyncClient() as http:
        image_data = await asyncio.gather(
            *[_fetch_image(http, url) for url in image_urls]
        )

    contents = _build_contents(list(image_data))

    gemini = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,  # Low temperature for consistent, repeatable output
    )

    response = await gemini.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    try:
        return _parse_response(response.text, len(image_urls))
    except (json.JSONDecodeError, ValidationError, KeyError):
        # One retry with temperature 0 and an explicit JSON reminder
        retry_contents = contents + [
            types.Part.from_text(text="IMPORTANT: Return ONLY valid JSON, nothing else.")
        ]
        retry_response = await gemini.aio.models.generate_content(
            model=model,
            contents=retry_contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        return _parse_response(retry_response.text, len(image_urls))


def analyze_gym_images_sync(
    image_urls: list[str],
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict:
    """Synchronous wrapper around analyze_gym_images (for scripting / testing)."""
    return asyncio.run(analyze_gym_images(image_urls, api_key, model))
