"""
Tests for the gym image analyzer.

Real-API tests require GEMINI_API_KEY in the environment.
Schema / parsing tests run without any API key.
"""

import json
import os

import pytest

from src.prompt import GymAnalysis


# ---------------------------------------------------------------------------
# Schema / parsing tests (no API key required)
# ---------------------------------------------------------------------------

VALID_RESPONSE = {
    "experience": {
        "equipmentBrand": {"value": "Technogym", "confidence": 0.92},
        "waterOption": {"value": "Water Station", "confidence": 0.87},
        "hasTowels": {"value": True, "confidence": 0.78},
    },
    "cardio": {
        "hasPeloton": {"value": False, "confidence": 0.95},
    },
    "imageCount": 2,
    "analysisNotes": None,
}


def test_schema_parses_valid_response():
    analysis = GymAnalysis(**VALID_RESPONSE)
    assert analysis.experience.equipmentBrand.value == "Technogym"
    assert analysis.experience.equipmentBrand.confidence == 0.92
    assert analysis.experience.waterOption.value == "Water Station"
    assert analysis.experience.hasTowels.value is True
    assert analysis.cardio.hasPeloton.value is False
    assert analysis.imageCount == 2
    assert analysis.analysisNotes is None


def test_schema_accepts_null_values():
    data = {
        "experience": {
            "equipmentBrand": {"value": None, "confidence": 0.2},
            "waterOption": {"value": "Unknown", "confidence": 0.4},
            "hasTowels": {"value": None, "confidence": 0.1},
        },
        "cardio": {
            "hasPeloton": {"value": None, "confidence": 0.0},
        },
        "imageCount": 1,
        "analysisNotes": "Image too dark to determine brand.",
    }
    analysis = GymAnalysis(**data)
    assert analysis.experience.equipmentBrand.value is None
    assert analysis.analysisNotes == "Image too dark to determine brand."


def test_schema_clamps_confidence_out_of_range():
    data = {
        "experience": {
            "equipmentBrand": {"value": "Matrix", "confidence": 1.5},  # clamped to 1.0
            "waterOption": {"value": "None", "confidence": -0.1},     # clamped to 0.0
            "hasTowels": {"value": False, "confidence": 0.8},
        },
        "cardio": {"hasPeloton": {"value": False, "confidence": 0.9}},
        "imageCount": 1,
    }
    analysis = GymAnalysis(**data)
    assert analysis.experience.equipmentBrand.confidence == 1.0
    assert analysis.experience.waterOption.confidence == 0.0


def test_schema_serializes_to_dict():
    analysis = GymAnalysis(**VALID_RESPONSE)
    result = analysis.model_dump()
    assert isinstance(result, dict)
    assert result["experience"]["equipmentBrand"]["value"] == "Technogym"
    assert result["cardio"]["hasPeloton"]["value"] is False


# ---------------------------------------------------------------------------
# Live API tests (skipped unless GEMINI_API_KEY is set)
# ---------------------------------------------------------------------------

SAMPLE_URLS = [
    "https://cdn-media.hotelgyms.com/images/gym/optimized/5ce4cb5b-ba0e-460f-aa0d-cec49520d64e.jpg",
    "https://cdn-media.hotelgyms.com/images/gym/optimized/e79f6b45-bad2-4125-928a-4a6ae8396c49.jpg",
]

@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_live_single_image():
    from src.analyzer import analyze_gym_images

    api_key = os.environ["GEMINI_API_KEY"]
    result = await analyze_gym_images([SAMPLE_URLS[0]], api_key)

    assert "experience" in result
    assert "cardio" in result
    assert result["imageCount"] == 1
    assert "equipmentBrand" in result["experience"]
    assert "waterOption" in result["experience"]
    assert "hasTowels" in result["experience"]
    assert "hasPeloton" in result["cardio"]


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_live_two_images():
    from src.analyzer import analyze_gym_images

    api_key = os.environ["GEMINI_API_KEY"]
    result = await analyze_gym_images(SAMPLE_URLS, api_key)

    assert result["imageCount"] == 2
    # Confidence scores must be in [0, 1]
    for field in result["experience"].values():
        assert 0.0 <= field["confidence"] <= 1.0
    assert 0.0 <= result["cardio"]["hasPeloton"]["confidence"] <= 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_live_validation_too_many_images():
    from src.analyzer import analyze_gym_images

    api_key = os.environ["GEMINI_API_KEY"]
    with pytest.raises(ValueError, match="Maximum 20"):
        await analyze_gym_images(SAMPLE_URLS * 11, api_key)  # 22 URLs


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_live_validation_empty_list():
    from src.analyzer import analyze_gym_images

    api_key = os.environ["GEMINI_API_KEY"]
    with pytest.raises(ValueError, match="At least one"):
        await analyze_gym_images([], api_key)
