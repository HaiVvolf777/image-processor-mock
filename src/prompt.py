"""
Gemini Vision prompt and response schema for hotel gym analysis.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Response schema (Pydantic)
# Used both for validation and for generating the JSON schema shown to Gemini.
# ---------------------------------------------------------------------------

class FieldWithConfidence(BaseModel):
    """A detected attribute paired with a confidence score (0.0 – 1.0)."""

    value: Any = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


class ExperienceCategory(BaseModel):
    equipmentBrand: FieldWithConfidence
    waterOption: FieldWithConfidence
    hasTowels: FieldWithConfidence


class CardioCategory(BaseModel):
    hasPeloton: FieldWithConfidence


class GymAnalysis(BaseModel):
    experience: ExperienceCategory
    cardio: CardioCategory
    imageCount: int = 0
    analysisNotes: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are a professional hotel gym analyst. You will receive one or more images of a hotel gym.
Treat ALL images as belonging to ONE gym and analyze them together.

Return ONLY a valid JSON object — no explanation, no markdown fences, no extra text.

─────────────────────────────────────────
DETECTION RULES
─────────────────────────────────────────

equipmentBrand
  Identify the single most visible / dominant equipment brand across cardio
  and strength machines (treadmills, ellipticals, bikes, cable machines,
  weight racks, multi-gyms).
  Known brands to look for: Technogym, Matrix, Precor, Life Fitness, Cybex,
  Nautilus, StairMaster, Hammer Strength, Keiser, Concept2, Peloton.
  If a brand logo is clearly visible but not in the list, return that brand name.
  If no brand can be identified, return null.

waterOption
  "Bottled Water"  — individual water bottles are stocked/provided
  "Water Station"  — water dispenser, cooler, or refill station is visible
  "None"           — no water amenity visible in any image
  "Unknown"        — images do not show enough of the gym to determine

hasTowels
  true   — towels visible on racks, folded on benches, in dispensers
  false  — gym area is clearly visible but no towels present
  null   — cannot determine from the available images

hasPeloton
  true   — a Peloton bike is visible (large portrait touchscreen tablet,
            distinctive curved carbon steel fork, Peloton logo)
  false  — no Peloton bikes visible (other bikes may be present)
  null   — cannot determine from the available images

confidence score
  1.0          — absolutely certain (logo / item unmistakably clear)
  0.80 – 0.99  — very confident
  0.60 – 0.79  — fairly confident
  0.40 – 0.59  — uncertain
  < 0.40       — too uncertain; set the corresponding value to null

imageCount
  Total number of images you received and analyzed.

analysisNotes
  A brief plain-text note ONLY if image quality was poor or detection was
  difficult. null otherwise.

─────────────────────────────────────────
REQUIRED JSON SCHEMA (return exactly this structure)
─────────────────────────────────────────

{
  "experience": {
    "equipmentBrand": { "value": "<brand name or null>", "confidence": 0.0 },
    "waterOption":    { "value": "<Bottled Water|Water Station|None|Unknown>", "confidence": 0.0 },
    "hasTowels":      { "value": true/false/null, "confidence": 0.0 }
  },
  "cardio": {
    "hasPeloton": { "value": true/false/null, "confidence": 0.0 }
  },
  "imageCount": 0,
  "analysisNotes": null
}
"""
