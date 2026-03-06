"""Quick test against the two sample images. Run from project root."""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from src.analyzer import analyze_gym_images

SAMPLE_URLS = [
    "https://cdn-media.hotelgyms.com/images/gym/optimized/5ce4cb5b-ba0e-460f-aa0d-cec49520d64e.jpg",
    "https://cdn-media.hotelgyms.com/images/gym/optimized/e79f6b45-bad2-4125-928a-4a6ae8396c49.jpg",
]

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found in .env")

    print(f"Analyzing {len(SAMPLE_URLS)} images...")
    result = await analyze_gym_images(SAMPLE_URLS, api_key)
    print(json.dumps(result, indent=2))

asyncio.run(main())
