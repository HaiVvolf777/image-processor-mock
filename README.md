# Hotel Gym Image Analyzer — Prototype

Analyzes hotel gym images using **Google Gemini Vision API** and returns
structured JSON with equipment brand, water options, towels, and Peloton
detection — each with a confidence score.

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

> Don't have an API key yet? See `docs/integration-guide.md` → Section 1.

### 3. Run a quick test

```bash
python - <<'EOF'
import asyncio, os
from dotenv import load_dotenv
from src.analyzer import analyze_gym_images

load_dotenv()
result = asyncio.run(analyze_gym_images(
    [
        "https://cdn-media.hotelgyms.com/images/gym/optimized/5ce4cb5b-ba0e-460f-aa0d-cec49520d64e.jpg",
        "https://cdn-media.hotelgyms.com/images/gym/optimized/e79f6b45-bad2-4125-928a-4a6ae8396c49.jpg",
    ],
    os.environ["GEMINI_API_KEY"]
))
import json; print(json.dumps(result, indent=2))
EOF
```

### 4. Run the API server (optional)

```bash
uvicorn src.api:app --reload --port 8000
# POST http://localhost:8000/analyze
```

### 5. Run tests

```bash
# Schema tests (no API key required)
pytest tests/ -v

# Live API tests (requires GEMINI_API_KEY)
GEMINI_API_KEY=your-key pytest tests/ -v
```

---

## Batch Processing (100k images)

Prepare a JSONL file where each line is:
```jsonl
{"gymId": "hotel-001", "imageUrls": ["https://...", "https://..."]}
```

Then run:
```bash
python -m src.batch \
  --input  records.jsonl \
  --output results.jsonl \
  --failed failed.jsonl \
  --concurrency 10
```

The batch processor is resumable — safe to interrupt and re-run.

---

## Response Format

```json
{
  "experience": {
    "equipmentBrand": { "value": "Technogym", "confidence": 0.92 },
    "waterOption":    { "value": "Water Station", "confidence": 0.87 },
    "hasTowels":      { "value": true, "confidence": 0.78 }
  },
  "cardio": {
    "hasPeloton": { "value": false, "confidence": 0.95 }
  },
  "imageCount": 2,
  "analysisNotes": null
}
```

---

## Project Structure

```
src/
  prompt.py      — Gemini prompt + Pydantic response schema
  analyzer.py    — Core analysis (single gym, 1–20 images)
  batch.py       — Async batch processor with checkpointing
  api.py         — FastAPI reference endpoint
tests/
  test_analyzer.py
docs/
  integration-guide.md   — .NET developer integration guide
```

---

## For the .NET Developer

See `docs/integration-guide.md` for:
- Google Cloud API key setup
- Full request/response spec
- C# code examples (both via HTTP and direct Gemini REST)
- Batch processing guide
- Retry strategy and latency expectations
