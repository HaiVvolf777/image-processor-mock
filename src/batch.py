"""
Batch processor for large-scale hotel gym image analysis.

Designed for ~100k gym records. Reads JSONL input, writes JSONL output,
checkpoints progress (safe to resume after interruption), and respects
Gemini rate limits via semaphore-gated concurrency.

Input JSONL format (one record per line):
    {"gymId": "hotel-123", "imageUrls": ["https://...", "https://..."]}

Output JSONL format:
    {
      "gymId": "hotel-123",
      "analysis": { ...GymAnalysis... },
      "processingTimeMs": 1842,
      "error": null
    }

Failed records are written to a separate JSONL for later retry.

Usage:
    python -m src.batch \
        --input  records.jsonl \
        --output results.jsonl \
        --failed failed.jsonl \
        --concurrency 10
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
from tqdm.asyncio import tqdm

from .analyzer import analyze_gym_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processed_ids(output_path: Path) -> set[str]:
    """Return the set of gymIds already present in the output file (checkpointing)."""
    processed: set[str] = set()
    if not output_path.exists():
        return processed
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if gym_id := record.get("gymId"):
                    processed.add(gym_id)
            except json.JSONDecodeError:
                pass
    return processed


async def _record_stream(input_path: Path) -> AsyncIterator[dict]:
    """Yield records from a JSONL file, skipping blank lines and bad JSON."""
    with input_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed JSON at line {line_no}")


# ---------------------------------------------------------------------------
# Core batch logic
# ---------------------------------------------------------------------------

async def process_batch(
    input_path: str | Path,
    output_path: str | Path,
    failed_path: str | Path,
    api_key: str,
    concurrency: int = 10,
    model: str = "gemini-2.0-flash",
) -> dict:
    """
    Process a JSONL file of gym records through Gemini Vision.

    Args:
        input_path:   Path to input JSONL file.
        output_path:  Path to output JSONL file (appended; supports resume).
        failed_path:  Path to failed-records JSONL file.
        api_key:      Google Gemini API key.
        concurrency:  Max parallel Gemini requests (default 10).
                      Gemini paid tier supports ~1000 RPM; 10 is a safe default.
        model:        Gemini model name.

    Returns:
        Summary dict: total, succeeded, failed, skipped, elapsed_seconds.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    failed_path = Path(failed_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    processed_ids = _load_processed_ids(output_path)
    semaphore = asyncio.Semaphore(concurrency)

    counters = {"total": 0, "succeeded": 0, "failed": 0, "skipped": 0}
    start = time.time()

    out_file = output_path.open("a")
    fail_file = failed_path.open("a")

    async def process_one(record: dict) -> None:
        gym_id = record.get("gymId", f"row-{counters['total']}")
        image_urls = record.get("imageUrls", [])
        counters["total"] += 1

        if gym_id in processed_ids:
            counters["skipped"] += 1
            return

        async with semaphore:
            t0 = time.time()
            for attempt in range(3):
                try:
                    analysis = await analyze_gym_images(image_urls, api_key, model)
                    elapsed_ms = int((time.time() - t0) * 1000)
                    result = {
                        "gymId": gym_id,
                        "analysis": analysis,
                        "processingTimeMs": elapsed_ms,
                        "error": None,
                    }
                    out_file.write(json.dumps(result) + "\n")
                    out_file.flush()
                    counters["succeeded"] += 1
                    return
                except httpx.HTTPStatusError as exc:
                    # 429 rate limit — back off and retry
                    if exc.response.status_code == 429:
                        await asyncio.sleep(2 ** attempt * 5)
                    else:
                        break
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        break

            # All retries failed
            fail_record = {"gymId": gym_id, "imageUrls": image_urls, "error": "max retries exceeded"}
            fail_file.write(json.dumps(fail_record) + "\n")
            fail_file.flush()
            counters["failed"] += 1

    tasks = []
    async for record in _record_stream(input_path):
        tasks.append(asyncio.create_task(process_one(record)))

    await tqdm.gather(*tasks, desc="Processing gyms")

    out_file.close()
    fail_file.close()

    return {
        "total": counters["total"],
        "succeeded": counters["succeeded"],
        "failed": counters["failed"],
        "skipped": counters["skipped"],
        "elapsed_seconds": round(time.time() - start, 1),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="Batch gym image analyzer")
    parser.add_argument("--input",       required=True,          help="Input JSONL file")
    parser.add_argument("--output",      default="results.jsonl", help="Output JSONL file")
    parser.add_argument("--failed",      default="failed.jsonl",  help="Failed records JSONL")
    parser.add_argument("--concurrency", type=int, default=10,    help="Parallel requests")
    parser.add_argument("--model",       default="gemini-2.0-flash")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY environment variable is not set.")

    summary = await process_batch(
        input_path=args.input,
        output_path=args.output,
        failed_path=args.failed,
        api_key=api_key,
        concurrency=args.concurrency,
        model=args.model,
    )

    print("\nBatch complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(_main())
