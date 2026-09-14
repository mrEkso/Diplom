"""Generate benchmark images for every entity in benchmark_entities.csv.

What this script does, in plain terms:
For every entity (e.g. "Bismarck") in the CSV file, this script asks each
configured image-generation model the simple prompt "An image of [Entity]."
and saves whatever image comes back. This lets us later compare how factually
correct different AI image models are.

Where images are saved:
    generated_images/<model_name>/<Category_or_Random_Tier>/<Entity>.png

How to run it:
    python3 generate_benchmark_images.py
    python3 generate_benchmark_images.py --models gpt_image_2
    python3 generate_benchmark_images.py --csv benchmark_entities.csv --sleep 2 --concurrency 4
"""

import argparse
import base64
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

# Reads secret keys (like API tokens) from a local .env file into the environment,
# so we never have to write them directly in this script.
load_dotenv()

# This makes sure file paths below always work, no matter which folder you were
# in when you ran the script (everything is relative to this script's own folder).
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH_DEFAULT = str(BASE_DIR.parent / "entities" / "benchmark_entities_final.csv")
IMAGES_ROOT = str(BASE_DIR / "generated_images")
DEFAULT_SLEEP_SECONDS = 1  # pause (in seconds) between each image request per worker
DEFAULT_CONCURRENCY = 10   # how many images to generate at the same time, by default
FAILURES_LOG = str(BASE_DIR / "generation_failures.csv")
MAX_RETRIES = 5  # how many times to retry a failed request before giving up

# Connection details for TU Dresden's ScaDS.AI image-generation service.
SCADS_API_URL = os.environ.get("SCADS_API_URL", "https://llm.scads.ai/v1")
SCADS_API_KEY = os.environ.get("SCADS_API_KEY")

# Connection details for OpenRouter, used to reach GPT Image 2.
OPENROUTER_API_URL = os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def call_with_retry(func, *args, max_retries=MAX_RETRIES, base_delay=2.0, **kwargs):
    """Run a function, and if it fails because a server is busy/overloaded, wait
    a bit and try again (instead of giving up immediately).

    Waits longer after each failed attempt ("exponential backoff"), which is the
    polite way to retry against APIs so we don't hammer them with requests :)
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            # 429 = too many requests, 5xx = server-side error. Both are worth retrying.
            if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"    Retryable error ({status}), retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise
        except requests.exceptions.RequestException as e:
            # Covers things like a dropped connection or timeout.
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"    Network error ({e}), retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise


# --- Model backends --------------------------------------------------------
# An "image backend" is just a function that takes a text prompt and returns
# the raw bytes of the generated image (PNG/JPEG). To support a new image
# model, write a function like the ones below and add it to MODEL_BACKENDS.


def generate_image_gpt_image_2(prompt):
    """Ask GPT Image 2 (via OpenRouter) to generate one image for the given prompt."""
    from openai import OpenAI

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see .env).")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_API_URL)
    result = client.images.generate(
        model="openai/gpt-image-2",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    data = result.data[0]

    # GPT Image models usually send the image back already encoded as base64 text.
    if getattr(data, "b64_json", None):
        return base64.b64decode(data.b64_json)

    # Fallback: if instead we got a download link, fetch the actual image bytes.
    resp = call_with_retry(requests.get, data.url, timeout=60)
    resp.raise_for_status()
    return resp.content


def generate_image_scads(prompt, model_id):
    """Ask a model hosted on TU Dresden's ScaDS.AI service to generate one image."""
    if not SCADS_API_KEY:
        raise RuntimeError("SCADS_API_KEY is not set (see .env).")

    headers = {
        "Authorization": f"Bearer {SCADS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "prompt": prompt,
        "n": 1,
    }

    resp = call_with_retry(
        requests.post,
        f"{SCADS_API_URL.rstrip('/')}/images/generations",
        headers=headers,
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()["data"][0]

    # The response either contains the image directly as base64 text...
    if "b64_json" in data and data["b64_json"]:
        return base64.b64decode(data["b64_json"])

    # ...or a URL we need to download the image from.
    image_url = data["url"]
    img_resp = call_with_retry(requests.get, image_url, timeout=60)
    img_resp.raise_for_status()
    return img_resp.content


# The list of image-generation models this script knows how to call.
# Each key is the name you pass to --models on the command line.
MODEL_BACKENDS = {
    "gpt_image_2": generate_image_gpt_image_2,
    "scads_flux2_dev": lambda prompt: generate_image_scads(prompt, "black-forest-labs/FLUX.2-dev"),
    "scads_flux2_klein": lambda prompt: generate_image_scads(prompt, "black-forest-labs/FLUX.2-klein-4B"),
}


# --- Helpers -----------------------------------------------------------------

def sanitize_component(name):
    """Turn any text into something safe to use as a file/folder name.

    Removes characters that aren't letters, numbers, spaces, underscores, or
    dashes, then replaces spaces with underscores (e.g. "Notre Dame!" -> "Notre_Dame").
    """
    cleaned = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    return cleaned.replace(" ", "_") or "unnamed"


def load_entities(csv_path):
    """Read the entities CSV file and return its rows as a list of dictionaries."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_failure(model_name, entity, error):
    """Append one failed generation attempt to the failures CSV, so it can be retried later."""
    is_new = not os.path.exists(FAILURES_LOG)
    with open(FAILURES_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["Model", "Entity", "Error"])
        writer.writerow([model_name, entity, str(error)])


def generate_one(model_name, backend, row, sleep_seconds):
    """Generate one image for one entity, and save it to disk.

    Skips the request entirely if the image file already exists (so re-running
    the script after a crash doesn't waste money regenerating everything).
    Returns a short text describing what happened (generated/skipped/failed).
    """
    entity = row["Entity"]
    category = row.get("Category", "Uncategorized") or "Uncategorized"
    # "Random" entities are split further into their popularity tier folder
    folder_category = category if category != "Random" else f"Random_{row['Popularity_Tier']}"

    out_dir = os.path.join(IMAGES_ROOT, model_name, sanitize_component(folder_category))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sanitize_component(entity)}.png")

    if os.path.exists(out_path):
        return f"[{model_name}] Skipping (exists): {entity}"

    print(f"[{model_name}] Started: {entity}", flush=True)

    # Every entity gets the exact same simple prompt, so results are comparable
    # across models (no extra hints/context are given).
    prompt = f"An image of {entity}."
    try:
        image_bytes = call_with_retry(backend, prompt)
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        result = f"[{model_name}] Generated: {entity}"
    except Exception as e:
        log_failure(model_name, entity, e)
        result = f"[{model_name}] FAILED: {entity} ({e})"
    finally:
        # Wait a bit after every request (success or failure) to avoid hitting rate limits.
        time.sleep(sleep_seconds)

    return result


def generate_for_model(model_name, entities, sleep_seconds, concurrency):
    """Generate images for every entity, for a single model, running several
    requests at the same time (in parallel) to finish faster."""
    backend = MODEL_BACKENDS[model_name]
    total = len(entities)
    done = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # Start all the image-generation jobs at once; the pool runs up to
        # `concurrency` of them in parallel and queues the rest.
        futures = [
            pool.submit(generate_one, model_name, backend, row, sleep_seconds)
            for row in entities
        ]
        # Print a progress line every time one of the jobs finishes (in any order).
        for future in as_completed(futures):
            done += 1
            print(f"({done}/{total}) {future.result()}", flush=True) #Parameter flush=True ensures that the output is printed immediately, which is useful for real-time progress tracking in long-running tasks.


def main():
    """Entry point: read command-line options, load entities, and generate all images."""
    # Parse command-line arguments, with defaults and help text.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=CSV_PATH_DEFAULT, help="Path to benchmark_entities.csv")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_BACKENDS.keys()),
        default=list(MODEL_BACKENDS.keys()),
        help="Which model backends to run (default: all)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds each worker waits after its own request (per-thread rate limiting)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of images to generate in parallel per model",
    )
    args = parser.parse_args()

    # Load the list of entities to generate images for.
    entities = load_entities(args.csv)
    print(f"Loaded {len(entities)} entities from {args.csv}")

    # Generate images one model at a time (each model runs its own batch in parallel).
    for model_name in args.models:
        generate_for_model(model_name, entities, args.sleep, args.concurrency)

    print("Done.")
    if os.path.exists(FAILURES_LOG):
        print(f"Some requests failed - see {FAILURES_LOG} for details (rerun the script to retry them).")


if __name__ == "__main__":
    main()

