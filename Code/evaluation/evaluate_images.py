"""Evaluate generated images with a two-step visual question-answering pipeline.

For every generated image, this script does the following:
1. Ask a judge model to describe only what it sees, without telling it the entity name.
2. Compare that blind description with the entity's Wikipedia summary.
3. Save type-match, feature-match, and hallucination results in a CSV file.

Keeping the first step blind helps prevent the entity name from telling the model
what it should see. Wikipedia provides the reference text used for the factuality check.
"""

import argparse
import base64
import csv
import json
import requests
import logging
import mimetypes
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

# Load API settings from the local .env file, if one exists.
load_dotenv()

# Key is read from an environment variable (never hardcode secrets in code).
# Judge model is ScaDS.AI-hosted (OpenAI-compatible), same backend used for generation.
# export SCADS_API_KEY="sk-..." or set it in the .env file
# The API key is read from the environment; it is never written into this file.
SCADS_API_KEY = os.environ.get("SCADS_API_KEY")
SCADS_API_URL = os.environ.get("SCADS_API_URL", "https://llm.scads.ai/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemma-4-31B-it")

# Build paths from this file's location, so the script works from any current directory.
BASE_DIR = Path(__file__).resolve().parent
ENTITIES_CSV_PATH = str(BASE_DIR.parent / "entities" / "benchmark_entities_final.csv")
GENERATED_IMAGES_DIR = str(BASE_DIR.parent / "generation" / "generated_images")

WIKI_CACHE_PATH = str(BASE_DIR / "wiki_summary_cache.json")
# Include the judge name in output files so different judges never overwrite each other's data.
_JUDGE_SUFFIX = JUDGE_MODEL.split("/")[-1].replace(" ", "_")
FAILURES_LOG = str(BASE_DIR / f"evaluation_failures_{_JUDGE_SUFFIX}.csv")
LOG_PATH = str(BASE_DIR / f"evaluation_{_JUDGE_SUFFIX}.log")
MAX_RETRIES = 5
# A small default keeps simultaneous image uploads reliable on a shared connection.
DEFAULT_CONCURRENCY = 2

# Wikimedia uses this contact information to identify the automated client and apply
# the appropriate request limit.
# https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits
WIKI_USER_AGENT = "DiplomaResearchBot/1.0 (davyd.okaianchenko@tu-dresden.de; TU Dresden bachelor thesis)"

# Limit Wikipedia requests separately from judge-model requests.
WIKI_MAX_CONCURRENT = 2
_wiki_semaphore = threading.Semaphore(WIKI_MAX_CONCURRENT)

# Write progress and failures to both a log file and the terminal.
logger = logging.getLogger("evaluate_images")
logger.setLevel(logging.INFO)
_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_file_handler)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_console_handler)

# These locks protect files and shared data from simultaneous worker threads.
_csv_lock = threading.Lock()
_failures_lock = threading.Lock()
_wiki_cache_file_lock = threading.Lock()  # guards the _wiki_cache dict + wiki_summary_cache.json

# One lock per entity prevents duplicate Wikipedia requests for the same title.
_wiki_entity_locks_guard = threading.Lock()
_wiki_entity_locks = {}


def _get_entity_lock(entity_name):
    """Return the shared lock belonging to one Wikipedia entity name."""
    with _wiki_entity_locks_guard:
        return _wiki_entity_locks.setdefault(entity_name, threading.Lock())


def call_with_retry(func, *args, max_retries=MAX_RETRIES, base_delay=2.0, **kwargs):
    """Call func with exponential backoff on rate-limit/server errors (429/5xx).

    func must raise on a bad HTTP status (e.g. call requests.Response.raise_for_status()
    itself) - plain requests.get()/post() calls do NOT raise on 429/5xx by themselves,
    so wrap them (see _get_and_raise/_post_and_raise) or this backoff never triggers.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Retryable error ({status}), retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Network error ({e}), retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise


def log_failure(entity, model_name, error):
    """Append one failed evaluation to the failure CSV so it can be investigated later."""
    with _failures_lock:
        is_new = not os.path.exists(FAILURES_LOG)
        with open(FAILURES_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["Entity", "Model", "Error"])
            writer.writerow([entity, model_name, str(error)])


def load_wiki_cache():
    """Read the saved Wikipedia summaries, or start with an empty cache."""
    if os.path.exists(WIKI_CACHE_PATH):
        with open(WIKI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_wiki_cache(cache):
    """Write all cached Wikipedia summaries back to disk as readable JSON."""
    with open(WIKI_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


_wiki_cache = load_wiki_cache()


def _get_and_raise(url, **kwargs):
    """Make GET raise an exception for bad status codes, enabling the retry logic."""
    resp = requests.get(url, **kwargs)
    resp.raise_for_status()
    return resp


def _get_wiki_summary_response(url, **kwargs):
    """Let a missing article pass, but raise other errors so temporary failures are retried."""
    resp = requests.get(url, **kwargs)
    if resp.status_code == 404:
        return resp
    resp.raise_for_status()
    return resp


def _post_and_raise(url, **kwargs):
    """Make POST raise an exception for bad status codes, enabling the retry logic."""
    resp = requests.post(url, **kwargs)
    resp.raise_for_status()
    return resp


def resolve_wikipedia_titles(entity_name, limit=3):
    """Find likely official Wikipedia titles for an entity name.

    Search results handle redirects, alternate spellings, and transliterations
    that could otherwise make a direct page request fail.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "opensearch", "search": entity_name, "limit": limit, "namespace": 0, "format": "json"}
    try:
        with _wiki_semaphore:
            resp = call_with_retry(
                _get_and_raise, url, params=params, timeout=15, headers={"User-Agent": WIKI_USER_AGENT}
            )
        data = resp.json()
        # Wikipedia normally returns titles in the second list item, but missing
        # or unusual responses can be shorter, so check the shape first.
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            return [entity_name]
        return data[1]
    except (requests.exceptions.RequestException, ValueError):
        return [entity_name]


def get_wikipedia_summary(entity_name):
    """Fetch and cache the plain-text Wikipedia lead used as RQ3 reference text.

    Returns None when no usable summary can be found. Temporary network failures
    are not cached as missing articles, so a later run can try them again.
    """
    with _get_entity_lock(entity_name):
        with _wiki_cache_file_lock:
            if entity_name in _wiki_cache:
                return _wiki_cache[entity_name] or None

        summary = None
        # Cache an empty value only when every candidate was genuinely missing.
        # A temporary network error must not permanently force model-memory fallback.
        genuinely_missing = True
        for title in resolve_wikipedia_titles(entity_name):
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
            try:
                with _wiki_semaphore:
                    resp = call_with_retry(
                        _get_wiki_summary_response, url, timeout=15, headers={"User-Agent": WIKI_USER_AGENT}
                    )
                if resp.status_code == 404:
                    continue
                data = resp.json()
                if data.get("type") == "disambiguation":
                    continue
                extract = data.get("extract")
                if extract:
                    summary = extract
                    genuinely_missing = False
                    break
                genuinely_missing = False
            except requests.exceptions.RequestException as e:
                logger.warning(f"[{entity_name}] Wikipedia fetch failed transiently: {e}")
                genuinely_missing = False  # Do not cache a temporary failure as "missing".
                continue

        if summary is not None or genuinely_missing:
            with _wiki_cache_file_lock:
                _wiki_cache[entity_name] = summary or ""
                save_wiki_cache(_wiki_cache)
        return summary


def prefetch_wiki_summaries(entity_names):
    """Fetch all summaries before image evaluation starts, so judge workers do not wait for Wikipedia."""
    unique_entities = sorted(set(entity_names))
    logger.info(f"Prefetching Wikipedia summaries for {len(unique_entities)} unique entities...")

    with ThreadPoolExecutor(max_workers=WIKI_MAX_CONCURRENT) as pool:
        futures = {pool.submit(get_wikipedia_summary, entity): entity for entity in unique_entities}
        for done, future in enumerate(as_completed(futures), start=1):
            entity = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.warning(f"[{entity}] Wikipedia prefetch failed: {e}")
            if done % 25 == 0 or done == len(futures):
                logger.info(f"Wikipedia prefetch progress: {done}/{len(futures)}")

    logger.info("Wikipedia prefetch complete.")


def encode_image(image_path):
    """Read an image and return the base64 text and MIME type needed by the API."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8'), mime_type


def call_judge_model(messages, response_format_json=True):
    """Send one chat request to the ScaDS.AI judge model and return its text response."""
    headers = {
        "Authorization": f"Bearer {SCADS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": 0,  # The same input should produce the same score as often as possible.
        "max_tokens": 1500,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    response = call_with_retry(
        _post_and_raise,
        f"{SCADS_API_URL.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,  # Kimi-K3 is larger/slower than GPT-4o; 60s was too tight for image inputs
    )
    return response.json()['choices'][0]['message']['content']


def call_judge_model_json(messages):
    """Request JSON from the judge and retry once if its first answer is invalid JSON."""
    content = call_judge_model(messages)
    if content is None:
        raise RuntimeError("Model returned empty content (likely a content-policy refusal)")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"Malformed JSON from {JUDGE_MODEL}, retrying once...")
        content = call_judge_model(messages)
        if content is None:
            raise RuntimeError("Model returned empty content on retry (likely a content-policy refusal)")
        return json.loads(content)


def blind_classify_image(image_path):
    """Step 1: describe the image without revealing the expected entity.

    This keeps RQ1 unbiased: the model must identify the visible object from
    pixels, rather than simply repeating the entity name supplied in a prompt.
    """
    base64_image, mime_type = encode_image(image_path)

    prompt = """
    You are an objective computer-vision classifier. You are NOT told what this
    image is supposed to depict - judge only what is visually present.

    Output a raw JSON object with exactly these keys:
    1. "predicted_type": The primary object type shown (e.g., Building, Person, Animal, Landscape, Object, None).
    2. "description": A brief text description of the main visual elements.
    3. "propositions": A list of short, atomic factual claims about what is visually shown
       (e.g., ["The building has a round tower", "There is a river in the foreground"]).
    """

    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON-only API. Always return valid JSON without markdown wrapping."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        }
    ]
    return call_judge_model_json(messages)


def grounded_fact_check(entity_name, wiki_summary, expected_type, expected_feature,
                         predicted_type, description, propositions):
    """Step 2: compare the blind description with Wikipedia reference text.

    Curated entities receive RQ1, RQ2, and RQ3 checks. Random-tier entities
    have no expected type or feature, so they receive only the RQ3 check.
    """
    ground_truth_block = (
        wiki_summary if wiki_summary
        else "(No Wikipedia summary available - use your best general knowledge instead.)"
    )
    skip_rq1_rq2 = (expected_type == "Unknown")

    if skip_rq1_rq2:
        prompt = f"""
        You are a strict fact-checking judge. You are given:
        - The real-world entity name: "{entity_name}"
        - Reference text about this entity (from its Wikipedia summary):
          \"\"\"{ground_truth_block}\"\"\"
        - Visual observations made by a separate, blind image classifier (it did NOT know
          the entity name), about a generated image that is supposed to depict this entity:
          predicted_type = "{predicted_type}"
          description = "{description}"
          propositions = {json.dumps(propositions, ensure_ascii=False)}

        Using ONLY the ground truth text above (plus general knowledge only if the ground
        truth text is marked unavailable), output a raw JSON object with exactly these keys:

        1. "rq3_hallucinations_count": Integer. How many propositions are FALSE or contradicted
           by the ground truth text?
        2. "rq3_hallucinations_details": Short string explaining which propositions are wrong,
           or "None" if all are consistent with the ground truth.
        """
    else:
        prompt = f"""
        You are a strict fact-checking judge. You are given:
        - The real-world entity name: "{entity_name}"
        - Reference text about this entity (from its Wikipedia summary):
          \"\"\"{ground_truth_block}\"\"\"
        - An expected object type for this entity: "{expected_type}"
        - An expected unique feature for this entity: "{expected_feature}"
        - Visual observations made by a separate, blind image classifier (it did NOT know
          the entity name), about a generated image that is supposed to depict this entity:
          predicted_type = "{predicted_type}"
          description = "{description}"
          propositions = {json.dumps(propositions, ensure_ascii=False)}

        Using ONLY the ground truth text above (plus general knowledge only if the ground
        truth text is marked unavailable), output a raw JSON object with exactly these keys:

        1. "rq1_type_match": Boolean. Does predicted_type match the expected reality of "{entity_name}"?
        2. "rq2_feature_present": Boolean or null. Do the propositions/description clearly
           indicate the feature "{expected_feature}"? If expected_feature is "N/A", output null.
        3. "rq3_hallucinations_count": Integer. How many propositions are FALSE or contradicted
           by the ground truth text?
        4. "rq3_hallucinations_details": Short string explaining which propositions are wrong,
           or "None" if all are consistent with the ground truth.
        """

    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON-only API. Always return valid JSON without markdown wrapping."
        },
        {"role": "user", "content": prompt},
    ]
    return call_judge_model_json(messages)


# These keys define the minimum JSON shape expected from each model response.
REQUIRED_STEP1_KEYS = ("predicted_type", "description", "propositions")
REQUIRED_STEP2_KEYS_FULL = ("rq1_type_match", "rq2_feature_present", "rq3_hallucinations_count", "rq3_hallucinations_details")
REQUIRED_STEP2_KEYS_RQ3_ONLY = ("rq3_hallucinations_count", "rq3_hallucinations_details")


def evaluate_image_with_judge(image_path, entity_name, expected_type, expected_feature, model_name="unknown"):
    """Run both judge steps for one image and return normalized result fields."""
    step1 = blind_classify_image(image_path)
    missing1 = [k for k in REQUIRED_STEP1_KEYS if k not in step1]
    if missing1:
        # A malformed first response is retried before it is used as input to step 2.
        logger.warning(f"[{entity_name}/{model_name}] blind classification missing keys {missing1}, retrying once...")
        step1 = blind_classify_image(image_path)
        missing1 = [k for k in REQUIRED_STEP1_KEYS if k not in step1]
        if missing1:
            logger.warning(f"[{entity_name}/{model_name}] blind classification response missing keys: {missing1}")
            log_failure(entity_name, model_name, f"blind classification missing keys: {missing1}")


    wiki_summary = get_wikipedia_summary(entity_name)
    skip_rq1_rq2 = (expected_type == "Unknown")
    step2 = grounded_fact_check(
        entity_name=entity_name,
        wiki_summary=wiki_summary,
        expected_type=expected_type,
        expected_feature=expected_feature,
        predicted_type=step1.get("predicted_type"),
        description=step1.get("description"),
        propositions=step1.get("propositions"),
    )
    required_step2_keys = REQUIRED_STEP2_KEYS_RQ3_ONLY if skip_rq1_rq2 else REQUIRED_STEP2_KEYS_FULL
    missing2 = [k for k in required_step2_keys if k not in step2]
    if missing2:
        logger.warning(f"[{entity_name}/{model_name}] grounded fact-check response missing keys: {missing2}")
        log_failure(entity_name, model_name, f"grounded fact-check missing keys: {missing2}")

    return {
        "rq1_predicted_type": step1.get("predicted_type"),
        # Random-tier rows have no expected type or feature, so their RQ1/RQ2 values stay null.
        "rq1_type_match": None if skip_rq1_rq2 else step2.get("rq1_type_match"),
        "rq2_feature_present": None if skip_rq1_rq2 else step2.get("rq2_feature_present"),
        "rq3_description": step1.get("description"),
        "rq3_propositions": step1.get("propositions"),
        "rq3_hallucinations_count": step2.get("rq3_hallucinations_count"),
        "rq3_hallucinations_details": step2.get("rq3_hallucinations_details"),
        "rq3_ground_truth_source": "wikipedia" if wiki_summary else "model_memory",
    }


def sanitize_component(name):
    """Convert an entity or category into the same safe filename form used during generation."""
    cleaned = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    return cleaned.replace(" ", "_") or "unnamed"


def evaluate_one(model_name, model_dir, row, writer, out_f):
    """Evaluate one saved image and append its scores to the shared results CSV."""
    entity = row['Entity']
    folder_category = row['Category'] if row['Category'] != "Random" else f"Random_{row['Popularity_Tier']}"
    image_path = os.path.join(model_dir, sanitize_component(folder_category), f"{sanitize_component(entity)}.png")

    if not os.path.exists(image_path):
        logger.warning(f"Image not found for {entity} by {model_name}. Skipped.")
        return

    logger.info(f"Evaluating: [{entity}] from model [{model_name}]...")

    try:
        vqa_results = evaluate_image_with_judge(
            image_path=image_path,
            entity_name=entity,
            expected_type=row['Type'].strip(),
            expected_feature=row['Unique_Feature'],
            model_name=model_name,
        )

        result_row = {
            "Entity": entity,
            "Category": row['Category'],
            "Popularity_Tier": row['Popularity_Tier'],
            "Model": model_name,
            "RQ1_Predicted_Type": vqa_results.get("rq1_predicted_type"),
            "RQ1_Type_Match": vqa_results.get("rq1_type_match"),
            "RQ2_Feature_Present": vqa_results.get("rq2_feature_present"),
            "RQ3_Description": vqa_results.get("rq3_description"),
            "RQ3_Propositions": json.dumps(vqa_results.get("rq3_propositions"), ensure_ascii=False),
            "RQ3_Hallucinations_Count": vqa_results.get("rq3_hallucinations_count"),
            "RQ3_Details": vqa_results.get("rq3_hallucinations_details"),
            "RQ3_Ground_Truth_Source": vqa_results.get("rq3_ground_truth_source"),
        }

        # Only one worker may write/flush the CSV at a time.
        with _csv_lock:
            writer.writerow(result_row)
            out_f.flush()

    except Exception as e:
        logger.error(f"ERROR during evaluation of {entity} ({model_name}): {e}")
        log_failure(entity, model_name, e)

    finally:
        # Each image uses two judge calls, so every worker pauses before its next task.
        time.sleep(1.5)


def main():
    """Load the dataset, find unfinished image/model pairs, and evaluate them in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of images to evaluate in parallel",
    )
    args = parser.parse_args()

    if not SCADS_API_KEY:
        logger.error("SCADS_API_KEY environment variable is not set.")
        return

    input_csv = ENTITIES_CSV_PATH
    output_csv = str(BASE_DIR / f"evaluation_results_{_JUDGE_SUFFIX}.csv")
    base_images_dir = GENERATED_IMAGES_DIR

    # Read the entity metadata that tells us the expected type and feature.
    with open(input_csv, mode='r', encoding='utf-8') as f:
        entities = list(csv.DictReader(f))

    # Define the stable column order for the results CSV.
    fieldnames = [
        "Entity", "Category", "Popularity_Tier", "Model",
        "RQ1_Predicted_Type", "RQ1_Type_Match",
        "RQ2_Feature_Present",
        "RQ3_Description", "RQ3_Propositions", "RQ3_Hallucinations_Count", "RQ3_Details",
        "RQ3_Ground_Truth_Source"
    ]

    # Create the output file once, including its header, so later runs can append safely.
    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Remember completed entity/model pairs so restarting does not repeat paid API calls.
    evaluated_records = set()
    if os.path.exists(output_csv):
        with open(output_csv, mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                evaluated_records.add(f"{row['Entity']}_{row['Model']}")

    logger.info(f"Starting automated VQA evaluation pipeline with judge model: {JUDGE_MODEL}")
    logger.info(f"NOTE: each image costs 2 {JUDGE_MODEL} calls (blind classification + grounded fact-check), "
                "roughly 2x the time/cost of a single-call pipeline - budget accordingly.")

    if not os.path.exists(base_images_dir):
        logger.error(f"Directory {base_images_dir} not found. Generate images first.")
        return

    # Fetch reference summaries before starting the larger judge worker pool.
    prefetch_wiki_summaries(row['Entity'] for row in entities)

    # Build only the (model, image) pairs that still need evaluation.
    tasks = []
    for model_name in os.listdir(base_images_dir):
        model_dir = os.path.join(base_images_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for row in entities:
            unique_key = f"{row['Entity']}_{model_name}"
            if unique_key in evaluated_records:
                continue
            tasks.append((model_name, model_dir, row))

    logger.info(f"{len(tasks)} image(s) queued for evaluation with concurrency={args.concurrency}.")

    # Append results while a thread pool evaluates several images at once.
    with open(output_csv, mode='a', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(evaluate_one, model_name, model_dir, row, writer, out_f)
                for (model_name, model_dir, row) in tasks
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()  # re-raise any unexpected (uncaught) exception
                if done % 10 == 0 or done == len(futures):
                    logger.info(f"Progress: {done}/{len(futures)} images evaluated.")


if __name__ == "__main__":
    main()
