"""Build a list of random Wikipedia articles, sorted into 3 "popularity" groups.

What this script does, step by step:
1. Ask Wikipedia for random article titles.
2. Check how long (in characters) each article is - a longer article usually
   means a more famous/well-documented topic.
3. Sort each article into one of 3 buckets ("Famous", "Medium", "Long-tail")
   based on its length.
4. Keep asking Wikipedia for more random articles until every bucket has
   enough items.
5. Save everything into a CSV file called benchmark_entities.csv.

Run it with:
    python3 fetch_random_wikipedia_entities.py
"""

import csv
import time
from pathlib import Path

import requests

# This makes sure the output file is always saved next to this script,
# no matter which folder you were in when you ran the command.
BASE_DIR = Path(__file__).resolve().parent

# Wikipedia asks every automated tool (bot) to identify itself with contact
# details in this header. Bots that don't do this get slowed down a lot
# (limited to only 10 requests per minute instead of much more).
# https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits
HEADERS = {"User-Agent": "DiplomaResearchBot/1.0 (davyd.okaianchenko@tu-dresden.de; TU Dresden bachelor thesis)"}

# The 3 popularity groups we want to fill, defined by article length in
# characters: (minimum length, maximum length). "None" means "no upper limit".
TIERS = {
    "Famous": (100_000, None),      # very long articles = very famous topics
    "Medium": (20_000, 100_000),    # medium-length articles
    "Long-tail": (500, 20_000),     # short articles = obscure/niche topics
}
# How many articles we want to collect for each of the 3 groups above.
ITEMS_PER_TIER = 66


def classify_tier(length):
    """Look at an article's length and decide which popularity group it belongs to.

    Returns the group name ("Famous", "Medium", or "Long-tail"), or None if the
    article is too short/long to fit any group.
    """
    for tier_name, (low, high) in TIERS.items():
        if length >= low and (high is None or length < high):
            return tier_name
    return None


def fetch_random_wikipedia_entities(items_per_tier=ITEMS_PER_TIER):
    """Keep fetching random Wikipedia articles until all 3 groups are full.

    Returns a dictionary like: {"Famous": [...], "Medium": [...], "Long-tail": [...]}
    where each list holds dicts of {"Entity": title, "Length": article length}.
    """
    url = "https://en.wikipedia.org/w/api.php"
    # Start with 3 empty baskets, one per popularity group.
    buckets = {tier_name: [] for tier_name in TIERS}
    # Remembers article titles we've already collected, so we never add the same one twice.
    seen_titles = set()

    print("Fetching random entities from Wikipedia API...")

    def buckets_full():
        """True once every basket has reached its target size."""
        return all(len(items) >= items_per_tier for items in buckets.values())

    # Keep looping - fetch a new random batch of articles - until all baskets are full.
    while not buckets_full():
        # Step 1: ask Wikipedia for a batch of random article titles (50 articles at a time).
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,  # Only main articles
            "rnlimit": 50,
        }
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=10).json()
        except requests.RequestException as e:
            print(f"Request failed, retrying: {e}")
            time.sleep(2)
            continue

        pages = res.get("query", {}).get("random", [])
        if not pages:
            continue

        # Step 2: ask Wikipedia how long each of those articles is.
        page_ids = "|".join(str(p["id"]) for p in pages)
        info_params = {
            "action": "query",
            "format": "json",
            "pageids": page_ids,
            "prop": "info",
        }
        try:
            info_res = requests.get(url, params=info_params, headers=HEADERS, timeout=10).json()
        except requests.RequestException as e:
            print(f"Request failed, retrying: {e}")
            time.sleep(2)
            continue

        page_info = info_res.get("query", {}).get("pages", {})

        # Step 3: sort each article into its popularity bucket, skipping junk/duplicates.
        for pid, pdata in page_info.items():
            title = pdata.get("title", "")
            length = pdata.get("length", 0)

            # Skip articles we already have, and skip "list" or "disambiguation"
            # pages since those aren't real single entities.
            if title in seen_titles or "List of" in title or "disambiguation" in title:
                continue

            tier_name = classify_tier(length)
            if tier_name is None or len(buckets[tier_name]) >= items_per_tier:
                continue

            buckets[tier_name].append({"Entity": title, "Length": length})
            seen_titles.add(title)

        # Show progress so we can see the buckets filling up while the script runs.
        counts = ", ".join(f"{t}={len(v)}/{items_per_tier}" for t, v in buckets.items())
        print(f"  Progress: {counts}")
        # Small pause between requests to be polite to Wikipedia's servers :)
        time.sleep(0.5)

    return buckets


def main():
    """Fetch the random entities and save them all into benchmark_entities.csv."""
    buckets = fetch_random_wikipedia_entities()

    # Flatten the 3 buckets into one single list of rows, ready for the CSV file.
    all_rows = []
    for tier_name, items in buckets.items():
        for item in items:
            all_rows.append(
                {
                    "Entity": item["Entity"],
                    "Category": "Random",
                    "Type": "Unknown",  # Will be determined by modeling in RQ1
                    "Unique_Feature": "N/A",
                    "Popularity_Tier": tier_name,
                    "Wiki_Length": item["Length"],
                }
            )

    # Write every row out to a CSV file so it can be used by the other scripts.
    filename = str(BASE_DIR / "benchmark_entities.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Entity",
                "Category",
                "Type",
                "Unique_Feature",
                "Popularity_Tier",
                "Wiki_Length",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"Successfully created dataset in file {filename}! Total records: {len(all_rows)}"
    )


if __name__ == "__main__":
    main()