"""Approximate breakdown of RQ3 propositions by kind (used in Section "What Counts as a Visual Fact").

The rule is a simple keyword match, so the shares are approximate:
- "text":   the proposition transcribes text written in the image
- "scene":  the proposition describes background, lighting, weather, or the surface
- "entity": everything else, i.e. statements about the depicted object itself

The script also reports
- how often the entity name itself is written into the image (a leak into the blind step), and
- how many images have every proposition judged false (complete failures).

All numbers use the fair common set (entities available for all three models).

Run:
    python3 proposition_kinds.py evaluation_results_gemma-4-31B-it.csv
"""

import json
import re
import sys

import pandas as pd

TEXT_RE = re.compile(r"\btext\b|reads|written|inscri|label|caption|title|'[^']+'|\"", re.I)
SCENE_RE = re.compile(
    r"\bsky\b|background|foreground|lighting|sunlight|cloud|blurred|grass|\btrees?\b|floor|surface|table",
    re.I,
)
# Entities whose Wikipedia reference was resolved to the wrong article (see Limitations).
WRONG_REFERENCE = {"Montezuma"}


def kind(prop):
    if TEXT_RE.search(prop):
        return "text"
    if SCENE_RE.search(prop):
        return "scene"
    return "entity"


def entity_name_rendered(entity, props):
    """True if a text proposition contains a word (>3 letters) of the entity name."""
    words = [w for w in re.findall(r"[a-z]+", entity.lower().split("(")[0]) if len(w) > 3]
    return any(TEXT_RE.search(p) and any(w in p.lower() for w in words) for p in props) if words else False


def main(path):
    df = pd.read_csv(path)
    df["props"] = df["RQ3_Propositions"].apply(json.loads)
    df["P"] = df["props"].apply(len)

    models = df["Model"].unique()
    common = set.intersection(*(set(df.loc[df["Model"] == m, "Entity"]) for m in models))
    df = df[df["Entity"].isin(common)].copy()
    print(f"Common set: {len(common)} entities, {len(df)} images, {df['P'].sum()} propositions\n")

    rows = [(m, kind(p)) for m, props in zip(df["Model"], df["props"]) for p in props]
    kinds = pd.DataFrame(rows, columns=["Model", "Kind"])
    print("Share of propositions by kind:")
    print(pd.crosstab(kinds["Model"], kinds["Kind"], normalize="index").round(3), "\n")

    df["name_rendered"] = [entity_name_rendered(e, p) for e, p in zip(df["Entity"], df["props"])]
    print("Share of images in which the entity name is written into the image:")
    print(df.groupby("Model")["name_rendered"].mean().round(3), "\n")

    ok = ~df["Entity"].isin(WRONG_REFERENCE)
    complete = ok & (df["P"] > 0) & (df["RQ3_Hallucinations_Count"] >= df["P"])
    person = complete & (df["RQ1_Predicted_Type"].str.lower() == "person")
    print(f"Complete failures (every proposition false, excluding {sorted(WRONG_REFERENCE)}): "
          f"{complete.sum()} of {ok.sum()} images; {person.sum()} of them depict a person")
    print(df[complete].groupby("Model").size())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation_results_gemma-4-31B-it.csv")

