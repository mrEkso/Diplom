"""Compute summary statistics and charts from evaluation_results.csv.

RQ1 (type match) and RQ2 (unique feature present) are only meaningful for the
Curated entities - Random-tier entities have expected_type "Unknown" and are
recorded as RQ1_Type_Match/RQ2_Feature_Present = null (empty) on purpose (see
evaluate_images.py). Those rows must be dropped via .notna() before computing
accuracy, otherwise pandas/numpy treat the missing values as falsy and the
RQ1/RQ2 accuracy would be artificially deflated by the whole Random tier.

RQ3 (hallucination count) is defined for every row, including Random-tier ones.

Model coverage differs across models: rows that failed generation/evaluation
(content-policy refusals, timeouts, etc.) never make it into evaluation_results.csv
and are only visible in evaluation_failures.csv. Comparisons across models should
either restrict to entities present for all models or explicitly report N per model.

The Wikipedia-wide extrapolation is a rough illustration, NOT a statistically
sound estimate: the benchmark samples Famous/Medium/Long-tail entities in equal
counts (stratified sampling for testing), while the real tier distribution across
all of English Wikipedia is heavily skewed towards long-tail articles. Any figure
derived from EXTRAPOLATION_TIER_WEIGHTS below must be re-weighted with the true
tier proportions before being quoted as a real-world estimate.

Usage:
    python3 analyze_results.py
    python3 analyze_results.py --csv evaluation_results.csv --failures-csv evaluation_failures.csv
"""

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Build default paths from this file's location, so the command works from any folder.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = str(BASE_DIR.parent / "evaluation" / "evaluation_results.csv")
DEFAULT_FAILURES_CSV = str(BASE_DIR.parent / "generation" / "generation_failures.csv")
DEFAULT_ENTITIES_CSV = str(BASE_DIR.parent / "entities" / "benchmark_entities_final.csv")
OUT_DIR = BASE_DIR / "analysis_output"

# This equal weighting describes the benchmark sample, not real Wikipedia.
# The output using it is deliberately marked as caveated.
WIKIPEDIA_TOTAL_ARTICLES = 7_000_000
EXTRAPOLATION_TIER_WEIGHTS = {"Famous": 1 / 3, "Medium": 1 / 3, "Long-tail": 1 / 3}

# Approximate real-world proportions used for the final reweighted table.
REAL_WIKIPEDIA_TIER_WEIGHTS = {"Famous": 0.01, "Medium": 0.09, "Long-tail": 0.90}
TIER_ORDER = ["Famous", "Medium", "Long-tail"]


def summarize(df):
    """Print the main RQ1, RQ2, and RQ3 averages and return filtered RQ data."""
    # Show the basic size and model coverage before printing any averages.
    print(f"Total rows: {len(df)}")
    print(f"Models: {sorted(df['Model'].unique())}")
    print()

    # RQ1 and RQ2 are empty for Random-tier rows by design, so remove those rows first.
    rq1_df = df[df["RQ1_Type_Match"].notna()]
    rq2_df = df[df["RQ2_Feature_Present"].notna()]

    print(f"RQ1 (type match) - evaluated on {len(rq1_df)}/{len(df)} rows (Random-tier rows excluded):")
    print(rq1_df.groupby("Model")["RQ1_Type_Match"].mean().rename("accuracy"))
    print()

    print(f"RQ2 (feature present) - evaluated on {len(rq2_df)}/{len(df)} rows (Random-tier rows excluded):")
    print(rq2_df.groupby("Model")["RQ2_Feature_Present"].mean().rename("accuracy"))
    print()

    # RQ3 is defined for every image, including Random-tier images.
    print("RQ3 (mean hallucination count per image), by model:")
    print(df.groupby("Model")["RQ3_Hallucinations_Count"].mean().rename("mean_hallucinations"))
    print()

    print("RQ3 (mean hallucination count per image), by model and popularity tier:")
    print(df.groupby(["Model", "Popularity_Tier"])["RQ3_Hallucinations_Count"].mean())
    print()

    return rq1_df, rq2_df


def compute_tables(df, rq1_df, rq2_df, failures_csv, entities_csv):
    """Create available-case tables and an explicit missing-evaluation register."""
    # Create the output directory once, if it does not already exist.
    OUT_DIR.mkdir(exist_ok=True)

    # Count distinct entities with a result for each model.
    coverage = df.groupby("Model")["Entity"].nunique().rename("Entities_Scored").reset_index()
    coverage.to_csv(OUT_DIR / "model_coverage.csv", index=False)
    print("Model coverage (entities scored per model):\n", coverage, "\n")

    # Calculate RQ1 type-match rate and sample size for each category/model pair.
    rq1_cat = (
        rq1_df.groupby(["Category", "Model"])["RQ1_Type_Match"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Type_Match_Rate", "count": "N"})
        .reset_index()
    )
    rq1_cat.to_csv(OUT_DIR / "rq1_type_match_by_category_model.csv", index=False)

    # Calculate RQ2 feature-presence rate and sample size for each category/model pair.
    rq2_cat = (
        rq2_df.groupby(["Category", "Model"])["RQ2_Feature_Present"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Feature_Present_Rate", "count": "N"})
        .reset_index()
    )
    rq2_cat.to_csv(OUT_DIR / "rq2_feature_present_by_category_model.csv", index=False)

    # Count the propositions in each JSON-like CSV cell before averaging by tier.
    tiered = df[df["Popularity_Tier"].notna()].copy()
    tiered["N_Propositions"] = tiered["RQ3_Propositions"].apply(_count_propositions)
    rq3_tier = (
        tiered.groupby(["Popularity_Tier", "Model"])
        .agg(
            Avg_Hallucinations=("RQ3_Hallucinations_Count", "mean"),
            Avg_Propositions=("N_Propositions", "mean"),
            N=("Entity", "count"),
        )
        .reset_index()
    )
    rq3_tier.to_csv(OUT_DIR / "rq3_hallucinations_by_tier_model.csv", index=False)

    # Project the equal-count benchmark tiers across Wikipedia for illustration only.
    extrap = rq3_tier.copy()
    tier_weight = extrap["Popularity_Tier"].map(EXTRAPOLATION_TIER_WEIGHTS)
    extrap["Projected_Articles_In_Tier"] = tier_weight * WIKIPEDIA_TOTAL_ARTICLES
    extrap.to_csv(OUT_DIR / "rq3_extrapolation_CAVEATED.csv", index=False)

    if Path(failures_csv).exists():
        fail_df = pd.read_csv(failures_csv)
        fail_df["Failure_Type"] = fail_df["Error"].apply(_classify_failure)
        refusal_summary = (
            fail_df.groupby(["Model", "Failure_Type"])["Entity"]
            .nunique()
            .rename("Failed_Entities")
            .reset_index()
        )
        refusal_summary.to_csv(OUT_DIR / "refusals_summary.csv", index=False)
        print("Failures by model and type:\n", refusal_summary, "\n")
    else:
        fail_df = pd.DataFrame(columns=["Entity", "Model", "Error", "Failure_Type"])
        print(f"No {failures_csv} found - no failure rows available.")

    expected = pd.read_csv(entities_csv)
    failure_lookup = (
        fail_df.drop_duplicates(["Entity", "Model"])
        .set_index(["Entity", "Model"])[["Failure_Type", "Error"]]
        if not fail_df.empty
        else pd.DataFrame(columns=["Failure_Type", "Error"])
    )
    missing_rows = []
    for model in sorted(df["Model"].unique()):
        scored = set(df.loc[df["Model"] == model, "Entity"])
        for _, entity_row in expected.iterrows():
            entity = entity_row["Entity"]
            if entity in scored:
                continue
            failure = failure_lookup.loc[(entity, model)] if (entity, model) in failure_lookup.index else None
            missing_rows.append(
                {
                    "Entity": entity,
                    "Category": entity_row.get("Category"),
                    "Popularity_Tier": entity_row.get("Popularity_Tier"),
                    "Model": model,
                    "Status": "Missing evaluation",
                    "Failure_Type": failure["Failure_Type"] if failure is not None else "Unknown",
                    "Failure_Detail": failure["Error"] if failure is not None else "No failure record found",
                }
            )
    pd.DataFrame(missing_rows).to_csv(OUT_DIR / "missing_evaluations.csv", index=False)

    print(f"All summary tables written to {OUT_DIR}/")
    return rq1_cat, rq2_cat, rq3_tier


def _classify_failure(error):
    """Group generation failures into refusal and non-refusal categories."""
    text = str(error).lower()
    if "safety system" in text or "content-policy" in text or "empty content" in text:
        return "Safety refusal"
    return "Other failure"


def compute_fair_comparison(df, entities_csv):
    """Compare models only on entity sets present for every model."""
    expected = pd.read_csv(entities_csv)
    model_entities = {
        model: set(df.loc[df["Model"] == model, "Entity"])
        for model in sorted(df["Model"].unique())
    }
    common_entities = set.intersection(*model_entities.values())
    fair_df = df[df["Entity"].isin(common_entities)].copy()
    rq_df = fair_df[fair_df["RQ1_Type_Match"].notna()]
    rows = []
    for model, group in fair_df.groupby("Model"):
        rq_group = rq_df[rq_df["Model"] == model]
        rows.append(
            {
                "Model": model,
                "Common_Entities": len(common_entities),
                "RQ1_N": len(rq_group),
                "RQ1_Type_Match_Rate": rq_group["RQ1_Type_Match"].mean(),
                "RQ2_N": rq_group["RQ2_Feature_Present"].notna().sum(),
                "RQ2_Feature_Present_Rate": rq_group["RQ2_Feature_Present"].mean(),
                "RQ3_N": len(group),
                "RQ3_Avg_Hallucinations": group["RQ3_Hallucinations_Count"].mean(),
                "RQ3_Avg_Propositions": group["N_Propositions"].mean()
                if "N_Propositions" in group
                else group["RQ3_Propositions"].apply(_count_propositions).mean(),
                "RQ3_Hallucination_Rate": group["RQ3_Hallucinations_Count"].sum()
                / group["RQ3_Propositions"].apply(_count_propositions).sum(),
            }
        )
    fair_summary = pd.DataFrame(rows)
    fair_summary.to_csv(OUT_DIR / "fair_comparison_overall.csv", index=False)

    fair_by_category = (
        fair_df.groupby(["Category", "Model"])
        .agg(
            N=("Entity", "count"),
            RQ1_Type_Match_Rate=("RQ1_Type_Match", "mean"),
            RQ2_Feature_Present_Rate=("RQ2_Feature_Present", "mean"),
            Avg_Hallucinations=("RQ3_Hallucinations_Count", "mean"),
        )
        .reset_index()
    )
    fair_by_category.to_csv(OUT_DIR / "fair_comparison_by_category.csv", index=False)
    print(f"Fair comparison uses {len(common_entities)} common entities.")
    return fair_df, expected


def compute_fact_analysis(df, output_suffix=""):
    """Measure how many propositions models generate and how that relates to errors."""
    fact_df = df.copy()
    fact_df["N_Propositions"] = fact_df["RQ3_Propositions"].apply(_count_propositions)
    fact_df["Image_Hallucination_Rate"] = (
        fact_df["RQ3_Hallucinations_Count"] / fact_df["N_Propositions"].replace(0, pd.NA)
    )
    by_tier = (
        fact_df[fact_df["Popularity_Tier"].notna()]
        .groupby(["Popularity_Tier", "Model"])
        .agg(
            N_Images=("Entity", "count"),
            Avg_Propositions=("N_Propositions", "mean"),
            Total_Propositions=("N_Propositions", "sum"),
            Total_Hallucinations=("RQ3_Hallucinations_Count", "sum"),
            Avg_Hallucination_Rate=("Image_Hallucination_Rate", "mean"),
        )
        .reset_index()
    )
    by_tier["Hallucination_Rate"] = (
        by_tier["Total_Hallucinations"] / by_tier["Total_Propositions"]
    )
    by_tier.to_csv(OUT_DIR / f"rq3_fact_generation_by_tier_model{output_suffix}.csv", index=False)

    correlations = []
    for model, group in fact_df.groupby("Model"):
        ranked_props = group["N_Propositions"].rank()
        ranked_hallucinations = group["RQ3_Hallucinations_Count"].rank()
        ranked_rates = group["Image_Hallucination_Rate"].rank()
        correlations.append(
            {
                "Model": model,
                "N_Images": len(group),
                "Spearman_Props_vs_Hallucinations": ranked_props.corr(ranked_hallucinations),
                "Spearman_Props_vs_Image_Hallucination_Rate": ranked_props.corr(ranked_rates),
            }
        )
    correlations = pd.DataFrame(correlations)
    correlations.to_csv(OUT_DIR / f"rq3_fact_count_correlations{output_suffix}.csv", index=False)

    pivot = by_tier.pivot(index="Popularity_Tier", columns="Model", values="Avg_Propositions")
    pivot = pivot.reindex(TIER_ORDER)
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Average Number of Generated Facts by Popularity Tier")
    ax.set_ylabel("Average propositions per image")
    ax.set_xlabel("Popularity tier")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"rq3_average_facts_by_tier{output_suffix}.png", dpi=150)
    plt.close()
    return fact_df


def write_limitations(df, fair_df, failures_csv, entities_csv):
    """Write the current missing-evaluation and interpretation caveats."""
    expected_count = pd.read_csv(entities_csv).shape[0]
    missing_count = expected_count * df["Model"].nunique() - len(df)
    failure_source = Path(failures_csv).name
    text = f"""# Analysis limitations

- The complete benchmark contains {expected_count} entities per model. The current result file contains {len(df)} successful evaluations, leaving {missing_count} missing entity/model evaluations.
- The missing evaluations are not scored as failures. They are listed in `missing_evaluations.csv`; the current failure source is `{failure_source}`.
- The current failure audit identifies nine safety refusals, all for `gpt_image_2`. This creates unequal coverage: 289 entities for `gpt_image_2` versus 298 for each Flux model.
- Fair model comparisons use the {fair_df["Entity"].nunique()} entities available for every model. RQ1 and RQ2 use only the common curated subset; Random-tier rows are excluded because their reference type and feature are undefined.
- Hallucination rate depends on how many propositions a model generates. Proposition-count tables and correlations are therefore reported alongside the raw hallucination counts; a high or low rate should not be interpreted independently of verbosity.
- Wikipedia lead summaries are the factual reference, so they may omit visible details. This is a limitation of the evaluation ground truth, not evidence that every unmentioned visual detail is objectively false.
- The reweighted Wikipedia extrapolation is illustrative only and should not be presented as a measured count of facts known by a model across Wikipedia.
"""
    (OUT_DIR / "limitations.md").write_text(text, encoding="utf-8")


def _count_propositions(cell):
    """Return the number of propositions in one serialized list, or zero if invalid."""
    try:
        return len(ast.literal_eval(cell)) if isinstance(cell, str) else 0
    except (ValueError, SyntaxError):
        return 0


def compute_hallucination_rate(df):
    """Calculate hallucinations divided by all propositions, overall and by tier.

    This rate is fairer than a raw hallucination count when models describe
    different numbers of visual facts per image.
    """
    # Work on a copy so adding the helper column does not modify the caller's DataFrame.
    df = df.copy()
    df["N_Propositions"] = df["RQ3_Propositions"].apply(_count_propositions)

    # First calculate one rate per model across all available images.
    overall = (
        df.groupby("Model")
        .apply(
            lambda g: g["RQ3_Hallucinations_Count"].sum() / g["N_Propositions"].sum(),
            include_groups=False,
        )
        .rename("Hallucination_Rate")
        .reset_index()
    )
    overall.to_csv(OUT_DIR / "rq3_hallucination_rate_by_model.csv", index=False)
    print("RQ3 - Hallucination rate (hallucinations / total propositions), by model:\n", overall, "\n")

    # Then repeat the same calculation separately for each popularity tier.
    by_tier = (
        df[df["Popularity_Tier"].notna()]
        .groupby(["Popularity_Tier", "Model"])
        .apply(
            lambda g: g["RQ3_Hallucinations_Count"].sum() / g["N_Propositions"].sum(),
            include_groups=False,
        )
        .rename("Hallucination_Rate")
        .reset_index()
    )
    by_tier.to_csv(OUT_DIR / "rq3_hallucination_rate_by_tier_model.csv", index=False)
    return overall, by_tier


def compute_reweighted_extrapolation(rq3_tier):
    """Reweight correct facts using approximate real Wikipedia tier proportions.

    The benchmark deliberately has equal tier sizes, so this correction prevents
    the final illustration from treating Famous, Medium, and Long-tail topics as
    equally common in Wikipedia.
    """
    rq3_tier = rq3_tier.copy()
    # A proposition is counted as correct when it is not classified as a hallucination.
    rq3_tier["Avg_Correct_Facts"] = (rq3_tier["Avg_Propositions"] - rq3_tier["Avg_Hallucinations"]).clip(lower=0)

    # Build one projected summary row for every model.
    rows = []
    for model, g in rq3_tier.groupby("Model"):
        weights = g["Popularity_Tier"].map(REAL_WIKIPEDIA_TIER_WEIGHTS)
        avg_correct_facts = (g["Avg_Correct_Facts"] * weights).sum() / weights.sum()
        rows.append(
            {
                "Model": model,
                "Avg_Correct_Facts_Per_Random_Entity": avg_correct_facts,
                "Projected_Known_Facts_Across_Wikipedia": avg_correct_facts * WIKIPEDIA_TOTAL_ARTICLES,
            }
        )
    reweighted = pd.DataFrame(rows)
    reweighted.to_csv(OUT_DIR / "rq3_extrapolation_reweighted.csv", index=False)
    print("RQ3 - Reweighted extrapolation (real Wikipedia tier proportions):\n", reweighted, "\n")
    return reweighted


def plot_grouped_bar(table, value_col, category_col, title, ylabel, out_name):
    """Save a grouped bar chart for a score split by category and model."""
    # Pivot makes categories the x-axis and models the separate bars.
    pivot = table.pivot(index=category_col, columns="Model", values=value_col)
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(category_col)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=150)
    plt.close()


def plot_hallucination_tax_line(rq3_tier):
    """Save a line chart showing how hallucinations change across popularity tiers."""
    pivot = rq3_tier.pivot(index="Popularity_Tier", columns="Model", values="Avg_Hallucinations")
    pivot = pivot.reindex(TIER_ORDER)
    ax = pivot.plot(kind="line", marker="o", figsize=(9, 5))
    ax.set_title("Factuality Tax: Hallucinations by Entity Popularity Tier")
    ax.set_ylabel("Avg. hallucinations per image")
    ax.set_xlabel("Popularity tier")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rq3_hallucination_tax_by_tier.png", dpi=150)
    plt.close()


def main():
    """Read input data, create all tables, and save all analysis charts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_RESULTS_CSV, help="Path to evaluation_results.csv")
    parser.add_argument(
        "--failures-csv", default=DEFAULT_FAILURES_CSV, help="Path to evaluation_failures.csv"
    )
    parser.add_argument(
        "--entities-csv", default=DEFAULT_ENTITIES_CSV, help="Path to the complete benchmark entity list"
    )
    args = parser.parse_args()

    # Read the successful evaluation rows into a pandas table.
    df = pd.read_csv(args.csv)
    # Produce printed summaries and the filtered data needed by later calculations.
    rq1_df, rq2_df = summarize(df)
    # Save detailed CSV tables and receive the tier-level RQ3 table for later steps.
    rq1_cat, rq2_cat, rq3_tier = compute_tables(
        df, rq1_df, rq2_df, args.failures_csv, args.entities_csv
    )
    # Save hallucination rates and the real-world-weighted illustration.
    compute_hallucination_rate(df)
    compute_reweighted_extrapolation(rq3_tier)
    fair_df, _ = compute_fair_comparison(df, args.entities_csv)
    compute_fact_analysis(df)
    compute_fact_analysis(fair_df, "_fair")
    write_limitations(df, fair_df, args.failures_csv, args.entities_csv)

    plot_grouped_bar(
        rq1_cat, "Type_Match_Rate", "Category",
        "RQ1: Object Type Prediction Accuracy by Category", "Type match rate",
        "rq1_type_match_by_category.png",
    )
    plot_grouped_bar(
        rq2_cat, "Feature_Present_Rate", "Category",
        "RQ2: Descriptive Feature Presence by Category", "Feature present rate",
        "rq2_feature_present_by_category.png",
    )
    plot_hallucination_tax_line(rq3_tier)
    print(f"Charts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
