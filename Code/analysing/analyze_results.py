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
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Build default paths from this file's location, so the command works from any folder.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = str(BASE_DIR.parent / "evaluation" / "evaluation_results_gemma-4-31B-it.csv")
DEFAULT_FAILURES_CSV = str(BASE_DIR.parent / "generation" / "generation_failures.csv")
DEFAULT_ENTITIES_CSV = str(BASE_DIR.parent / "entities" / "benchmark_entities_final.csv")
OUT_DIR = BASE_DIR / "analysis_output"

# This equal weighting describes the benchmark sample, not real Wikipedia.
# The output using it is deliberately marked as caveated and intentionally uses the
# same fact-count metric as the reweighted projection for fair comparison.
WIKIPEDIA_TOTAL_ARTICLES = 7_237_727
EXTRAPOLATION_TIER_WEIGHTS = {"Famous": 1 / 3, "Medium": 1 / 3, "Long-tail": 1 / 3}

# Real-world proportions used for the final reweighted table. These values are
# derived from the Wikipedia article distribution (Famous, Medium, Long-tail).
REAL_WIKIPEDIA_TIER_WEIGHTS = {"Famous": 0.0032, "Medium": 0.0535, "Long-tail": 0.9433}
TIER_ORDER = ["Famous", "Medium", "Long-tail"]


def wilson_interval(successes, trials, z=1.96):
    """Return a 95% Wilson interval for a binomial proportion."""
    if trials == 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return centre - margin, centre + margin


def two_proportion_z_test(successes_a, trials_a, successes_b, trials_b):
    """Return z and two-sided normal-approximation p for two proportions."""
    proportion_a = successes_a / trials_a
    proportion_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    standard_error = math.sqrt(
        pooled * (1 - pooled) * (1 / trials_a + 1 / trials_b)
    )
    if standard_error == 0:
        return float("nan"), float("nan")
    z_score = (proportion_a - proportion_b) / standard_error
    p_value = math.erfc(abs(z_score) / math.sqrt(2))
    return z_score, p_value


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

    # Project the equal-count benchmark tiers across Wikipedia for illustration only,
    # using the same fact-count metric as the reweighted estimate so the tables are directly comparable.
    extrap = rq3_tier.copy()
    extrap["Avg_Correct_Facts"] = (extrap["Avg_Propositions"] - extrap["Avg_Hallucinations"]).clip(lower=0)
    naive_rows = []
    for model, g in extrap.groupby("Model"):
        weights = g["Popularity_Tier"].map(EXTRAPOLATION_TIER_WEIGHTS)
        avg_correct = (g["Avg_Correct_Facts"] * weights).sum() / weights.sum()
        naive_rows.append(
            {
                "Model": model,
                "Extrapolation_Method": "naive_equal_tier",
                "Avg_Correct_Facts_Per_Entity": avg_correct,
                "Projected_Known_Facts_Across_Wikipedia": avg_correct * WIKIPEDIA_TOTAL_ARTICLES,
                "Extrapolation_Set": "available-case",
            }
        )
    pd.DataFrame(naive_rows).to_csv(OUT_DIR / "rq3_extrapolation_CAVEATED.csv", index=False)

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
        rq1_successes = int(rq_group["RQ1_Type_Match"].sum())
        rq2_group = rq_group[rq_group["RQ2_Feature_Present"].notna()]
        rq2_successes = int(rq2_group["RQ2_Feature_Present"].sum())
        proposition_counts = group["RQ3_Propositions"].apply(_count_propositions)
        total_propositions = int(proposition_counts.sum())
        total_hallucinations = int(group["RQ3_Hallucinations_Count"].sum())
        rq1_lower, rq1_upper = wilson_interval(rq1_successes, len(rq_group))
        rq2_lower, rq2_upper = wilson_interval(rq2_successes, len(rq2_group))
        rq3_lower, rq3_upper = wilson_interval(
            total_hallucinations, total_propositions
        )
        rows.append(
            {
                "Model": model,
                "Common_Entities": len(common_entities),
                "RQ1_N": len(rq_group),
                "RQ1_Successes": rq1_successes,
                "RQ1_Type_Match_Rate": rq_group["RQ1_Type_Match"].mean(),
                "RQ1_Wilson_Lower": rq1_lower,
                "RQ1_Wilson_Upper": rq1_upper,
                "RQ2_N": rq_group["RQ2_Feature_Present"].notna().sum(),
                "RQ2_Successes": rq2_successes,
                "RQ2_Feature_Present_Rate": rq_group["RQ2_Feature_Present"].mean(),
                "RQ2_Wilson_Lower": rq2_lower,
                "RQ2_Wilson_Upper": rq2_upper,
                "RQ3_N": len(group),
                "RQ3_Total_Propositions": total_propositions,
                "RQ3_Total_Hallucinations": total_hallucinations,
                "RQ3_Avg_Hallucinations": group["RQ3_Hallucinations_Count"].mean(),
                "RQ3_Avg_Propositions": group["N_Propositions"].mean()
                if "N_Propositions" in group
                else group["RQ3_Propositions"].apply(_count_propositions).mean(),
                "RQ3_Hallucination_Rate": group["RQ3_Hallucinations_Count"].sum()
                / group["RQ3_Propositions"].apply(_count_propositions).sum(),
                "RQ3_Wilson_Lower": rq3_lower,
                "RQ3_Wilson_Upper": rq3_upper,
            }
        )
    fair_summary = pd.DataFrame(rows)
    fair_summary.to_csv(OUT_DIR / "fair_comparison_overall.csv", index=False)

    tests = []
    model_labels = {
        "gpt_image_2": "GPT Image 2",
        "scads_flux2_dev": "FLUX.2-dev",
        "scads_flux2_klein": "FLUX.2-klein-4B",
    }
    available_models = sorted(fair_summary["Model"].unique())
    for metric, success_column, n_column in [
        ("RQ1_Type_Match", "RQ1_Successes", "RQ1_N"),
        ("RQ2_Feature_Present", "RQ2_Successes", "RQ2_N"),
    ]:
        for model_a, model_b in combinations(available_models, 2):
            row_a = fair_summary[fair_summary["Model"] == model_a].iloc[0]
            row_b = fair_summary[fair_summary["Model"] == model_b].iloc[0]
            z_score, p_value = two_proportion_z_test(
                int(row_a[success_column]), int(row_a[n_column]),
                int(row_b[success_column]), int(row_b[n_column]),
            )
            tests.append(
                {
                    "Comparison": f"{model_labels[model_a]} vs {model_labels[model_b]}",
                    "Metric": metric,
                    "Z": z_score,
                    "P_Two_Sided": p_value,
                    "Significant_At_0.05": p_value < 0.05,
                }
            )
    pd.DataFrame(tests).to_csv(OUT_DIR / "fair_statistical_tests.csv", index=False)
    print("Pairwise two-proportion tests:\n", pd.DataFrame(tests), "\n")

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
    intervals = by_tier.apply(
        lambda row: wilson_interval(
            int(row["Total_Propositions"] - row["Total_Hallucinations"]),
            int(row["Total_Propositions"]),
        ),
        axis=1,
    )
    by_tier["Hallucination_Wilson_Lower"] = [1 - item[1] for item in intervals]
    by_tier["Hallucination_Wilson_Upper"] = [1 - item[0] for item in intervals]
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
    return by_tier


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
- The reweighted Wikipedia extrapolation is illustrative only and should not be presented as a measured count of facts known by a model across Wikipedia. It is computed on the available-case per-model coverage and is therefore not a fair-set estimate; the fair comparison uses the common entity subset instead.
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

    This projection uses the available-case model coverage and should be read as an
    illustrative population-level scaling exercise, not as a fair-set estimate.
    The underlying benchmark sample is deliberately tier-balanced, so this correction
    prevents Famous, Medium, and Long-tail topics from being treated as equally
    common in English Wikipedia.
    """
    rq3_tier = rq3_tier.copy()
    # A proposition is counted as correct when it is not classified as a hallucination.
    rq3_tier["Avg_Correct_Facts"] = (rq3_tier["Avg_Propositions"] - rq3_tier["Avg_Hallucinations"]).clip(lower=0)

    # Build one projected summary row for every model, using the same "total known facts"
    # metric as the naive extrapolation so the two estimates remain directly comparable.
    rows = []
    for model, g in rq3_tier.groupby("Model"):
        weights = g["Popularity_Tier"].map(REAL_WIKIPEDIA_TIER_WEIGHTS)
        avg_correct_facts = (g["Avg_Correct_Facts"] * weights).sum() / weights.sum()
        rows.append(
            {
                "Model": model,
                "Extrapolation_Method": "reweighted_wikipedia",
                "Avg_Correct_Facts_Per_Entity": avg_correct_facts,
                "Projected_Known_Facts_Across_Wikipedia": avg_correct_facts * WIKIPEDIA_TOTAL_ARTICLES,
                "Extrapolation_Set": "available-case",
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
    """Save a line chart showing normalized hallucination rates by tier."""
    pivot = rq3_tier.pivot(index="Popularity_Tier", columns="Model", values="Hallucination_Rate")
    pivot = pivot.reindex(TIER_ORDER)
    ax = pivot.plot(kind="line", marker="o", figsize=(9, 5))
    ax.set_title("Hallucination Rate by Entity Popularity Tier")
    ax.set_ylabel("Hallucination rate")
    ax.set_xlabel("Popularity tier")
    ax.set_ylim(0, 0.30)
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
    fair_fact_by_tier = compute_fact_analysis(fair_df, "_fair")
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
    plot_hallucination_tax_line(fair_fact_by_tier)
    print(f"Charts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
