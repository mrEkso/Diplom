# Measuring Factual Knowledge in Text-to-Image Models

Research code and aggregate benchmark results for a bachelor's thesis on the
factual knowledge of text-to-image (T2I) models.

## Research Questions

1. For how many real-world entities can T2I models correctly predict the entity type?
2. For how many entities do they generate distinctive descriptive features?
3. Of the facts expressed in generated images, how many are true?

## Benchmark

The benchmark contains 298 entities:

- 100 curated entities from four domains: landmarks and architecture, historical figures, rare flora and fauna, and cultural artifacts and foods.
- 198 randomly sampled entities split into Famous, Medium, and Long-tail popularity tiers.

All models receive the same zero-context prompt: `An image of [entity].`
Curated entities have type and distinctive-feature annotations. Random entities
are used for proposition-level factuality analysis.

## Models and Evaluation

The generation pipeline supports GPT Image 2, FLUX.2-dev, and FLUX.2-klein-4B.
Images are evaluated by a blind visual question-answering pipeline. The
evaluator records type match, distinctive-feature presence, generated atomic
propositions, and hallucinations against Wikipedia lead summaries.

The main comparison uses the 289 entities available for every model. Nine GPT
Image 2 generations were refused by the provider safety system and are reported
as missing evaluations rather than scored as failures.

## Repository Layout

```text
Code/entities/       Benchmark entity lists and sampling utilities
Code/generation/     Image-generation pipeline
Code/evaluation/     Blind VQA and proposition evaluation
Code/analysing/      Reproducible aggregate analysis and result tables
Template/            LaTeX thesis source
```

Generated images, API logs, local caches, thesis drafts, and reference PDFs are
excluded from the public repository. Aggregate CSV tables and analysis figures
are included as research artifacts.

## Reproduction

Use Python 3.9+ and install the required packages used by the scripts, including
`pandas`, `matplotlib`, `requests`, `python-dotenv`, and `openai`.

1. Copy `Code/.env_example` to `Code/.env` and fill in local API credentials.
2. Run the generation script to create images.
3. Run the evaluation script with the desired judge model.
4. Run the analysis script:

```bash
python3 Code/analysing/analyze_results.py \
  --csv Code/evaluation/evaluation_results_gemma-4-31B-it.csv
```

The analysis writes coverage, fair-comparison, refusal, missing-evaluation,
proposition-count, hallucination-rate, and extrapolation tables to
`Code/analysing/analysis_output/`.

## Security and Data Policy

Never commit `Code/.env`, API keys, provider logs, or generated images. API
credentials must be supplied through environment variables. If a credential is
ever exposed, revoke it and create a replacement before continuing.

The benchmark and aggregate outputs are provided for research and thesis
reproduction. Generated images and provider responses may be subject to model,
service, copyright, and usage restrictions and are therefore not redistributed
here.

## Thesis

The thesis source is in `Template/`. The compiled PDF is intentionally omitted
from the public code repository; the source can be built with `latexmk` in an
environment containing the TU Dresden LaTeX template dependencies.