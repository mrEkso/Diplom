# T2I Factuality Benchmark

This repository contains the code, benchmark data, aggregate results, and
generated images for a bachelor's thesis evaluating factual knowledge in
text-to-image models.

## Contents

- `entities/`: curated and random benchmark entity lists.
- `generation/`: image-generation pipeline and generated benchmark images.
- `evaluation/`: blind visual evaluation and proposition-level factuality checks.
- `analysing/`: reproducible analysis code and aggregate tables/figures.

The generated PNG files are tracked with Git LFS because the image collection is
approximately 2.2 GB. Clone this repository with Git LFS enabled to retrieve the
images.

## Models

The benchmark supports GPT Image 2, FLUX.2-dev, and FLUX.2-klein-4B. All models
receive the same zero-context prompt: `An image of [entity].`

## Reproduction

Install the Python dependencies used by the scripts, copy `.env_example` to
`.env`, and provide local API credentials through environment variables. Never
commit `.env` or provider logs.

The aggregate analysis can be regenerated with:

```bash
python3 analysing/analyze_results.py \
  --csv evaluation/evaluation_results_gemma-4-31B-it.csv
```

## Data and usage note

The generated images are research artifacts produced by external image models.
Their redistribution and reuse may be subject to provider terms, copyright,
and other restrictions. The repository is intended for academic research and
benchmark reproduction.