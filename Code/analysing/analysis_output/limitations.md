# Analysis limitations

- The complete benchmark contains 298 entities per model. The current result file contains 885 successful evaluations, leaving 9 missing entity/model evaluations.
- The missing evaluations are not scored as failures. They are listed in `missing_evaluations.csv`; the current failure source is `generation_failures.csv`.
- The current failure audit identifies nine safety refusals, all for `gpt_image_2`. This creates unequal coverage: 289 entities for `gpt_image_2` versus 298 for each Flux model.
- Fair model comparisons use the 289 entities available for every model. RQ1 and RQ2 use only the common curated subset; Random-tier rows are excluded because their reference type and feature are undefined.
- Hallucination rate depends on how many propositions a model generates. Proposition-count tables and correlations are therefore reported alongside the raw hallucination counts; a high or low rate should not be interpreted independently of verbosity.
- Wikipedia lead summaries are the factual reference, so they may omit visible details. This is a limitation of the evaluation ground truth, not evidence that every unmentioned visual detail is objectively false.
- The reweighted Wikipedia extrapolation is illustrative only and should not be presented as a measured count of facts known by a model across Wikipedia.
