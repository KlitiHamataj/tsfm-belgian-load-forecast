---
id: timestral-209m-nosym
family: timestral
display_name: Mistral Timestral 209M (symmetric averaging off)
source: {repo: https://github.com/mistralai/timestral, docs: https://docs.mistral.ai/models/timestral-1-0, hf_id: mistralai/Timestral-209M}
weights_date: "2026-10-15"
training_data_disclosed: false
revision: 9874d252ab0aae16f7f7c999f98f98d6ad7d8365
params: ~209M
color: "#6a51a3"
default_params: {batch_size: 16, symmetric_averaging: false}
ablation_of: timestral-209m
---

## What it is
Ablation of `timestral-209m`: the SAME checkpoint (same HF id and pinned revision) called with
`symmetric_averaging=False`. Everything else (adapter, covariates, context handling, quantiles) is identical.
It exists so the default stays untouched for every `timestral-*` id while the speed/accuracy effect of the mirrored
pass can be measured as a separate, explicitly named model.

## How the adapter calls it
`tsbench.models.timestral:TimestralAdapter`, `.forecast(..., symmetric_averaging=False, clip_nonnegative=True)`.
With averaging on (the library default), the library also forecasts `-context` with every covariate row negated and
averages the reflected quantiles; off, only the direct pass runs, so expect ~0.5x the CPU time of `timestral-209m`.
The setting is recorded in `run_meta.json` (`adapter_runtime.symmetric_averaging`) and is part of the run_id.

## Input contract, covariate handling, quirks
As `timestral-209m` (see that card, including the covariate-path profiling).

## Training data and cutoff
- Same checkpoint as `timestral-209m` (identical weights); everything below applies unchanged.
- **Stated training-data cutoff: none.** The tech report (arXiv:2610.37585, submitted 2026-10-27) gives no cutoff and
  no time range for its real data; neither do the model card, blog post, docs or GitHub README.
- **Weights date (hard upper bound):** `model.safetensors` last changed **2026-10-15**, HF commit `7394bc19`
  ("Super-squash branch 'main' using huggingface_hub"); all four sizes were uploaded the same day.
- **Training data (as disclosed):** the tech report (§2.1, §4.1) confirms the two-stage pipeline: pre-training
  "exclusively on synthetic data" (context 8,192), then post-training "on a mixture of real and synthetic data with
  extended context" (32,768). §4.1 describes only the synthetic corpus. The real post-training datasets are **still not
  named** and no domains are listed; energy, Belgian or Elia data are neither confirmed nor excluded. (Energy appears
  only in IND-Eval, an evaluation benchmark, Appendix A.2.) Blog post: "The majority of the training data was synthetic".
- **Benchmark overlap:** not addressed. The tech report has no decontamination or train/test-overlap statement, including
  for its own benchmarks (GIFT-Eval, TIME, fev-bench); the blog's "nothing we backtest was seen during pre-training"
  refers to customer fine-tuning in Forge, not to these weights.
- **Implication for this benchmark (2025 test year):** the weights postdate all of 2025 and the real post-training data is
  undisclosed (unchanged by the tech report). Elia load (ods001) and the Elia day-ahead forecasts are public, so **it cannot be ruled out that 2025 Belgian
  load was in the training data.** Any result where Timestral beats a model with an earlier weights date must carry this
  caveat.
- Sources: Timestral tech report, https://arxiv.org/abs/2610.37585 ; https://huggingface.co/mistralai/Timestral-209M ; https://mistral.ai/news/timestral/ ;
  https://docs.mistral.ai/models/timestral-1-0 ; https://github.com/mistralai/timestral ; weight commit date via
  `HfApi.get_paths_info(..., ["model.safetensors"], revision=<pinned>, expand=True)`. Checked 2026-10-20; tech report read 2026-11-13.

## Verification done
2026-10-20: `symmetric_averaging` semantics read from `timestral/forecaster.py` (v1.0.0, git a534bdb). Not run yet.

## Citation
Mistral AI, Timestral 1.0 (2026-10-15). https://github.com/mistralai/timestral
Petralia et al., "Timestral", arXiv:2610.37585 (2026-10-27). https://arxiv.org/abs/2610.37585
