---
id: timestral-123m-nosym
family: timestral
display_name: Mistral Timestral 123M (symmetric averaging off)
source: {repo: https://github.com/mistralai/timestral, docs: https://docs.mistral.ai/models/timestral-1-0, hf_id: mistralai/Timestral-123M}
weights_date: "2026-10-15"
training_data_disclosed: false
revision: 5a147287addda85f6acd8da3fb8858fe9af7e8f7
params: ~123M
color: "#9e9ac8"
default_params: {batch_size: 16, symmetric_averaging: false}
ablation_of: timestral-123m
---

## What it is
Ablation of `timestral-123m`: the SAME checkpoint (same HF id and pinned revision) called with
`symmetric_averaging=False`. Everything else (adapter, covariates, context handling, quantiles) is identical.
It exists so the default stays untouched for every `timestral-*` id while the speed/accuracy effect of the mirrored
pass can be measured as a separate, explicitly named model.

## How the adapter calls it
`tsbench.models.timestral:TimestralAdapter`, `.forecast(..., symmetric_averaging=False, clip_nonnegative=True)`.
With averaging on (the library default), the library also forecasts `-context` with every covariate row negated and
averages the reflected quantiles; off, only the direct pass runs, so expect ~0.5x the CPU time of `timestral-123m`.
The setting is recorded in `run_meta.json` (`adapter_runtime.symmetric_averaging`) and is part of the run_id.

## Input contract, covariate handling, quirks
As `timestral-123m` (see that card, including the covariate-path profiling).

## Training data and cutoff
- Same checkpoint as `timestral-123m` (identical weights); everything below applies unchanged.
- **Stated training-data cutoff: none** (model card, blog post, docs and GitHub README give no date; the model card says
  "Tech report: coming soon!").
- **Weights date (hard upper bound):** `model.safetensors` last changed **2026-10-15**, HF commit `154fec14`
  ("Super-squash branch 'main' using huggingface_hub"); all four sizes were uploaded the same day.
- **Training data (as disclosed):** blog post: "Trained on a large and diverse corpus"; "The majority of the training data
  was synthetic"; pre-training on synthetic data only, then "post-trained on a mix of real and synthetic data with a 4x
  longer context". The real datasets are **not named**; energy, Belgian or Elia data are neither confirmed nor excluded.
- **Benchmark overlap:** not addressed for the released model (the blog's "nothing we backtest was seen during
  pre-training" refers to customer fine-tuning in Forge, not to these weights).
- **Implication for this benchmark (2025 test year):** the weights postdate all of 2025 and the real post-training data is
  undisclosed. Elia load (ods001) and the Elia day-ahead forecasts are public, so **it cannot be ruled out that 2025 Belgian
  load was in the training data.** Any result where Timestral beats a model with an earlier weights date must carry this
  caveat.
- Sources: https://huggingface.co/mistralai/Timestral-123M ; https://mistral.ai/news/timestral/ ;
  https://docs.mistral.ai/models/timestral-1-0 ; https://github.com/mistralai/timestral ; weight commit date via
  `HfApi.get_paths_info(..., ["model.safetensors"], revision=<pinned>, expand=True)`. Checked 2026-10-20.

## Verification done
2026-10-20: `symmetric_averaging` semantics read from `timestral/forecaster.py` (v1.0.0, git a534bdb). Created 2026-10-20 after the 209M ablation showed nosym better and faster, to test whether the effect holds at 123M.

## Citation
Mistral AI, Timestral 1.0 (2026-10-15). https://github.com/mistralai/timestral
