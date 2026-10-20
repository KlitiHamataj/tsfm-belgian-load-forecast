---
id: timestral-209m-nosym
family: timestral
display_name: Mistral Timestral 209M (symmetric averaging off)
source: {repo: https://github.com/mistralai/timestral, docs: https://docs.mistral.ai/models/timestral-1-0, hf_id: mistralai/Timestral-209M}
revision: 9874d252ab0aae16f7f7c999f98f98d6ad7d8365
params: ~209M
color: "#6a51a3"
default_params: {batch_size: 16, symmetric_averaging: false}
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

## Verification done
2026-10-20: `symmetric_averaging` semantics read from `timestral/forecaster.py` (v1.0.0, git a534bdb). Not run yet.

## Citation
Mistral AI, Timestral 1.0 (2026-10-15). https://github.com/mistralai/timestral
