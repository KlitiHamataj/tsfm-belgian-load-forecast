---
id: timestral-123m-nosym
family: timestral
display_name: Mistral Timestral 123M (symmetric averaging off)
source: {repo: https://github.com/mistralai/timestral, docs: https://docs.mistral.ai/models/timestral-1-0, hf_id: mistralai/Timestral-123M}
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

## Verification done
2026-10-20: `symmetric_averaging` semantics read from `timestral/forecaster.py` (v1.0.0, git a534bdb). Created 2026-10-20 after the 209M ablation showed nosym better and faster, to test whether the effect holds at 123M.

## Citation
Mistral AI, Timestral 1.0 (2026-10-15). https://github.com/mistralai/timestral
