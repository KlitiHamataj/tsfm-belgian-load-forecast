---
id: chronos-2
family: chronos
display_name: Amazon Chronos-2
publisher: Amazon
release_date: "2025-10"
license: Apache 2.0
source: {repo: https://github.com/amazon-science/chronos-forecasting, docs: https://github.com/amazon-science/chronos-forecasting, hf_id: amazon/chronos-2}
revision: 29ec3766d36d6f73f0696f85560a422f50e8498c
params: ~120M
max_context: 8192
max_horizon_per_pass: 1024
covariates: {past: true, future: true, categorical: true}
nan_input: true
outputs:
  type: quantiles
  levels: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
device: [cpu, cuda]
adapter: tsbench.models.chronos2:Chronos2Adapter
adapter_version: 1
color: "#e6550d"
default_params: {batch_size: 64}
---

## What it is
Amazon's Chronos-2 encoder-only time series foundation model with native past + known-future covariates
via group attention. Checkpoint `amazon/chronos-2`, `chronos-forecasting` 2.3.2.

## How the adapter calls it
`Chronos2Pipeline.from_pretrained(hf_id, device_map, revision=<pinned>)`, then
`pipe.predict([{target, past_covariates:{name: arr(T)}, future_covariates:{name: arr(H)}}], prediction_length=H,
batch_size, context_length=8192)`. Each window is its own group (no cross-learning). Output per window:
`(1, 21, H)`; levels from `pipe.quantiles`.

## Input contract
NaN allowed (none in our windows). Internal arcsinh scaling. Frequency not an argument. The adapter cuts the
context to `max_context` (8192 steps = 85.3 days) itself and reports the fed length as `effective_context`.

## Covariate handling
All four covariates are numeric; each is given for the context (past_covariates) and the horizon
(future_covariates), i.e. treated as known-future.

## Known quirks
- 90-day context (8640 steps) exceeds the model maximum: truncated to 8192 under `context_policy: truncate`.
- Horizon limit 16 patches x 64 = 1024 steps per pass; we use 116..124.

## Verification done
2026-10-20: read `chronos/chronos2/pipeline.py` (`predict`, long-horizon logic), HF `config.json`
(context_length 8192, 21 quantiles, output patch 16); revision pinned.

## Citation
Ansari et al., Chronos-2 (Amazon, 2025). https://github.com/amazon-science/chronos-forecasting
