---
id: chronos-2
family: chronos
display_name: Amazon Chronos-2
publisher: Amazon
release_date: "2025-10"
license: Apache 2.0
source: {repo: https://github.com/amazon-science/chronos-forecasting, docs: https://github.com/amazon-science/chronos-forecasting, hf_id: amazon/chronos-2}
weights_date: "2025-10-30"
training_data_disclosed: true
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

## Training data and cutoff
- **Stated training-data cutoff: none** (model card and technical report give no date).
- **Weights date (hard upper bound on any training data):** `model.safetensors` last changed **2025-10-30**, HF commit
  `95a9710e` ("Add model weights and config"). The pinned revision `29ec3766` (2026-06-05) only changed documentation.
- **Training data (disclosed):** model card: "Subset of Chronos Datasets", "Subset of GIFT-Eval Pretrain", "Synthetic
  univariate and multivariate data". Technical report (Sec. 4.1, Table 6) lists the real datasets; energy-domain ones are
  Electricity, Solar, Wind Farms, London Smart Meters and Buildings 900K. No Belgian or Elia dataset is listed; EPF-BE (a
  Belgian electricity-price task) appears only in the evaluation set (Table 10).
- **Benchmark overlap (Sec. 5.1):** test portions of GIFT-Eval excluded ("we carefully ensured that it did not overlap with
  the test portions of any GIFT-Eval task"), with "partial overlap with the training portions of some GIFT-Eval datasets";
  fev-bench and Chronos Benchmark II not seen in training.
- **Implication for this benchmark (2025 test year):** training on 2025 Elia load is unlikely (no Belgian load dataset is
  listed and the listed real datasets are older public sets) but cannot be formally excluded, since no cutoff is published.
  Nov-Dec 2025 postdate the weights.
- Sources: https://huggingface.co/amazon/chronos-2 ; https://arxiv.org/abs/2510.15821 (Sec. 4.1, Table 6, Sec. 5.1, Table 10);
  weight commit date via `HfApi.get_paths_info("amazon/chronos-2", ["model.safetensors"], revision=<pinned>, expand=True)`.
  Checked 2026-10-20.

## Verification done
2026-10-20: read `chronos/chronos2/pipeline.py` (`predict`, long-horizon logic), HF `config.json`
(context_length 8192, 21 quantiles, output patch 16); revision pinned.

## Citation
Ansari et al., Chronos-2 (Amazon, 2025). https://github.com/amazon-science/chronos-forecasting
