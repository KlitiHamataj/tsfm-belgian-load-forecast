---
id: t0-gamma
family: t0
display_name: The Forecasting Company t0-gamma
publisher: The Forecasting Company
release_date: "2026-10-12"
license: Apache 2.0
source: {repo: https://github.com/theforecastingcompany/tfc-t0, docs: https://huggingface.co/theforecastingcompany/t0-gamma, hf_id: theforecastingcompany/t0-gamma, paper: https://arxiv.org/abs/2609.24559}
revision: e08aee083778f1f32da4974c0a34276776a35c52
params: 256M
max_context: null
max_horizon_per_pass: 1024
covariates: {past: true, future: true, categorical: false}
nan_input: true
outputs:
  type: quantiles
  levels: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
device: [cpu, cuda]
adapter: tsbench.models.t0:T0Adapter
adapter_version: 1
color: "#7b3294"
default_params: {batch_size: 16, trim_leading_missing: true, symmetric_averaging: true, clip_nonnegative: true, snap_integers: true}
---

## What it is
Open-weights probabilistic time series foundation model from The Forecasting Company, third public iteration of the
t0 family (after t0-alpha 102M and t0-beta 256M; the paper, arXiv:2609.24559 v1, 21 Sep 2026, covers alpha and beta).
Decoder-style patch transformer: 24 layers (16 time-attention, 8 group-attention; `group_every_n` 3), embed 1024,
feedforward 2048, 8 heads, patch 32, causal mean/std scaler with arcsinh. 255,614,272 parameters (counted after loading
the pinned revision). HF repo created 2026-10-12 (used as release date; the card gives none).

## How the adapter calls it
`tsbench.models.t0:T0Adapter` -> `T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma", revision=<pinned>)`
(`PyTorchModelHubMixin`), `.to(device).eval()`, then per batch of windows with equal (context, horizon):
`predict(context (B, T), horizon=H, quantile_levels=<the 21 trained levels>, future_covariates=(B, F, T+H), **adjustments)`
-> `Forecast.quantiles` (B, H, 21). Asking for exactly the trained levels means nothing is interpolated or extrapolated.

**Inference settings (a choice, read this).** `predict` has four adjustments that the library defaults to OFF:
`trim_leading_missing`, `symmetric_averaging`, `clip_nonnegative`, `snap_integers`. The tfc-t0 README states that all four
on "produce `t0-gamma`'s published scores", and the docstring calls that "the recommended setting". This card turns
all four ON, so t0-gamma runs as its publisher evaluates it (same principle as Timestral, whose library default is
already on). On our data only `symmetric_averaging` matters: there are no NaNs (trim is a no-op), load is not
integer-valued (snap is a no-op) and forecasts are positive (clip is a no-op). `symmetric_averaging` runs the model
twice (on the series and on its negation) and roughly doubles CPU cost. The settings are recorded in `run_meta.json`
(`adapter_runtime`) and are part of the run_id.

## Input contract
- NaN marks a missing observation (none occur in our windows). Scaling is internal. No frequency argument.
- **No documented maximum context.** Neither the model card, the GitHub README, the paper abstract nor `config.json`
  state a trained or maximum context length (examples use 512-1024 steps; the paper evaluates t0-beta with "a context
  of nearly a year" on Victoria electricity). `max_context: null`, so the adapter feeds the configured context in full
  (up to 90 days = 8640 steps = 270 patches) and reports that as `effective_context`. Accuracy at long contexts is
  therefore an empirical question here, not a documented capability.
- Horizon <= 1024 steps per forward pass; our 116-124 steps are a single pass.

## Covariate handling
All four covariates (`day_of_week`, `hour_of_day`, `be_holidays`, `be_temperature`) are passed as NUMERIC
`future_covariates` over context + horizon (known-future: conditioned on, not forecast), same as for Chronos-2 and
Timestral. Past-only covariates would go in as extra context variates (not used here). No categorical path is documented.
`group_ids` cannot be combined with `future_covariates` (library constraint); we do not need it (one target per window).

## Known quirks
- Requires `tfc-t0>=0.6.0`: older versions ignore the `xpos_scale_layout` / scaler fields of this checkpoint's
  `config.json` and "silently degrade the forecast" (tfc-t0 README). Pinned in `pyproject.toml`.
- Shares the project environment (torch 2.14.1+cpu) with Chronos-2 and Timestral; no separate env needed.
- Quantile output is float32 on the model's device.

## Verification done
2026-10-20: read the HF model card, the tfc-t0 GitHub README and `t0/forecaster.py` (`predict`, `_series_from_array`,
adjustment helpers), `t0/rollout.py` (single pass up to `DEFAULT_MAX_HORIZON` = 1024), `t0/torch/model/model.py`
(`PyTorchModelHubMixin`, so `revision=` is honoured), and `config.json` at the pinned revision (21 levels 0.01..0.99,
patch 32, 24 layers). Loaded the pinned revision on CPU, counted parameters, and checked output shapes on a normal day
(H = 120) and the autumn DST day (H = 124) with `full` covariates.

## Citation
Meyer, L., Sole, C., Xiang, H., Li, N., Franceschino, L., Quera-Bofarull, A., Scholl, M. P., Fainberg, J., & Négiar, G.
(2026). t0: A Time-Series Foundation Model for Forecasting with Context. arXiv:2609.24559.
