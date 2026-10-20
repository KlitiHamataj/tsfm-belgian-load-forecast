---
id: timestral-5m
family: timestral
display_name: Mistral Timestral 5M
source: {repo: https://github.com/mistralai/timestral, docs: https://docs.mistral.ai/models/timestral-1-0, hf_id: mistralai/Timestral-5M}
revision: bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f
params: 5.21M
color: "#9ecae1"
---

## What it is
Mistral AI zero-shot probabilistic time series foundation model, 5.21M parameters. Same architecture
family and config as the other Timestral sizes (patch 32, block 32, context 32,768, 21 quantiles
0.01..0.99); only width/depth differ (verified from each `config.json`, 2026-10-20).

## How the adapter calls it
`tsbench.models.timestral:TimestralAdapter` -> `TimestralForecaster.from_pretrained("mistralai/Timestral-5M",
revision=<pinned>)`, then `.forecast(context (B,T), prediction_length=H, known_covariates=(B,C,T+H))`.
Library defaults kept: `symmetric_averaging=True` (a second, mirrored pass: ~2x CPU cost),
`clip_nonnegative=True`. Windows with equal (context, horizon) are batched.

## Input contract
- NaN marks missing values (none occur in our windows). Scaling is internal (causal patch loc/scale).
- Frequency is not an argument; the model sees raw steps (15-min here).
- Context above `max_context` is cut to the most recent steps; the adapter does this itself and
  reports the length it fed as `effective_context`.

## Covariate handling
All covariates (`day_of_week`, `hour_of_day`, `be_holidays`, `be_temperature`) are passed as NUMERIC
`known_covariates` over context+horizon, same as for Chronos-2. The categorical path
(`categorical_known_covariates`, target-encoded) is NOT used; it is reserved for a separately named
covariate set.

## Known quirks
- Horizons > 1024 are decoded block-by-block with median feedback; we use H = 116..124 (single pass).
- Quantile outputs are clipped at 0 when the observed context is non-negative.
- Output is sorted along the quantile axis (no crossing).
- **Covariates are ~10x slower than univariate on CPU; it is model compute, not the adapter.**
  Profiled 2026-10-20 (8 threads, 60-day context, 8 windows of 2025, `symmetric_averaging` default = on), timing the
  library call `TimestralForecaster.forecast` separately from the rest of `TimestralAdapter.predict`:

  | model | covariates | model forward (s/window) | adapter own work (ms/window) | adapter share |
  |---|---|---|---|---|
  | timestral-28m | none | 0.133 | 0.1 | 0.05% |
  | timestral-28m | full (4) | 1.278 | 0.1 | 0.01% |
  | timestral-209m | none | 0.776 | 0.1 | 0.01% |
  | timestral-209m | full (4) | 9.207 | 0.3 | <0.01% |

  The window builder (data slicing, outside the adapter) adds 24-34 ms/window. Full leaderboard cells agree
  (timestral-209m, 60d: 2967 s with `full` vs 287 s with `none`).
  Cause: any covariate switches the library to its two-stream covariate path (`TimestralModel._predict_window`):
  each covariate is an extra row through the transformer, the decoder stream spans the whole window (2N tokens
  instead of the horizon), and explicit (R, 1, 2N, 2N) attention masks are built. Synthetic scan (28m, B=8): 0.13
  s/window with 0, 0.66 with 1, 0.75 with 2, 1.26 with 4 covariates; batch size has no effect (1.3-1.6 s/window
  for B = 1..16). `symmetric_averaging` stays at the library default (on) for every `timestral-*` id; turning it off
  roughly halves both paths (see `timestral-209m-nosym`).

## Verification done
2026-10-20: read `timestral/forecaster.py` (v1.0.0, git a534bdb) and the repo README; HF config checked;
revision pinned to the commit above.

## Citation
Mistral AI, Timestral 1.0 (2026-10-15). https://github.com/mistralai/timestral
