---
id: timestral-209m
family: timestral
display_name: Mistral Timestral 209M
source: {repo: https://github.com/mistralai/timestral, docs: https://docs.mistral.ai/models/timestral-1-0, hf_id: mistralai/Timestral-209M}
weights_date: "2026-10-15"
training_data_disclosed: false
revision: 9874d252ab0aae16f7f7c999f98f98d6ad7d8365
params: ~209M
color: "#08306b"
---

## What it is
Mistral AI zero-shot probabilistic time series foundation model, ~209M parameters. Same architecture
family and config as the other Timestral sizes (patch 32, block 32, context 32,768, 21 quantiles
0.01..0.99); only width/depth differ (verified from each `config.json`, 2026-10-20).

## How the adapter calls it
`tsbench.models.timestral:TimestralAdapter` -> `TimestralForecaster.from_pretrained("mistralai/Timestral-209M",
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

## Training data and cutoff
- **Stated training-data cutoff: none** (model card, blog post, docs and GitHub README give no date; the model card says
  "Tech report: coming soon!").
- **Weights date (hard upper bound):** `model.safetensors` last changed **2026-10-15**, HF commit `7394bc19`
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
- Sources: https://huggingface.co/mistralai/Timestral-209M ; https://mistral.ai/news/timestral/ ;
  https://docs.mistral.ai/models/timestral-1-0 ; https://github.com/mistralai/timestral ; weight commit date via
  `HfApi.get_paths_info(..., ["model.safetensors"], revision=<pinned>, expand=True)`. Checked 2026-10-20.

## Verification done
2026-10-20: read `timestral/forecaster.py` (v1.0.0, git a534bdb) and the repo README; HF config checked;
revision pinned to the commit above.

## Citation
Mistral AI, Timestral 1.0 (2026-10-15). https://github.com/mistralai/timestral
