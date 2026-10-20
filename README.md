# elia-load-tsfm-benchmark

Zero-shot time series foundation models (Chronos-2, Mistral Timestral 5M/28M/123M/209M) versus baselines on
**Belgian day-ahead electricity load** (Elia open data, 15-min). CPU only, `uv`, Python 3.12.
Adding a model = one adapter file + one model card + its id in the `tsfm-all` group.

## Results
_Auto-filled from `outputs/experiments/leaderboard/summary.md` after `bench summarize` (not run yet)._

## Quickstart
```bash
uv sync                                   # CPU torch from the PyTorch CPU index; no CUDA wheels
uv run bench prepare                      # raw CSVs -> data/processed/*.parquet + manifest.json
uv run bench list-models
uv run pytest
uv run bench run --config configs/experiments/smoke_test.yaml
uv run bench run --config configs/experiments/leaderboard.yaml
uv run bench summarize --config configs/experiments/leaderboard.yaml
uv run bench gates --config configs/experiments/leaderboard.yaml   # leakage + determinism per model
```
The repo must live on the WSL filesystem (not `/mnt/c`): file I/O there is much slower.

## Data provenance
Raw files are read-only, in `data/` and **not committed** (the SHA-256 hashes are in `configs/protocol.yaml`;
`bench prepare` refuses a mismatch).
- `data/load.csv`: Elia Open Data ods001 (downloaded 2026-10-19), 15-min, 2024-01-01 to 2025-12-31 Brussels, 70,176 rows.
  Target = `Total Load` (MW); reference forecaster = `Day-ahead 6PM forecast` (+P10/P90).
- `data/weather_cov.csv`: Open-Meteo Previous Runs API, Charleroi, `temperature_2m_previous_day1`, hourly UTC,
  linearly interpolated to 15 min; the leading NaN block (before 2024-01-19 12:00 UTC) is left NaN and asserted
  absent from every context/horizon used.
- **Caveat:** Open-Meteo has no issue timestamps, so availability of the temperature at 18:00 the day before cannot be proven.
- Elia open data is reusable with attribution; cite Elia when publishing results.

## Protocol
- One origin per day at 18:00 Europe/Brussels; the last observed value is 17:45. Nothing at/after 18:00 enters the context.
- Horizon = 24 bridge steps (18:00-23:45, predicted, **not scored**) + the next local calendar day.
- **DST:** the scored day has 92/96/100 steps, so horizons are 116/120/124 and the full local day is scored.
  The internship pipeline scored a fixed 96 steps, so DST days differ from the old numbers.
- Scored days: every day of 2025 (365 windows). Context may reach into 2024; nothing before 2024-01-01 is ever read.
- Metrics: pooled WAPE (primary), MAE, RMSE, sMAPE, median-window WAPE (labelled), bootstrap 95% CI over days,
  paired deltas + sign test, slices, probabilistic metrics on the common quantile set 0.1..0.9, latency.
- Covariate sets: `none`, `calendar` (day_of_week, hour_of_day, be_holidays), `full` (+ be_temperature). All numeric.

## Adding a model
See [docs/adding_a_model.md](docs/adding_a_model.md).

## Limitations
- Open-Meteo availability timing (see above); Elia may have revised historical data since the internship files.
- Baselines (`last_value`, `daily_persistence`, `weekly_persistence`, `seasonal_moving_average`, `exp_weighted_seasonal`)
  use only quarter-hours observed before 18:00; exact definitions are in their cards. They are not claimed to reproduce
  the internship values.
- Timestral with covariates is ~10x slower than univariate on CPU; this is the library's covariate path (see the cards).
- CPU-only latency depends on thread count (recorded in `run_meta.json`).
