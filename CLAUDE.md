# CLAUDE.md

Zero-shot TSFM benchmark on Belgian day-ahead load (Elia ods001). CPU only, `uv`, Python 3.12.

## Architecture
- `configs/protocol.yaml` (fixed protocol, covariate sets, eval quantiles), `configs/experiments/*.yaml` (run matrix),
  `configs/model_groups.yaml` (groups).
- `model_cards/<id>.md`: YAML front matter is READ by code (capabilities, pinned HF revision, `adapter`,
  `adapter_version`, colour). `_families/*.yaml` is merged under each card.
- `src/tsbench/`: `data/prepare.py` (raw -> parquet + manifest), `windows.py` (the ONLY place that slices data),
  `models/{base,chronos2,timestral,baselines}.py` (adapters), `runner.py` (matrix, run_id, resume, cell execution),
  `cli.py` (`bench ...`; each model runs in its own subprocess via the hidden `worker` command), `metrics.py`,
  `summarize.py`, `leakage.py` (gates), `analyses.py` (`bench mirrored-pass-test`, `bench retime`; results land in
  `outputs/experiments/<name>/analyses/` and are appended to that summary.md).
- Outputs: `outputs/runs/<model>/<cov>/ctxNNd/` (shared across experiments), `outputs/experiments/<name>/` (aggregates only).

## Invariants (do not break)
1. Never read data before 2024-01-01 Brussels (asserted in `load_processed` and `build_task`).
2. Origin = 18:00 Europe/Brussels; the last context value is 17:45; nothing at/after 18:00 enters the context.
3. The 24 bridge steps (18:00-23:45) are predicted but NEVER scored. Scored = next local calendar day
   (92/96/100 steps; horizon 116/120/124). The internship pipeline scored a fixed 96 steps.
4. Raw data in `data/*.csv` is read-only and not committed. Hashes live in `protocol.yaml`; `bench prepare` refuses a mismatch.
5. No NaN fill of the leading temperature block; `build_task` asserts no NaN inside any used context/horizon.
6. Adapters truncate the context themselves and report `effective_context` as what they actually fed.
   Never copy it from config.
7. Bump `adapter_version` in the card whenever adapter behaviour changes (it is part of run_id, so resume re-runs stale cells).
8. Timestral covariates are NUMERIC (same as Chronos-2). Categorical handling only via a separately named covariate set.
9. Probabilistic metrics use only `eval_quantiles` (0.1..0.9); all native levels are still stored in points.parquet.
10. HF revisions are pinned by commit hash in the cards.

## Checks
```
uv run pytest                                    # unit tests (fast)
uv run bench list-models
uv run bench prepare
uv run bench run --config configs/experiments/baselines_only.yaml   # elia_reference must give 3.036% WAPE
uv run bench leakage-test --model chronos-2      # one model; writes outputs/gates/<model>.json
uv run bench gates --config configs/experiments/leaderboard.yaml   # leakage + determinism for every model in the config
uv run bench summarize --config configs/experiments/<x>.yaml
uv run ruff check src tests && uv run mypy src
```
Gates (reported in summary.md): `elia_reference` pooled WAPE on 2025 = 3.036 ± 0.001; leakage and determinism per model
(stale when the card's revision or adapter_version changed since recording); 365/365 windows per run.

Baselines: `last_value` (17:45 held flat), `daily_persistence`, `weekly_persistence`, `seasonal_moving_average` and
`exp_weighted_seasonal` (day-type matched since adapter_version 2). Exact definitions live in their cards; they are
not claimed to reproduce the internship values.

## Experiment naming convention
Name experiments by the question they answer, never by model names: `leaderboard.yaml` (group `tsfm-all` + `baselines`,
`[none, full]`, `[15, 30, 45, 60, 75, 90]`; the main experiment) and `smoke_test.yaml` (timestral-5m, none, [60]).

## Adding a model
See `docs/adding_a_model.md`: one adapter file + one card + add the id to `tsfm-all` in `configs/model_groups.yaml`.
Re-running `leaderboard.yaml` then computes only the new cells (resume by `run_id` + complete window count).
