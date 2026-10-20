# Adding a model

1. **Read the official docs/code** for the model's inference API. Do not guess it.
2. Write `src/tsbench/models/<name>.py` with an adapter subclassing `BaseAdapter`:
   `load(params, device)` and `predict(list[ForecastTask]) -> list[Forecast]`.
   - Truncate the context to `card.max_context` YOURSELF and set `Forecast.effective_context` to the number of
     steps you actually fed (the runner records it; never copy it from config).
   - Return quantiles keyed by level; include 0.5. Do not slice data: the window builder does.
3. Copy `model_cards/_TEMPLATE.md` to `model_cards/<id>.md` and fill it in. Pin the HF revision by commit hash.
   Set `adapter_version: 1` and bump it whenever adapter behaviour changes.
4. Add the id to the `tsfm-all` group in `configs/model_groups.yaml` (that is the only config line needed).
5. Run `bench list-models`, `pytest`, `bench leakage-test --model <id>`.

## Experiment naming and resume

Experiments are named by the question they answer, never by model names:
- `configs/experiments/leaderboard.yaml` is the main experiment (`tsfm-all` + `baselines`, covariates `[none, full]`,
  contexts `[15, 30, 45, 60, 75, 90]`).
- `configs/experiments/smoke_test.yaml` is the fast end-to-end check (`timestral-5m`, `none`, `[60]`).

Run folders (`outputs/runs/...`) are shared across experiments and keyed by cell. After adding a model to `tsfm-all`,
re-run `bench run --config configs/experiments/leaderboard.yaml`: cells whose `run_id` matches and whose window count is
complete are skipped, so only the new model's cells are computed. Then `bench summarize --config ...`.
