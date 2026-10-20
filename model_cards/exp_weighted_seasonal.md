---
id: exp_weighted_seasonal
family: baseline
kind: baseline
display_name: Exp.-weighted seasonal (day-type matched)
publisher: this repo
license: n/a
source: {}
revision: n/a
params: "0"
max_context: null
covariates: {past: false, future: false, categorical: false}
outputs: {type: point, levels: [0.5]}
device: [cpu]
adapter: tsbench.models.baselines:ExpWeightedSeasonalAdapter
adapter_version: 2
color: "#238b45"
ignores_context_and_covariates: true
needs_reference: false
default_params: {n_days: 28, alpha: 0.3, day_matching: day_type}
---

## Definition
Same source selection as `seasonal_moving_average` (the `n_days` most recent matching days, same quarter-hour,
observed at or before 17:45 on the origin day), but weighted: weight_j = alpha * (1 - alpha)^j, j = 0 for the most
recent matching day, normalised to sum to 1.
- `n_days` = 28, `alpha` = 0.3 (defaults; with alpha 0.3 the weight beyond ~10 matches is negligible).
- `day_matching` = `day_type` (workday / saturday / sunday_holiday; see seasonal_moving_average).
- Context: 7*n_days + 2 = 198 days (reaches mid-2024 for January 2025 origins; never before 2024-01-01).
  Point forecast only.

adapter_version 2 (2026-10-20): default changed from any-day matching (v1, weekend 9.50% vs weekday 5.11% on 2025)
to day-type matching.

## Provenance of the defaults
- Day-type matching was **adopted after seeing initial 2025 results**: the v1 (any-day) version showed weekend WAPE
  ~1.9x weekday WAPE (9.50% vs 5.11%), which motivated matching. It is therefore not a blind, pre-registered choice.
- **n_days and alpha unchanged (not tuned)**: 28 and 0.3, the values used before the change. Other values were
  evaluated only to check the effect and were not adopted (n_days 8, alpha 0.5 scores 3.65% vs 3.84% on 2025).
- **Holiday calendar**: Belgian public holidays (`holidays.Belgium`, local date) enter only through the
  sunday_holiday day-type group: a holiday is matched with Sundays and other holidays, never with workdays.

All lookups use LOCAL (Europe/Brussels) wall-clock quarter-hours and only the context handed over by the window builder, so a source value at or after 18:00 on the origin day is unavailable by construction. Gaps (the skipped spring-DST hour on a source day) are forward-filled along the horizon; the repeated autumn hour is averaged.

These definitions are this repo's own. They are NOT claimed to reproduce the internship values (the internship definitions and data file differ).
