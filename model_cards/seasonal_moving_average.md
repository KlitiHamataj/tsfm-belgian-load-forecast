---
id: seasonal_moving_average
family: baseline
kind: baseline
display_name: Seasonal moving average (day-type matched)
publisher: this repo
license: n/a
source: {}
revision: n/a
params: "0"
max_context: null
covariates: {past: false, future: false, categorical: false}
outputs: {type: point, levels: [0.5]}
device: [cpu]
adapter: tsbench.models.baselines:SeasonalMovingAverageAdapter
adapter_version: 2
color: "#74c476"
ignores_context_and_covariates: true
needs_reference: false
default_params: {n_days: 7, day_matching: day_type}
---

## Definition
P50 at target quarter-hour t on target day d = unweighted mean of the values at the same local quarter-hour on the
`n_days` most recent MATCHING days d-k (k = 1, 2, ...) on which that quarter-hour was observed at or before 17:45
on the origin day.
- `n_days` = 7 (default).
- `day_matching` = `day_type` (default): a source day matches when it has the same day type as d, with types
  workday (Mon-Fri, not a Belgian public holiday), saturday, and sunday_holiday (Sunday or a Belgian public holiday,
  `holidays.Belgium`). Alternatives: `all` (any day) and `same_weekday`.
- Lags searched: k = 1 .. 7*n_days + 2 (all: n_days + 2). If fewer than n_days matches exist, the mean uses those found.
- Context: 7*n_days + 2 = 51 days. Point forecast only.

adapter_version 2 (2026-10-20): default changed from any-day matching (v1) to day-type matching, because v1's
weekend WAPE was ~1.7x its weekday WAPE (8.49% vs 4.91% on 2025).

## Provenance of the defaults
- Day-type matching was **adopted after seeing initial 2025 results**: the v1 (any-day) version showed weekend WAPE
  ~1.7x weekday WAPE (8.49% vs 4.91%), which motivated matching. It is therefore not a blind, pre-registered choice.
- **n_days unchanged (not tuned)**: 7, the value used before the change. Other values were evaluated only to check
  the effect and were not adopted (n_days = 3 scores 3.86% vs 4.24% on 2025).
- **Holiday calendar**: Belgian public holidays (`holidays.Belgium`, local date) enter only through the
  sunday_holiday day-type group: a holiday is matched with Sundays and other holidays, never with workdays.

All lookups use LOCAL (Europe/Brussels) wall-clock quarter-hours and only the context handed over by the window builder, so a source value at or after 18:00 on the origin day is unavailable by construction. Gaps (the skipped spring-DST hour on a source day) are forward-filled along the horizon; the repeated autumn hour is averaged.

These definitions are this repo's own. They are NOT claimed to reproduce the internship values (the internship definitions and data file differ).
