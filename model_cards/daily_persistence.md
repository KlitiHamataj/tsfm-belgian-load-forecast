---
id: daily_persistence
family: baseline
kind: baseline
display_name: Daily persistence (latest same quarter-hour)
publisher: this repo
license: n/a
source: {}
revision: n/a
params: "0"
max_context: null
covariates: {past: false, future: false, categorical: false}
outputs: {type: point, levels: [0.5]}
device: [cpu]
adapter: tsbench.models.baselines:DailyPersistenceAdapter
adapter_version: 1
color: "#737373"
ignores_context_and_covariates: true
needs_reference: false
default_params: {}
---

## Definition
For each target quarter-hour, the most recent observed value at the same local quarter-hour, observed at or before
17:45 on the origin day. For a target on scored day D:
- 00:00-17:45 on D -> value on D-1 (the origin day) at that quarter-hour;
- 18:00-23:45 on D -> value on D-2 (D-1 after 17:45 is not yet observed);
- bridge steps (18:00-23:45 on D-1) -> value on D-2.
Any source day qualifies. Parameters: none. Context: 3 days. Point forecast only.
Equivalent to `seasonal_moving_average` with n_days = 1 and day_matching = all.

No day-type matching and no holiday calendar: this baseline does not select source days by type.

## Training data and cutoff
- **No training and no learned parameters.** Each forecast is computed only from the context the window builder hands over
  (nothing at or after 18:00 on the origin day), so there is no training-data cutoff and no contamination path.
- Caveat on design, not data: the seasonal baselines' day-type matching was adopted after seeing 2025 results (see
  "Provenance of the defaults" where present). Checked 2026-10-20.

All lookups use LOCAL (Europe/Brussels) wall-clock quarter-hours and only the context handed over by the window builder, so a source value at or after 18:00 on the origin day is unavailable by construction. Gaps (the skipped spring-DST hour on a source day) are forward-filled along the horizon; the repeated autumn hour is averaged.

These definitions are this repo's own. They are NOT claimed to reproduce the internship values (the internship definitions and data file differ).
