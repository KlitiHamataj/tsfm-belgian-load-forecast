---
id: last_value
family: baseline
kind: baseline
display_name: Last value (17:45 held flat)
publisher: this repo
license: n/a
source: {}
revision: n/a
params: "0"
max_context: null
covariates: {past: false, future: false, categorical: false}
outputs: {type: point, levels: [0.5]}
device: [cpu]
adapter: tsbench.models.baselines:LastValueAdapter
adapter_version: 1
color: "#969696"
ignores_context_and_covariates: true
needs_reference: false
default_params: {}
---

## Definition
P50 for every horizon step (bridge + scored day) = the last observed value at 17:45 local on the origin day.
Parameters: none. Context: 1 day. Point forecast only.

Renamed from `persistence` (2026-10-20); same behaviour.

No day-type matching and no holiday calendar: this baseline does not select source days by type.

## Training data and cutoff
- **No training and no learned parameters.** Each forecast is computed only from the context the window builder hands over
  (nothing at or after 18:00 on the origin day), so there is no training-data cutoff and no contamination path.
- Caveat on design, not data: the seasonal baselines' day-type matching was adopted after seeing 2025 results (see
  "Provenance of the defaults" where present). Checked 2026-10-20.

All lookups use LOCAL (Europe/Brussels) wall-clock quarter-hours and only the context handed over by the window builder, so a source value at or after 18:00 on the origin day is unavailable by construction. Gaps (the skipped spring-DST hour on a source day) are forward-filled along the horizon; the repeated autumn hour is averaged.

These definitions are this repo's own. They are NOT claimed to reproduce the internship values (the internship definitions and data file differ).
