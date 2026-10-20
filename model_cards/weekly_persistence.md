---
id: weekly_persistence
family: baseline
kind: baseline
display_name: Weekly persistence (t-7d)
publisher: this repo
license: n/a
source: {}
revision: n/a
params: "0"
max_context: null
covariates: {past: false, future: false, categorical: false}
outputs: {type: point, levels: [0.5]}
device: [cpu]
adapter: tsbench.models.baselines:WeeklyPersistenceAdapter
adapter_version: 1
color: "#bdbdbd"
ignores_context_and_covariates: true
needs_reference: false
default_params: {}
---

## Definition
P50 at target quarter-hour t = observed value at the same local wall-clock time 7 days earlier (always available,
since t-7d < origin for every horizon step). Parameters: none. Context: 8 days. Point forecast only.

No day-type matching and no holiday calendar: this baseline does not select source days by type.

All lookups use LOCAL (Europe/Brussels) wall-clock quarter-hours and only the context handed over by the window builder, so a source value at or after 18:00 on the origin day is unavailable by construction. Gaps (the skipped spring-DST hour on a source day) are forward-filled along the horizon; the repeated autumn hour is averaged.

These definitions are this repo's own. They are NOT claimed to reproduce the internship values (the internship definitions and data file differ).
