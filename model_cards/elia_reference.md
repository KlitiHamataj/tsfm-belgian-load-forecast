---
id: elia_reference
family: baseline
kind: baseline
display_name: Elia day-ahead 6PM forecast
publisher: this repo
license: n/a
source: {}
revision: n/a
params: "0"
max_context: null
covariates: {past: false, future: false, categorical: false}
outputs: {type: quantiles, levels: [0.1, 0.5, 0.9]}
device: [cpu]
adapter: tsbench.models.baselines:EliaReferenceAdapter
adapter_version: 1
color: "#000000"
ignores_context_and_covariates: true
needs_reference: true
default_params: {}
---

## What it is
Elia's own 'Day-ahead 6PM forecast' (P50) with its P10/P90, taken from the dataset for the horizon timestamps.
Acceptance gate: pooled WAPE on 2025 = 3.036% (+/-0.001). No context, no covariates; the window builder hands the
adapter the reference columns because the card says `needs_reference`.
