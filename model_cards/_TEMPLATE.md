---
id: my-model                # must equal the file name
family: my-family           # optional shared facts in model_cards/_families/<family>.yaml
display_name: My Model
publisher: Someone
release_date: "YYYY-MM-DD"
license: ...
source: {repo: ..., docs: ..., hf_id: org/name}
revision: <40-char HF commit hash>      # PIN IT
params: 10M
max_context: 2048           # steps; null = unbounded
max_horizon_per_pass: 256
covariates: {past: false, future: true, categorical: false}
nan_input: true
outputs: {type: quantiles, levels: [0.1, 0.5, 0.9]}   # must include 0.5
device: [cpu, cuda]
adapter: tsbench.models.my_model:MyAdapter
adapter_version: 1          # bump whenever adapter behaviour changes -> resume re-runs stale cells
color: "#123456"            # fixed colour in all figures
default_params: {}
---

## What it is
## How the adapter calls it
## Input contract (NaN, scaling, frequency)
## Covariate handling
## Known quirks
## Verification done (date + what)
## Citation
