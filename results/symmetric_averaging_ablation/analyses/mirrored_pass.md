## Mirrored-pass test (timestral-209m, none, 60d, 365 windows)

direct = the pass `symmetric_averaging=False` returns; mirrored = forecast of −context, negated back and quantile-flipped; average = what `symmetric_averaging=True` returns.

| pass     |   WAPE ↓ % |   CRPS≈ ↓ MW |   P10-P90 coverage % |   width ↓ MW |
|:---------|-----------:|-------------:|---------------------:|-------------:|
| direct   |      3     |       218.6  |                 79.5 |          844 |
| mirrored |      3.114 |       226.5  |                 79.3 |          873 |
| average  |      3.035 |       220.89 |                 79.8 |          859 |

- Mirrored vs direct, paired over days: ΔWAPE +0.114 pp [+0.049, +0.176], mirrored better on 145/365 days (sign p 0.000102); ΔCRPS≈ +7.90 MW [+3.56, +12.12].
- Correlation of per-point P50 errors, direct vs mirrored: **0.9755** (mean |P50 direct − P50 mirrored| = 62.6 MW; direct RMSE 397 MW).
- Reproduction checks (max |ΔP50| vs the stored runs, MW): direct_vs_timestral-209m-nosym_max_abs_diff_P50_mw = 0, average_vs_timestral-209m_max_abs_diff_P50_mw = 0
