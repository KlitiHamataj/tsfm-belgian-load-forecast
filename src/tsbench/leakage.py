"""Acceptance gates recorded per model in outputs/gates/<model>.json:

- leakage: replace every target value at/after the origin with seeded noise; predictions must not change.
- determinism: the same cell twice on CPU must give bit-identical quantiles.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from tsbench.cards import ModelCard, load_all_cards
from tsbench.config import ROOT, Experiment, load_protocol
from tsbench.data.prepare import load_processed
from tsbench.runner import Cell, resolve_params, run_cell, set_threads, validate_covariate_set
from tsbench.windows import build_task, window_specs

log = logging.getLogger(__name__)
GATES = ROOT / "outputs" / "gates"
TEST_DATES = ["2025-03-15", "2025-03-30", "2025-05-01", "2025-10-26", "2025-12-25"]  # normal, DST x2, holidays


def _setting(card: ModelCard, cov_set: str, ctx_days: int) -> tuple[str, int | None]:
    if card.ignores_context_and_covariates:
        return "default", None
    if not (card.covariates.past and card.covariates.future):
        return "none", ctx_days
    return cov_set, ctx_days


def _record(card: ModelCard, key: str, rec: dict) -> None:
    GATES.mkdir(parents=True, exist_ok=True)
    f = GATES / f"{card.id}.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[key] = {**rec, "revision": card.revision, "adapter_version": card.adapter_version,
                 "recorded_utc": pd.Timestamp.now("UTC").isoformat()}
    f.write_text(json.dumps(data, indent=2))


def load_gates(model_id: str) -> dict:
    f = GATES / f"{model_id}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _load(card: ModelCard, exp: Experiment):
    set_threads(exp.num_threads)
    adapter = card.adapter_class()(card)
    adapter.load(resolve_params(card, exp), "cpu")
    return adapter


def leakage_test(model_id: str, cov_set: str = "full", ctx_days: int = 30, seed: int = 123) -> bool:
    protocol = load_protocol()
    card = load_all_cards()[model_id]
    cov_set, ctx = _setting(card, cov_set, ctx_days)
    if cov_set != "default":
        validate_covariate_set(card, cov_set, protocol)
    df = load_processed(protocol)
    specs = {str(s.scored_date.date()): s for s in window_specs(protocol)}
    exp = Experiment(name="gates", models=[model_id])
    params = resolve_params(card, exp)
    cov_cols = protocol.covariate_sets.get(cov_set, [])
    steps = (ctx or card.adapter_class().required_context_days(params)) * protocol.steps_per_day
    adapter = _load(card, exp)
    rng = np.random.default_rng(seed)
    per_date = {}
    for d in TEST_DATES:
        sp = specs[d]
        noisy = df.copy()
        fut = noisy.index >= sp.origin
        noisy.loc[fut, "load"] = rng.uniform(0, 1e5, fut.sum())
        t_clean = build_task(df, sp, protocol, steps, cov_cols, card.outputs.levels, card.needs_reference)
        t_noisy = build_task(noisy, sp, protocol, steps, cov_cols, card.outputs.levels, card.needs_reference)
        assert t_clean.context.index.max() < sp.origin
        a, b = adapter.predict([t_clean])[0], adapter.predict([t_noisy])[0]
        per_date[d] = all(np.array_equal(a.quantiles[q], b.quantiles[q]) for q in a.quantiles)
        log.info("leakage %s %s: %s", model_id, d, "OK (identical)" if per_date[d] else "FAIL")
    ok = all(per_date.values())
    _record(card, "leakage", {"passed": ok, "covariate_set": cov_set, "context_days": ctx, "dates": per_date,
                              "noise_seed": seed})
    return ok


def determinism_test(model_id: str, cov_set: str = "full", ctx_days: int = 15, n_windows: int = 8) -> bool:
    protocol = load_protocol()
    card = load_all_cards()[model_id]
    cov_set, ctx = _setting(card, cov_set, ctx_days)
    exp = Experiment(name="gates", models=[model_id], covariate_sets=[cov_set if cov_set != "default" else "none"],
                     contexts_days=[ctx or 1])
    df = load_processed(protocol)
    specs = window_specs(protocol)
    adapter = _load(card, exp)
    cell = Cell(model_id, cov_set, ctx)
    runs = [run_cell(adapter, card, cell, exp, protocol, df, specs, "gates", 0.0, 1, limit=n_windows, write=False)["points"]
            for _ in range(2)]
    qcols = [c for c in runs[0].columns if c.startswith("P") and c[1:].isdigit()]
    max_diff = max(float(np.abs(runs[0][c].to_numpy() - runs[1][c].to_numpy()).max()) for c in qcols)
    ok = all(np.array_equal(runs[0][c].to_numpy(), runs[1][c].to_numpy()) for c in qcols)
    log.info("determinism %s: %s (max |diff| %.3g MW)", model_id, "bit-identical" if ok else "DIFFERENT", max_diff)
    _record(card, "determinism", {"passed": ok, "bit_identical": ok, "max_abs_diff_mw": max_diff, "covariate_set": cov_set,
                                  "context_days": ctx, "n_windows": n_windows, "quantiles_compared": len(qcols)})
    return ok
