"""Run matrix expansion, run_id, cell execution, resume."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pandas as pd
import psutil
import torch
import yaml
from tqdm import tqdm

from tsbench import metrics as M
from tsbench.cards import ModelCard
from tsbench.config import ROOT, Experiment, Protocol
from tsbench.data.prepare import data_hash, git_commit, load_processed
from tsbench.models.base import BaseAdapter, Forecast
from tsbench.windows import WindowSpec, build_task, scored_mask, window_specs

log = logging.getLogger(__name__)
RUNS = ROOT / "outputs" / "runs"


@dataclass(frozen=True)
class Cell:
    model_id: str
    covariate_set: str  # "default" for baselines
    context_days: int | None  # None for baselines w/o learned context -> adapter-defined
    skip_reason: str | None = None

    @property
    def folder(self) -> Path:
        ctx = "default" if self.covariate_set == "default" else f"ctx{self.context_days:02d}d"
        return RUNS / self.model_id / self.covariate_set / ctx if self.covariate_set != "default" else (
            RUNS / self.model_id / "default"
        )

    @property
    def label(self) -> str:
        return f"{self.model_id}/{self.covariate_set}/" + (
            "-" if self.covariate_set == "default" else f"ctx{self.context_days}d"
        )


def expand_cells(exp: Experiment, protocol: Protocol, cards: dict[str, ModelCard], model_ids: list[str]) -> list[Cell]:
    cells: list[Cell] = []
    for mid in model_ids:
        card = cards[mid]
        if card.ignores_context_and_covariates:
            cells.append(Cell(mid, "default", None))
            continue
        for cs in exp.covariate_sets:
            validate_covariate_set(card, cs, protocol)  # refuse, never silently drop
            for d in exp.contexts_days:
                reason = None
                if card.max_context is not None and d * protocol.steps_per_day > card.max_context:
                    if exp.context_policy == "skip":
                        reason = f"context {d}d = {d * protocol.steps_per_day} steps > max_context {card.max_context}"
                cells.append(Cell(mid, cs, d, reason))
    return cells


def validate_covariate_set(card: ModelCard, cs: str, protocol: Protocol) -> None:
    if cs not in protocol.covariate_sets:
        raise ValueError(f"unknown covariate set {cs!r}; defined: {list(protocol.covariate_sets)}")
    if protocol.covariate_sets[cs] and not (card.covariates.future and card.covariates.past):
        raise ValueError(f"model {card.id} cannot use covariate set {cs!r} (card.covariates={card.covariates})")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def make_run_id(card: ModelCard, cell: Cell, params: dict, exp: Experiment, protocol: Protocol, dhash: str) -> str:
    return _sha(
        {
            "model": card.id,
            "revision": card.revision,
            "adapter_version": card.adapter_version,
            "params": params,
            "covariate_set": cell.covariate_set,
            "covariates": protocol.covariate_sets.get(cell.covariate_set, []),
            "context_days": cell.context_days,
            # baselines ignore context/covariates, so the policy must not split their cell across experiments
            "context_policy": None if card.ignores_context_and_covariates else exp.context_policy,
            "device": exp.device,
            "seed": exp.seed,
            "protocol": protocol.model_dump(exclude={"metrics": {"n_boot", "bootstrap_seed"}}),
            "data": dhash,
        }
    )


def cell_complete(cell: Cell, run_id: str, n_expected: int) -> bool:
    f = cell.folder
    try:
        meta = json.loads((f / "run_meta.json").read_text())
        m = json.loads((f / "metrics.json").read_text())
    except Exception:
        return False
    return (
        meta.get("run_id") == run_id
        and m.get("n_windows") == n_expected
        and (f / "points.parquet").exists()
        and (f / "windows.csv").exists()
    )


def resolve_params(card: ModelCard, exp: Experiment) -> dict:
    return {**card.default_params, **exp.model_params.get(card.id, {})}


def effective_ctx_days(card: ModelCard, cell: Cell, params: dict) -> int:
    if cell.context_days is not None:
        return cell.context_days
    req = card.adapter_class().required_context_days(params)
    assert req is not None
    return req


def set_threads(n: int | None) -> int:
    n = n or psutil.cpu_count(logical=False) or 1
    torch.set_num_threads(n)
    return n


def package_versions() -> dict:
    out = {}
    for p in ["torch", "chronos-forecasting", "timestral", "pandas", "numpy", "transformers"]:
        try:
            out[p] = metadata.version(p)
        except Exception:
            pass
    return out


def run_cell(
    adapter: BaseAdapter,
    card: ModelCard,
    cell: Cell,
    exp: Experiment,
    protocol: Protocol,
    df: pd.DataFrame,
    specs: list[WindowSpec],
    run_id: str,
    load_seconds: float,
    threads: int,
    limit: int | None = None,
    write: bool = True,
) -> dict:
    params = resolve_params(card, exp)
    ctx_days = effective_ctx_days(card, cell, params)
    ctx_steps = ctx_days * protocol.steps_per_day
    cov_cols = protocol.covariate_sets.get(cell.covariate_set, []) if cell.covariate_set != "default" else []
    specs = specs[:limit] if limit else specs
    chunk = int(params.get("batch_size", 64)) if not card.ignores_context_and_covariates else 64
    torch.manual_seed(exp.seed)

    pts, wins = [], []
    t_inf = 0.0
    bar = tqdm(range(0, len(specs), chunk), desc=cell.label, leave=False)
    for s in bar:
        sub = specs[s : s + chunk]
        tasks = [
            build_task(df, sp, protocol, ctx_steps, cov_cols, card.outputs.levels, card.needs_reference)
            for sp in sub
        ]
        t0 = time.perf_counter()
        fcs: list[Forecast] = adapter.predict(tasks)
        t_inf += time.perf_counter() - t0
        for sp, task, fc in zip(sub, tasks, fcs):
            assert 0.5 in fc.quantiles and len(fc.index) == sp.horizon
            assert (fc.index == task.horizon_index).all()
            mask = scored_mask(sp, protocol).to_numpy()
            od = str(sp.origin.tz_convert(protocol.timezone).date())
            frame = pd.DataFrame(
                {
                    "dataset": "elia_load",
                    "model": card.id,
                    "context": ctx_days if cell.covariate_set != "default" else None,
                    "covariate_set": cell.covariate_set,
                    "origin": sp.origin,
                    "origin_date": od,
                    "timestamp": fc.index,
                    "step": range(sp.horizon),
                    "is_scored": mask,
                    "actual": df.loc[fc.index, "load"].to_numpy(),
                }
            )
            for q, v in sorted(fc.quantiles.items()):
                frame[f"P{round(q * 100):02d}"] = v
            pts.append(frame)
            wins.append(
                {
                    "origin_date": od,
                    "scored_date": str(sp.scored_date.date()),
                    "is_holiday": sp.is_holiday,
                    "is_weekend": sp.is_weekend,
                    "horizon": sp.horizon,
                    "context_steps_configured": ctx_steps,
                    "context_steps_effective": fc.effective_context,
                    "context_finite_effective": fc.context_finite,
                }
            )
    points = pd.concat(pts, ignore_index=True)
    meta = pd.DataFrame(wins).set_index("origin_date")
    windows = M.window_table(points, meta)
    windows.insert(0, "method", card.id)
    mets = M.compute_metrics(
        points, windows, protocol.eval_quantiles, protocol.interval, protocol.metrics.n_boot, protocol.metrics.bootstrap_seed
    )
    eff = windows["context_steps_effective"]
    run_meta = {
        "run_id": run_id,
        "model_id": card.id,
        "family": card.family,
        "variant": card.display_name,
        "hf_id": card.source.get("hf_id"),
        "revision": card.revision,
        "adapter_version": card.adapter_version,
        "params": params,
        "covariate_set": cell.covariate_set,
        "covariates": cov_cols,
        "context_days_configured": ctx_days if cell.covariate_set != "default" else None,
        "context_steps_configured": ctx_steps,
        "context_steps_effective_min": int(eff.min()),
        "context_steps_effective_max": int(eff.max()),
        "context_days_effective": float(eff.max() / protocol.steps_per_day),
        "context_policy": exp.context_policy,
        "device": exp.device,
        "num_threads": threads,
        "batch_size": chunk,
        "seed": exp.seed,
        "n_windows": len(windows),
        "timings": {
            "load_seconds": load_seconds,
            "inference_seconds": t_inf,
            "per_window_seconds": t_inf / len(windows),
        },
        "data_hash_processed": data_hash(protocol),
        "raw_load_sha256": protocol.dataset.raw_load_sha256,
        "raw_weather_sha256": protocol.dataset.raw_weather_sha256,
        "weather_caveat": "Open-Meteo has no issue timestamps; availability at 18:00 the day before cannot be proven.",
        "packages": package_versions(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "adapter_runtime": adapter.runtime_info(),
        "finished_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    if write:
        f = cell.folder
        f.mkdir(parents=True, exist_ok=True)
        points.to_parquet(f / "points.parquet")
        windows.to_csv(f / "windows.csv", index=False)
        (f / "metrics.json").write_text(json.dumps(mets, indent=2))
        (f / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))
        (f / "resolved_config.yaml").write_text(
            yaml.safe_dump({"cell": cell.label, "experiment": exp.model_dump(), "protocol": protocol.model_dump()})
        )
    return {"metrics": mets, "run_meta": run_meta, "points": points, "windows": windows}


def run_model_cells(exp: Experiment, protocol: Protocol, card: ModelCard, cells: list[Cell], status_cb) -> None:
    """All cells of one model in series, loading the model once. Failures are isolated per cell."""
    df = load_processed(protocol)
    specs = window_specs(protocol)
    dhash = data_hash(protocol)
    params = resolve_params(card, exp)
    todo = []
    for c in cells:
        rid = make_run_id(card, c, params, exp, protocol, dhash)
        if c.skip_reason:
            status_cb(c, rid, "skipped", c.skip_reason, 0.0)
        elif cell_complete(c, rid, len(specs)):
            status_cb(c, rid, "skipped", "complete (run_id match)", 0.0)
        else:
            todo.append((c, rid))
    if not todo:
        return
    threads = set_threads(exp.num_threads)
    adapter = card.adapter_class()(card)
    t0 = time.perf_counter()
    try:
        adapter.load(params, exp.device)
    except Exception as e:
        log.exception("load failed for %s", card.id)
        for c, rid in todo:
            status_cb(c, rid, "failed", f"load: {e!r}", 0.0)
        return
    load_s = time.perf_counter() - t0
    for c, rid in todo:
        t1 = time.perf_counter()
        try:
            res = run_cell(adapter, card, c, exp, protocol, df, specs, rid, load_s, threads)
            status_cb(c, rid, "ok", f"WAPE {res['metrics']['wape_pct']:.2f}%", time.perf_counter() - t1)
        except Exception as e:
            log.exception("cell failed: %s", c.label)
            status_cb(c, rid, "failed", repr(e), time.perf_counter() - t1)
