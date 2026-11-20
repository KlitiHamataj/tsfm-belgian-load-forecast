"""Export scored 2025 points of selected run cells for the static dashboard (docs/dashboard/).

data.json  : 15-min actual + P10/P50/P90 per series on a regular UTC grid (timestamps are rebuilt from
             `start_utc` + i * `step_min`; the page converts them to Europe/Brussels). Values in integer MW.
daily.json : per local scored day: steps (92/96/100), holiday/weekend flags, and per series pooled WAPE with the
             sums it is built from, so any set of days can be pooled exactly.
Only `is_scored` rows are exported (the 24 bridge steps are never shown or scored).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from tsbench.cards import ModelCard, load_all_cards
from tsbench.config import ROOT, Protocol
from tsbench.runner import Cell

log = logging.getLogger(__name__)
DEFAULT_CONFIG = ROOT / "configs" / "dashboard.yaml"


class SeriesRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    covariates: str
    context_days: int | None = None
    label: str | None = None
    short: str
    default: bool = False

    @property
    def cell(self) -> Cell:
        return Cell(self.model, self.covariates, self.context_days)


class DashboardCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series: list[SeriesRef]
    quantiles: list[str]
    check_against: str
    tolerance_pp: float
    out_dir: str


def load_dashboard_cfg(path: str | Path | None = None) -> DashboardCfg:
    return DashboardCfg(**yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text()))


def pooled_wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.abs(forecast - actual).sum() / np.abs(actual).sum() * 100)


def daily_table(local_dates: pd.Series, actual: np.ndarray, p50: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per local date: n_steps and, per series, sum|err|, sum|actual| and pooled WAPE (%)."""
    df = pd.DataFrame({"date": local_dates.to_numpy(), "abs_actual": np.abs(actual)})
    for sid, f in p50.items():
        df[f"err:{sid}"] = np.abs(f - actual)
    g = df.groupby("date", sort=True)
    out = pd.DataFrame({"n_steps": g.size(), "sum_actual": g["abs_actual"].sum()})
    for sid in p50:
        out[f"sum_abs_err:{sid}"] = g[f"err:{sid}"].sum()
        out[f"wape:{sid}"] = out[f"sum_abs_err:{sid}"] / out["sum_actual"] * 100
    return out.reset_index()


def typical_week(daily: dict, series: list[str]) -> tuple[str, str, float]:
    """Monday-Sunday week (fully inside the data, no public holiday) whose pooled WAPE is closest to the full-year
    WAPE, summed over `series`. Returns (monday, sunday, score in pp); ties go to the earliest week."""
    dates = pd.to_datetime(pd.Series(daily["dates"]))
    act = np.asarray(daily["sum_actual"])
    err = {s: np.asarray(daily["sum_abs_err"][s]) for s in series}
    full = {s: err[s].sum() / act.sum() * 100 for s in series}
    hol = np.asarray(daily["is_holiday"], bool)
    best: tuple[float, int] | None = None
    for i in np.flatnonzero(dates.dt.dayofweek.to_numpy() == 0):
        if i + 7 > len(dates) or (dates[i + 6] - dates[i]).days != 6 or hol[i:i + 7].any():
            continue
        score = sum(abs(err[s][i:i + 7].sum() / act[i:i + 7].sum() * 100 - full[s]) for s in series)
        if best is None or score < best[0] - 1e-12:
            best = (float(score), int(i))
    if best is None:
        raise ValueError("no holiday-free Monday-Sunday week in the data")
    return daily["dates"][best[1]], daily["dates"][best[1] + 6], best[0]


def _load_scored(cell: Cell, quantiles: list[str]) -> pd.DataFrame:
    path = cell.folder / "points.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the cell {cell.label} first (outputs/ is not committed)")
    d = pd.read_parquet(path, columns=["timestamp", "origin_date", "is_scored", "actual", *quantiles])
    d = d[d.is_scored].sort_values("timestamp").reset_index(drop=True)
    if d.origin_date.nunique() != 365:
        raise ValueError(f"{cell.label}: {d.origin_date.nunique()}/365 windows")
    return d


def _expected_wape(cfg: DashboardCfg, s: SeriesRef) -> float:
    w = pd.read_csv(ROOT / cfg.check_against, dtype=str)
    cov = "-" if s.covariates == "default" else s.covariates
    ctx = "-" if s.context_days is None else str(s.context_days)
    row = w[(w.model == s.model) & (w.covariates == cov) & (w["ctx (days)"] == ctx)]
    if len(row) != 1:
        raise ValueError(f"{s.cell.label}: expected one row in {cfg.check_against}, found {len(row)}")
    return float(row.wape_pct.iloc[0])


def _caveat(card: ModelCard, protocol: Protocol) -> bool:
    return bool(card.weights_date and card.weights_date > protocol.scored_end and card.training_data_disclosed is False)


def export_dashboard(protocol: Protocol, cfg: DashboardCfg | None = None) -> dict[str, float]:
    cfg = cfg or load_dashboard_cfg()
    cards = load_all_cards()
    frames = {s.model: _load_scored(s.cell, cfg.quantiles) for s in cfg.series}
    first = next(iter(frames.values()))
    ts, actual = first.timestamp, first.actual.to_numpy()
    for sid, d in frames.items():
        if not (d.timestamp.equals(ts) and np.allclose(d.actual.to_numpy(), actual)):
            raise ValueError(f"{sid}: timestamps or actuals differ from {next(iter(frames))}")
    step = ts.diff().dropna().unique()
    if len(step) != 1 or step[0] != pd.Timedelta(minutes=protocol.freq_minutes):
        raise ValueError("scored points are not on a regular grid")

    act_i = np.rint(actual).astype(int)
    series, p50 = {}, {}
    for s in cfg.series:
        card, d = cards[s.model], frames[s.model]
        q = {k: np.rint(d[k].to_numpy()).astype(int) for k in cfg.quantiles}
        p50[s.model] = q["P50"]
        flag = _caveat(card, protocol)
        label = s.label or f"{card.display_name} ({s.context_days} days)"
        series[s.model] = {
            "label": label + (" †" if flag else ""), "short": s.short + (" †" if flag else ""),
            "cell": s.cell.label, "color": card.color, "default": s.default, "contamination_caveat": flag,
            **{k: v.tolist() for k, v in q.items()},
        }
    local = ts.dt.tz_convert(protocol.timezone)
    data = {
        "start_utc": ts.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"), "step_min": protocol.freq_minutes, "n": len(ts),
        "timezone": protocol.timezone, "unit": "MW", "actual": act_i.tolist(), "series": series,
    }

    day = daily_table(local.dt.strftime("%Y-%m-%d"), act_i.astype(float), {k: v.astype(float) for k, v in p50.items()})
    flags = pd.read_csv(cfg.series[0].cell.folder / "windows.csv", usecols=["scored_date", "is_holiday", "is_weekend"])
    day = day.merge(flags, left_on="date", right_on="scored_date", how="left").drop(columns="scored_date")
    if day.is_holiday.isna().any():
        raise ValueError("windows.csv does not cover every scored day")
    names = holidays.Belgium(years=sorted({int(x[:4]) for x in day.date}))
    daily = {
        "series": list(p50), "dates": day.date.tolist(), "n_steps": day.n_steps.astype(int).tolist(),
        "is_holiday": day.is_holiday.astype(bool).tolist(), "is_weekend": day.is_weekend.astype(bool).tolist(),
        "holiday_name": [names.get(pd.Timestamp(x).date()) for x in day.date],
        "sum_actual": day.sum_actual.round(1).tolist(),
        "sum_abs_err": {k: day[f"sum_abs_err:{k}"].round(1).tolist() for k in p50},
        "wape": {k: day[f"wape:{k}"].round(3).tolist() for k in p50},
    }

    defaults = [s.model for s in cfg.series if s.default]
    mon, sun, score = typical_week(daily, defaults)
    data["default_view"] = {"start": mon, "end": sun, "rule": "typical_week", "series": defaults,
                            "score_pp": round(score, 4)}
    log.info("typical week: %s to %s (sum |week - year| WAPE = %.3f pp over %s)", mon, sun, score, defaults)

    checks = {}
    for s in cfg.series:
        got, exp = pooled_wape(act_i.astype(float), p50[s.model].astype(float)), _expected_wape(cfg, s)
        if abs(got - exp) > cfg.tolerance_pp:
            raise ValueError(f"{s.cell.label}: exported WAPE {got:.4f}% vs {cfg.check_against} {exp:.4f}%")
        checks[s.model] = got

    out = ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    (out / "daily.json").write_text(json.dumps(daily, separators=(",", ":")))
    log.info("wrote %s (%.2f MB) and daily.json", (out / "data.json").relative_to(ROOT),
             (out / "data.json").stat().st_size / 1e6)
    return checks
