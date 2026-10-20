"""Raw CSVs -> one tidy parquet on the 15-min UTC index + manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from tsbench.config import ROOT, Protocol

log = logging.getLogger(__name__)

LOAD_COLS = {
    "Total Load": "load",
    "Most recent forecast": "most_recent_p50",
    "Most recent P10": "most_recent_p10",
    "Most recent P90": "most_recent_p90",
    "Day-ahead 6PM forecast": "ref_p50",
    "Day-ahead 6PM P10": "ref_p10",
    "Day-ahead 6PM P90": "ref_p90",
    "Week-ahead forecast": "week_ahead_p50",
}
TEMP_COL = "temperature_2m_previous_day1 (°C)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_covariates(index_utc: pd.DatetimeIndex, temp_hourly: pd.Series, tz: str) -> pd.DataFrame:
    local = index_utc.tz_convert(tz)
    be = holidays.Belgium(years=sorted(set(local.year)))
    is_hol = pd.Series(local.date, index=index_utc).map(lambda d: d in be).astype(float)
    # Hourly UTC -> 15-min by linear interpolation; NO extrapolation / bfill of the leading NaN block.
    temp = (
        temp_hourly.reindex(temp_hourly.index.union(index_utc))
        .interpolate(method="time", limit_area="inside")
        .reindex(index_utc)
    )
    return pd.DataFrame(
        {
            "day_of_week": local.dayofweek.astype(float),
            "hour_of_day": local.hour.astype(float),
            "be_holidays": is_hol.to_numpy(),
            "be_temperature": temp.to_numpy(),
        },
        index=index_utc,
    )


def prepare(protocol: Protocol, root: Path = ROOT) -> Path:
    ds = protocol.dataset
    load_p, wx_p = root / ds.raw_load, root / ds.raw_weather
    h_load, h_wx = sha256(load_p), sha256(wx_p)
    for name, got, want in [("load", h_load, ds.raw_load_sha256), ("weather", h_wx, ds.raw_weather_sha256)]:
        if got != want:
            raise RuntimeError(f"{name} SHA-256 differs from protocol.yaml: {got} != {want}. Tell the owner.")

    raw = pd.read_csv(load_p, sep=";", encoding="utf-8-sig")
    raw["ts"] = pd.to_datetime(raw["Datetime"], utc=True)
    df = raw.set_index("ts").sort_index()[list(LOAD_COLS)].rename(columns=LOAD_COLS)
    assert df.index.is_unique and not df.isna().any().any()
    step = pd.Timedelta(minutes=protocol.freq_minutes)
    assert (df.index.to_series().diff().dropna() == step).all(), "gaps in load index"
    df.index = pd.DatetimeIndex(df.index, freq=step)
    assert df.index.tz_convert(protocol.timezone).min() >= pd.Timestamp(ds.min_data_date, tz=protocol.timezone)

    wx = pd.read_csv(wx_p, skiprows=3)
    wx["time"] = pd.to_datetime(wx["time"], utc=True)
    temp_hourly = wx.set_index("time")[TEMP_COL].astype(float)

    out = df.join(build_covariates(df.index, temp_hourly, protocol.timezone))
    out.index.name = "timestamp_utc"
    dest = root / ds.processed
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest)

    first_valid = out["be_temperature"].first_valid_index()
    manifest = {
        "raw_load": {"path": ds.raw_load, "sha256": h_load, "rows": len(df)},
        "raw_weather": {"path": ds.raw_weather, "sha256": h_wx, "rows": len(wx)},
        "processed_sha256": sha256(dest),
        "processed_rows": len(out),
        "index_start_utc": str(out.index.min()),
        "index_end_utc": str(out.index.max()),
        "temperature_first_valid_utc": str(first_valid),
        "temperature_nan_15min_rows": int(out["be_temperature"].isna().sum()),
        "weather_caveat": (
            "Open-Meteo has no issue timestamps; availability of temperature_2m_previous_day1 "
            "at 18:00 the day before cannot be proven."
        ),
        "code_version": git_commit(),
    }
    (root / ds.manifest).write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s (%d rows)", dest, len(out))
    return dest


def load_processed(protocol: Protocol, root: Path = ROOT) -> pd.DataFrame:
    df = pd.read_parquet(root / protocol.dataset.processed)
    assert df.index.tz_convert(protocol.timezone).min() >= pd.Timestamp(
        protocol.dataset.min_data_date, tz=protocol.timezone
    ), "data before min_data_date"
    return df


def data_hash(protocol: Protocol, root: Path = ROOT) -> str:
    return json.loads((root / protocol.dataset.manifest).read_text())["processed_sha256"]


__all__ = ["build_covariates", "data_hash", "load_processed", "np", "prepare"]
