"""Window builder. The ONLY place that slices data, so leakage protection lives here."""

from __future__ import annotations

from dataclasses import dataclass

import holidays
import pandas as pd

from tsbench.config import Protocol
from tsbench.models.base import ForecastTask


@dataclass(frozen=True)
class WindowSpec:
    scored_date: pd.Timestamp  # local date (tz-naive midnight) of the scored day
    origin: pd.Timestamp  # UTC, 18:00 local on the previous local day
    horizon: int  # bridge_steps + steps in the scored local day (DST aware)
    scored_steps: int
    is_holiday: bool
    is_weekend: bool

    @property
    def origin_label(self) -> str:
        return str(self.origin.tz_convert("Europe/Brussels"))


def window_specs(p: Protocol) -> list[WindowSpec]:
    tz, step = p.timezone, pd.Timedelta(minutes=p.freq_minutes)
    be = holidays.Belgium(years=range(pd.Timestamp(p.scored_start).year, pd.Timestamp(p.scored_end).year + 1))
    hh, mm = map(int, p.issue_time.split(":"))
    out = []
    for d in pd.date_range(p.scored_start, p.scored_end, freq="D"):
        prev = d - pd.Timedelta(days=1)
        origin_local = pd.Timestamp(year=prev.year, month=prev.month, day=prev.day, hour=hh, minute=mm, tz=tz)
        day_start = d.tz_localize(tz)
        day_end = (d + pd.Timedelta(days=1)).tz_localize(tz)  # local midnight after the scored day
        origin = origin_local.tz_convert("UTC")
        scored_steps = int((day_end.tz_convert("UTC") - day_start.tz_convert("UTC")) / step)
        bridge = int((day_start.tz_convert("UTC") - origin) / step)
        assert bridge == p.bridge_steps, f"bridge {bridge} != configured {p.bridge_steps} on {d.date()}"
        out.append(
            WindowSpec(
                scored_date=d,
                origin=origin,
                horizon=bridge + scored_steps,
                scored_steps=scored_steps,
                is_holiday=d.date() in be,
                is_weekend=d.dayofweek >= 5,
            )
        )
    return out


def build_task(
    df: pd.DataFrame,
    spec: WindowSpec,
    p: Protocol,
    ctx_steps: int,
    cov_cols: list[str],
    quantiles: list[float],
    with_reference: bool = False,
) -> ForecastTask:
    step = pd.Timedelta(minutes=p.freq_minutes)
    h_idx = pd.DatetimeIndex([spec.origin + i * step for i in range(spec.horizon)])
    c_idx = pd.DatetimeIndex([spec.origin - (ctx_steps - i) * step for i in range(ctx_steps)])
    assert c_idx[-1] < spec.origin and c_idx[-1] + step == spec.origin, "context must end at 17:45"
    assert c_idx[0].tz_convert(p.timezone) >= pd.Timestamp(p.dataset.min_data_date, tz=p.timezone), (
        f"context for {spec.scored_date.date()} reaches before {p.dataset.min_data_date}"
    )
    context = df["load"].reindex(c_idx)
    assert context.notna().all(), "NaN in target context"
    past = fut = None
    if cov_cols:
        past, fut = df.loc[c_idx, cov_cols], df.loc[h_idx, cov_cols]
        assert not past.isna().any().any(), f"NaN in past covariates ({spec.scored_date.date()})"
        assert not fut.isna().any().any(), f"NaN in future covariates ({spec.scored_date.date()})"
    ref = None
    if with_reference:
        ref = df.loc[h_idx, ["ref_p10", "ref_p50", "ref_p90"]].copy()
    return ForecastTask(
        origin=spec.origin,
        context=context,
        past_covariates=past,
        future_covariates=fut,
        horizon=spec.horizon,
        quantiles=quantiles,
        horizon_index=h_idx,
        reference=ref,
        tz=p.timezone,
    )


def scored_mask(spec: WindowSpec, p: Protocol) -> pd.Series:
    """Boolean per horizon step: True where the step is in the scored local day."""
    step = pd.Timedelta(minutes=p.freq_minutes)
    h_idx = pd.DatetimeIndex([spec.origin + i * step for i in range(spec.horizon)])
    local = h_idx.tz_convert(p.timezone)
    return pd.Series(local.normalize().tz_localize(None) == spec.scored_date, index=h_idx)
