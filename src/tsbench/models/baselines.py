"""Baselines as models, so they flow through the same code path.

Seasonal lookups use local wall-clock time (same quarter-hour of the day on previous local days)
and only see the context, so a source timestamp >= origin is unavailable by construction.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache

import holidays
import numpy as np
import pandas as pd

from tsbench.models.base import BaseAdapter, Forecast, ForecastTask


def _wall_lookup(ctx: pd.Series, tz: str) -> dict[pd.Timestamp, float]:
    """local naive wall time -> mean value (mean handles the repeated autumn hour)."""
    wall = ctx.index.tz_convert(tz).tz_localize(None)
    return ctx.groupby(wall).mean().to_dict()


def _horizon_wall(t: ForecastTask) -> pd.DatetimeIndex:
    return t.horizon_index.tz_convert(t.tz).tz_localize(None)


def _fill(v: np.ndarray, last: float) -> np.ndarray:
    """ffill gaps (e.g. the skipped spring hour on the source day), fall back to last observed."""
    s = pd.Series(v).ffill().bfill()
    return s.fillna(last).to_numpy()


def _forecast(t: ForecastTask, p50: np.ndarray) -> Forecast:
    return Forecast(
        index=t.horizon_index,
        quantiles={0.5: p50.astype(np.float64)},
        effective_context=len(t.context),
        context_finite=int(t.context.notna().sum()),
    )


class LastValueAdapter(BaseAdapter):
    @classmethod
    def required_context_days(cls, params: dict) -> int:
        return 1

    def predict(self, batch):
        return [_forecast(t, np.full(t.horizon, t.context.iloc[-1])) for t in batch]


class WeeklyPersistenceAdapter(BaseAdapter):
    @classmethod
    def required_context_days(cls, params: dict) -> int:
        return 8

    def predict(self, batch):
        out = []
        for t in batch:
            look = _wall_lookup(t.context, t.tz)
            v = np.array([look.get(w - pd.Timedelta(days=7), np.nan) for w in _horizon_wall(t)])
            out.append(_forecast(t, _fill(v, t.context.iloc[-1])))
        return out


@cache
def _be_holidays(year: int) -> frozenset[date]:
    return frozenset(holidays.Belgium(years=year))


def day_type(d: date) -> str:
    """workday (Mon-Fri, not a public holiday) | saturday | sunday_holiday (Sunday or BE public holiday)."""
    if d.weekday() == 6 or d in _be_holidays(d.year):
        return "sunday_holiday"
    return "saturday" if d.weekday() == 5 else "workday"


DAY_MATCHING = ("all", "same_weekday", "day_type")


def _day_matches(target: date, source: date, rule: str) -> bool:
    if rule == "all":
        return True
    if rule == "same_weekday":
        return target.weekday() == source.weekday()
    if rule == "day_type":
        return day_type(target) == day_type(source)
    raise ValueError(f"day_matching must be one of {DAY_MATCHING}, got {rule!r}")


class _SeasonalBase(BaseAdapter):
    """Average of the same local quarter-hour over the `n_days` most recent MATCHING days on which that
    quarter-hour was observed before the origin (18:00). Lags searched: 1 .. max_lag_days."""

    default_n = 7
    default_matching = "all"

    @classmethod
    def _cfg(cls, params: dict) -> tuple[int, str]:
        return int(params.get("n_days", cls.default_n)), str(params.get("day_matching", cls.default_matching))

    @classmethod
    def max_lag_days(cls, params: dict) -> int:
        n, rule = cls._cfg(params)
        return n + 2 if rule == "all" else 7 * n + 2  # +2: lag 1 (and 2 near DST) may be unavailable

    @classmethod
    def required_context_days(cls, params: dict) -> int:
        return cls.max_lag_days(params)

    def _weights(self, k: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict(self, batch):
        n, rule = self._cfg(self.params)
        max_lag = self.max_lag_days(self.params)
        out = []
        for t in batch:
            look = _wall_lookup(t.context, t.tz)
            walls = _horizon_wall(t)
            # lags (in days) whose source day matches each target day, most recent first
            lags = {
                d: [k for k in range(1, max_lag + 1) if _day_matches(d, d - timedelta(days=k), rule)]
                for d in {w.date() for w in walls}
            }
            v = np.full(t.horizon, np.nan)
            for i, w in enumerate(walls):
                vals = []
                for k in lags[w.date()]:
                    x = look.get(w - pd.Timedelta(days=k), np.nan)
                    if not np.isnan(x):
                        vals.append(x)
                        if len(vals) == n:
                            break
                if vals:
                    v[i] = np.average(vals, weights=self._weights(len(vals)))
            out.append(_forecast(t, _fill(v, t.context.iloc[-1])))
        return out


class DailyPersistenceAdapter(_SeasonalBase):
    """Most recent observed value at the same local quarter-hour (n_days = 1, any day)."""

    @classmethod
    def _cfg(cls, params: dict) -> tuple[int, str]:
        return 1, "all"

    def _weights(self, k):
        return np.ones(k)


class SeasonalMovingAverageAdapter(_SeasonalBase):
    default_n = 7

    def _weights(self, k):
        return np.ones(k)


class ExpWeightedSeasonalAdapter(_SeasonalBase):
    default_n = 28

    def _weights(self, k):
        a = float(self.params.get("alpha", 0.3))
        return a * (1 - a) ** np.arange(k)  # most recent first


class EliaReferenceAdapter(BaseAdapter):
    @classmethod
    def required_context_days(cls, params: dict) -> int:
        return 1

    def predict(self, batch):
        out = []
        for t in batch:
            assert t.reference is not None
            r = t.reference
            out.append(
                Forecast(
                    index=t.horizon_index,
                    quantiles={
                        0.1: r["ref_p10"].to_numpy(),
                        0.5: r["ref_p50"].to_numpy(),
                        0.9: r["ref_p90"].to_numpy(),
                    },
                    effective_context=0,  # context is not used
                    context_finite=0,
                )
            )
        return out
