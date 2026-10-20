"""Common model interface. Adapters never see the full dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol as TypingProtocol

import numpy as np
import pandas as pd

from tsbench.cards import ModelCard


@dataclass
class ForecastTask:
    origin: pd.Timestamp  # 18:00 local, as UTC
    context: pd.Series  # target strictly before origin (UTC index), last value 17:45 local
    past_covariates: pd.DataFrame | None  # over the context index
    future_covariates: pd.DataFrame | None  # over the horizon index
    horizon: int
    quantiles: list[float]
    horizon_index: pd.DatetimeIndex
    reference: pd.DataFrame | None = None  # Elia day-ahead p10/p50/p90 over horizon (only if card.needs_reference)
    tz: str = "Europe/Brussels"


@dataclass
class Forecast:
    index: pd.DatetimeIndex
    quantiles: dict[float, np.ndarray]  # must include 0.5
    effective_context: int  # context steps ACTUALLY fed to the model (measured by the adapter)
    context_finite: int = 0  # of those, non-NaN
    extra: dict = field(default_factory=dict)


class ForecastModel(TypingProtocol):
    card: ModelCard

    def load(self, params: dict, device: str) -> None: ...

    def predict(self, batch: list[ForecastTask]) -> list[Forecast]: ...


class BaseAdapter:
    def __init__(self, card: ModelCard):
        self.card = card
        self.params: dict = {}

    def load(self, params: dict, device: str) -> None:
        self.params = {**self.card.default_params, **params}
        self.device = device

    @classmethod
    def required_context_days(cls, params: dict) -> int | None:
        """Baselines return the context they need; learned models return None (config decides)."""
        return None

    def predict(self, batch: list[ForecastTask]) -> list[Forecast]:  # pragma: no cover
        raise NotImplementedError

    def runtime_info(self) -> dict:
        return {}
