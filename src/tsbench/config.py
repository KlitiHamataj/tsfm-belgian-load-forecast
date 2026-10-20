"""Pydantic schemas for protocol and experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetCfg(_Strict):
    raw_load: str
    raw_weather: str
    processed: str
    manifest: str
    raw_load_sha256: str
    raw_weather_sha256: str
    min_data_date: str


class MetricsCfg(_Strict):
    n_boot: int = 2000
    bootstrap_seed: int = 42
    elia_reference_wape_pct: float
    elia_reference_tolerance: float


class Protocol(_Strict):
    dataset: DatasetCfg
    timezone: str
    freq_minutes: int
    issue_time: str
    bridge_steps: int
    scored_start: str
    scored_end: str
    reference_model: str
    primary_comparison_model: str
    eval_quantiles: list[float]
    interval: tuple[float, float]
    metrics: MetricsCfg
    covariate_sets: dict[str, list[str]]

    @property
    def steps_per_day(self) -> int:
        return 24 * 60 // self.freq_minutes

    @model_validator(mode="after")
    def _check(self) -> Protocol:
        if "none" not in self.covariate_sets or self.covariate_sets["none"]:
            raise ValueError("covariate_sets must define 'none' as an empty list")
        if 0.5 not in self.eval_quantiles:
            raise ValueError("eval_quantiles must contain 0.5")
        return self


class ModelRef(_Strict):
    group: str


class Experiment(_Strict):
    name: str
    models: list[str | ModelRef]
    exclude: list[str] = Field(default_factory=list)
    model_params: dict[str, dict] = Field(default_factory=dict)
    covariate_sets: list[str] = ["none"]
    contexts_days: list[int] = [60]
    context_policy: Literal["skip", "truncate"] = "skip"
    device: Literal["cpu", "cuda"] = "cpu"
    num_threads: int | None = None
    seed: int = 42
    conclusion: str | None = None  # human-written finding, printed at the top of summary.md (not part of run_id)


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_protocol(path: str | Path | None = None) -> Protocol:
    return Protocol(**_load_yaml(path or ROOT / "configs" / "protocol.yaml"))


def load_experiment(path: str | Path) -> Experiment:
    return Experiment(**_load_yaml(path))


def load_groups(path: str | Path | None = None) -> dict[str, list[str]]:
    return _load_yaml(path or ROOT / "configs" / "model_groups.yaml")
