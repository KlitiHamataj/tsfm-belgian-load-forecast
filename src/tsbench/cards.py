"""Model cards: YAML front matter (read by code) + human notes."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tsbench.config import ROOT

CARDS_DIR = ROOT / "model_cards"


class Covariates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    past: bool = False
    future: bool = False
    categorical: bool = False


class Outputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str  # "quantiles" | "point"
    levels: list[float]


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    family: str
    kind: str = "model"  # "model" | "baseline"
    display_name: str
    publisher: str
    release_date: str | None = None
    license: str
    source: dict[str, str] = Field(default_factory=dict)
    revision: str  # pinned HF commit hash, or "n/a" for baselines
    params: str
    max_context: int | None  # steps; None = unbounded
    max_horizon_per_pass: int | None = None
    covariates: Covariates
    nan_input: bool = False
    outputs: Outputs
    device: list[str] = ["cpu"]
    adapter: str
    adapter_version: int  # bump whenever adapter behaviour changes -> new run_id
    color: str
    needs_reference: bool = False  # adapter consumes the Elia day-ahead forecast column
    ignores_context_and_covariates: bool = False  # baselines: run once under "default"
    ablation_of: str | None = None  # base model id this card varies (same checkpoint, different settings)
    default_params: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    def adapter_class(self) -> type:
        mod, cls = self.adapter.split(":")
        return getattr(importlib.import_module(mod), cls)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_card(path: Path) -> ModelCard:
    text = path.read_text()
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML front matter")
    _, fm, body = text.split("---", 2)
    data = yaml.safe_load(fm)
    fam = CARDS_DIR / "_families" / f"{data.get('family')}.yaml"
    if fam.exists():
        data = _deep_merge(yaml.safe_load(fam.read_text()), data)
    card = ModelCard(**data)
    card.notes = body.strip()
    if card.id != path.stem:
        raise ValueError(f"{path}: card id {card.id!r} must equal file name")
    return card


def load_all_cards() -> dict[str, ModelCard]:
    return {
        c.id: c
        for c in (load_card(p) for p in sorted(CARDS_DIR.glob("*.md")) if not p.name.startswith("_"))
    }
