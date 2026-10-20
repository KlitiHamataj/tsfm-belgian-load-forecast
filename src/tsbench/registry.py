"""Model registry: cards on disk are the source of truth; groups come from YAML."""

from __future__ import annotations

from tsbench.cards import ModelCard, load_all_cards
from tsbench.config import Experiment, ModelRef, load_groups


def expand_models(exp: Experiment, cards: dict[str, ModelCard] | None = None) -> list[str]:
    cards = cards or load_all_cards()
    groups = load_groups()
    ids: list[str] = []
    for m in exp.models:
        names = groups[m.group] if isinstance(m, ModelRef) else [m]
        if isinstance(m, ModelRef) and m.group not in groups:
            raise KeyError(f"unknown group {m.group!r}")
        for n in names:
            if n not in cards:
                raise KeyError(f"unknown model id {n!r}; see `bench list-models`")
            if n not in ids and n not in exp.exclude:
                ids.append(n)
    return ids
