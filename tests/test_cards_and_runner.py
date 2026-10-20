import pytest

from tsbench.cards import load_all_cards
from tsbench.config import Experiment
from tsbench.registry import expand_models
from tsbench.runner import expand_cells, validate_covariate_set


def test_cards_resolve_and_adapters_import():
    cards = load_all_cards()
    assert {"chronos-2", "timestral-5m", "timestral-28m", "timestral-123m", "timestral-209m", "elia_reference"} <= set(cards)
    for c in cards.values():
        assert 0.5 in c.outputs.levels and c.adapter_class() and c.adapter_version >= 1
        if c.family == "timestral":
            assert c.max_context == 32768 and len(c.outputs.levels) == 21 and len(c.revision) == 40


def test_covariate_set_validation(protocol):
    cards = load_all_cards()
    validate_covariate_set(cards["chronos-2"], "full", protocol)
    with pytest.raises(ValueError):
        validate_covariate_set(cards["last_value"], "full", protocol)
    with pytest.raises(ValueError):
        validate_covariate_set(cards["chronos-2"], "nonexistent", protocol)


def test_groups_exclude_and_context_policy(protocol):
    cards = load_all_cards()
    exp = Experiment(name="t", models=["chronos-2", {"group": "timestral-all"}, {"group": "baselines"}],
                     exclude=["timestral-5m"], covariate_sets=["none", "full"], contexts_days=[60, 90],
                     context_policy="skip")
    ids = expand_models(exp, cards)
    assert "timestral-5m" not in ids and "timestral-28m" in ids
    cells = expand_cells(exp, protocol, cards, ids)
    assert len([c for c in cells if c.model_id == "last_value"]) == 1  # baselines run once
    skipped = [c for c in cells if c.skip_reason]
    assert {(c.model_id, c.context_days) for c in skipped} == {("chronos-2", 90)}  # 8640 > 8192
