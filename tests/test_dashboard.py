"""Dashboard export: committed docs/dashboard/*.json must reproduce the published full-year WAPE (runs in CI)."""

import json

import numpy as np
import pandas as pd
import pytest

from tsbench.config import ROOT
from tsbench.dashboard import _expected_wape, daily_table, load_dashboard_cfg, pooled_wape, typical_week

CFG = load_dashboard_cfg()
OUT = ROOT / CFG.out_dir


@pytest.fixture(scope="module")
def data():
    return json.loads((OUT / "data.json").read_text())


@pytest.fixture(scope="module")
def daily():
    return json.loads((OUT / "daily.json").read_text())


def test_data_covers_2025_on_a_regular_grid(data):
    assert data["n"] == len(data["actual"]) == 35040
    start = pd.Timestamp(data["start_utc"])
    local = (start + pd.to_timedelta(np.arange(data["n"]) * data["step_min"], "min")).tz_convert(data["timezone"])
    assert local[0] == pd.Timestamp("2025-01-01 00:00", tz=data["timezone"])
    assert local[-1] == pd.Timestamp("2025-12-31 23:45", tz=data["timezone"])
    assert list(data["series"]) == [s.model for s in CFG.series]


@pytest.mark.parametrize("ref", CFG.series, ids=lambda s: s.cell.label)
def test_full_year_wape_matches_results(data, ref):
    a = np.asarray(data["actual"], float)
    f = np.asarray(data["series"][ref.model]["P50"], float)
    assert pooled_wape(a, f) == pytest.approx(_expected_wape(CFG, ref), abs=CFG.tolerance_pp)


def test_daily_sums_pool_back_to_full_year(data, daily):
    assert len(daily["dates"]) == 365 and sum(daily["n_steps"]) == data["n"]
    assert sorted(set(daily["n_steps"])) == [92, 96, 100]
    a = np.asarray(data["actual"], float)
    for sid in daily["series"]:
        f = np.asarray(data["series"][sid]["P50"], float)
        pooled = sum(daily["sum_abs_err"][sid]) / sum(daily["sum_actual"]) * 100
        assert pooled == pytest.approx(pooled_wape(a, f), abs=1e-4)


def test_daily_table_handles_dst_lengths_and_pools_per_day():
    dates = pd.Series(["d92"] * 92 + ["d96"] * 96 + ["d100"] * 100)
    actual = np.r_[np.full(92, 100.0), np.full(96, 200.0), np.full(100, 50.0)]
    f = actual.copy()
    f[:92] += 10  # 10% everywhere on the short day
    f[92:92 + 48] -= 40  # half of the normal day off by 20%
    t = daily_table(dates, actual, {"m": f}).set_index("date")
    assert t.loc[["d92", "d96", "d100"], "n_steps"].tolist() == [92, 96, 100]
    assert t.loc["d92", "wape:m"] == pytest.approx(10.0)
    assert t.loc["d96", "wape:m"] == pytest.approx(10.0)  # pooled: 48*40 / (96*200)
    assert t.loc["d100", "wape:m"] == 0.0
    assert t["sum_abs_err:m"].sum() / t["sum_actual"].sum() * 100 == pytest.approx(pooled_wape(actual, f))


def test_default_view_is_the_rule_based_typical_week(data, daily):
    v = data["default_view"]
    defaults = [s.model for s in CFG.series if s.default]
    assert v["series"] == defaults and [s for s, d in data["series"].items() if d["default"]] == defaults
    mon, sun, _ = typical_week(daily, defaults)
    assert (v["start"], v["end"]) == (mon, sun)
    assert pd.Timestamp(mon).dayofweek == 0 and pd.Timestamp(sun) - pd.Timestamp(mon) == pd.Timedelta(days=6)
    assert v["day"] == (pd.Timestamp(mon) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")  # Wednesday of that week
    i = daily["dates"].index(mon)
    assert not any(daily["is_holiday"][i:i + 7])


def test_typical_week_rule_on_synthetic_data():
    dates = pd.date_range("2025-01-06", periods=28).strftime("%Y-%m-%d").tolist()  # four Monday-Sunday weeks
    week_wape = [5.0, 2.0, 3.1, 3.0]  # week 4 is closest to the year but contains a holiday
    err = [w for w in week_wape for _ in range(7)]
    hol = [False] * 21 + [False, False, True, False, False, False, False]
    daily = {"dates": dates, "sum_actual": [100.0] * 28, "sum_abs_err": {"m": err}, "is_holiday": hol}
    year = sum(err) / 2800 * 100  # 3.275
    mon, sun, score = typical_week(daily, ["m"])
    assert (mon, sun) == ("2025-01-20", "2025-01-26")
    assert score == pytest.approx(abs(3.1 - year))
