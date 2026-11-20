"""README Holidays section: holiday list (names, weekend flag) and per-day WAPE from points."""

import pandas as pd

from tsbench.config import load_protocol
from tsbench.summarize import daily_wape, holiday_list


def test_holiday_list_2025():
    h = holiday_list(load_protocol())
    assert len(h) == 12
    assert int((~h.is_weekend).sum()) == 9
    names = dict(zip(h.date.astype(str), h.name))
    assert names["2025-05-01"] == "Labour Day"
    assert names["2025-06-08"] == "Whit Sunday" and names["2025-06-09"] == "Whit Monday"
    assert set(h[h.is_weekend].date.astype(str)) == {"2025-04-20", "2025-06-08", "2025-11-01"}


def test_daily_wape_uses_local_scored_day_only():
    ts = pd.date_range("2025-03-29 23:00", periods=8, freq="h", tz="UTC")
    p = pd.DataFrame({"timestamp": ts, "is_scored": [False] + [True] * 7,
                      "actual": [1000.0] * 8, "P50": [0.0] + [990.0] * 6 + [1100.0]})
    d = daily_wape(p, "Europe/Brussels")
    assert list(d.index.astype(str)) == ["2025-03-30"]  # 23:00 UTC = 00:00 CET on the 30th, but unscored
    assert abs(d.iloc[0] - (6 * 10 + 100) / 7000 * 100) < 1e-12
