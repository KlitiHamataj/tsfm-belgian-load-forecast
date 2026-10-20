from datetime import date

import numpy as np

from tsbench.cards import load_all_cards
from tsbench.models.baselines import (
    DailyPersistenceAdapter,
    SeasonalMovingAverageAdapter,
    WeeklyPersistenceAdapter,
    day_type,
)
from tsbench.windows import build_task, window_specs


def _task(protocol, df, date, days):
    sp = next(s for s in window_specs(protocol) if str(s.scored_date.date()) == date)
    return sp, build_task(df, sp, protocol, days * 96, [], [0.5])


def test_weekly_persistence_uses_same_wall_clock(protocol, df):
    sp, t = _task(protocol, df, "2025-06-10", 8)
    a = WeeklyPersistenceAdapter(load_all_cards()["weekly_persistence"]); a.load({}, "cpu")
    f = a.predict([t])[0]
    exp = df["load"].reindex(t.horizon_index - np.timedelta64(7, "D")).to_numpy()
    np.testing.assert_allclose(f.quantiles[0.5], exp)


def test_seasonal_ma_uses_only_available_sources(protocol, df):
    sp, t = _task(protocol, df, "2025-06-10", 9)
    a = SeasonalMovingAverageAdapter(load_cards := load_all_cards()["seasonal_moving_average"]); a.load({"n_days": 7, "day_matching": "all"}, "cpu")
    f = a.predict([t])[0]
    # last step 23:45 on June 10: lag 1 (June 9 23:45) is after the 17:45 cut -> must use June 8..June 2
    h = t.horizon_index[-1]
    manual = np.mean([df["load"][h - np.timedelta64(k, "D")] for k in range(2, 9)])
    assert np.isclose(f.quantiles[0.5][-1], manual)
    assert load_cards.id == "seasonal_moving_average"


def test_day_type():
    assert day_type(date(2025, 6, 10)) == "workday"  # Tuesday
    assert day_type(date(2025, 6, 14)) == "saturday"
    assert day_type(date(2025, 6, 15)) == "sunday_holiday"
    assert day_type(date(2025, 5, 1)) == "sunday_holiday"  # Labour Day, Thursday


def test_daily_persistence_latest_observed_same_quarter_hour(protocol, df):
    sp, t = _task(protocol, df, "2025-06-10", 3)
    a = DailyPersistenceAdapter(load_all_cards()["daily_persistence"]); a.load({}, "cpu")
    f = a.predict([t])[0].quantiles[0.5]
    h = t.horizon_index
    local = h.tz_convert(protocol.timezone)
    for i in range(len(h)):
        # target on D at 18:00-23:45 -> 2 days back (D-1 after 17:45 unobserved);
        # target on D at 00:00-17:45, or a bridge step on D-1 (source D-2) -> 1 day back
        k = 2 if (local[i].day == 10 and local[i].hour >= 18) else 1
        assert f[i] == df["load"][h[i] - np.timedelta64(k, "D")]


def test_day_type_matching_uses_same_type_days(protocol, df):
    # Saturday 2025-06-14: the 1 most recent matching day is the previous Saturday (June 7)
    sp, t = _task(protocol, df, "2025-06-14", 9)
    a = SeasonalMovingAverageAdapter(load_all_cards()["seasonal_moving_average"])
    a.load({"n_days": 1, "day_matching": "day_type"}, "cpu")
    f = a.predict([t])[0].quantiles[0.5]
    h = t.horizon_index[-1]
    assert np.isclose(f[-1], df["load"][h - np.timedelta64(7, "D")])
