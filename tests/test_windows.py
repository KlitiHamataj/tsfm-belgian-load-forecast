import numpy as np
import pandas as pd
import pytest

from tsbench.windows import build_task, scored_mask, window_specs


def _spec(protocol, date):
    return next(s for s in window_specs(protocol) if str(s.scored_date.date()) == date)


def test_365_windows(protocol):
    specs = window_specs(protocol)
    assert len(specs) == 365
    assert specs[0].origin == pd.Timestamp("2024-12-31 17:00", tz="UTC")  # 18:00 CET


@pytest.mark.parametrize(
    "date,scored,horizon", [("2025-06-10", 96, 120), ("2025-03-30", 92, 116), ("2025-10-26", 100, 124)]
)
def test_horizon_and_dst(protocol, df, date, scored, horizon):
    sp = _spec(protocol, date)
    assert (sp.scored_steps, sp.horizon) == (scored, horizon)
    mask = scored_mask(sp, protocol)
    assert mask.sum() == scored and (~mask).sum() == protocol.bridge_steps
    assert not mask.iloc[: protocol.bridge_steps].any()  # bridge never scored
    t = build_task(df, sp, protocol, 96 * 7, ["day_of_week"], [0.5])
    local = t.horizon_index[mask.to_numpy()].tz_convert(protocol.timezone)
    assert local[0].strftime("%H:%M") == "00:00" and local[-1].strftime("%H:%M") == "23:45"
    assert (local.normalize().tz_localize(None) == sp.scored_date).all()


def test_context_never_reaches_origin(protocol, df):
    for sp in window_specs(protocol)[::30] + window_specs(protocol)[-1:]:
        t = build_task(df, sp, protocol, 96 * 15, [], [0.5])
        assert t.context.index.max() < sp.origin
        assert t.context.index.max().tz_convert(protocol.timezone).strftime("%H:%M") == "17:45"
        assert t.horizon_index[0] == sp.origin


def test_covariates_align_with_targets(protocol, df):
    sp = _spec(protocol, "2025-05-01")  # holiday
    t = build_task(df, sp, protocol, 96 * 10, ["be_holidays", "hour_of_day"], [0.5])
    assert (t.future_covariates.index == t.horizon_index).all()
    assert (t.past_covariates.index == t.context.index).all()
    local = t.horizon_index.tz_convert(protocol.timezone)
    assert (t.future_covariates["hour_of_day"].to_numpy() == local.hour).all()
    assert t.future_covariates["be_holidays"].iloc[-1] == 1.0  # scored day is May 1
    assert t.future_covariates["be_holidays"].iloc[0] == 0.0  # bridge = Apr 30


def test_no_data_before_2024(protocol, df):
    assert df.index.tz_convert(protocol.timezone).min() >= pd.Timestamp("2024-01-01", tz=protocol.timezone)
    with pytest.raises(AssertionError):
        build_task(df, window_specs(protocol)[0], protocol, 96 * 400, [], [0.5])


def test_nan_temperature_is_detected(protocol, df):
    sp = window_specs(protocol)[0]
    # early 2025 with a context reaching into the NaN block must raise rather than silently fill
    bad = df.copy()
    bad.loc[bad.index[-5000], "be_temperature"] = np.nan
    with pytest.raises(AssertionError):
        build_task(bad, sp, protocol, 96 * 370, ["be_temperature"], [0.5])
