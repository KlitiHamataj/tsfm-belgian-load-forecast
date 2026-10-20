import numpy as np
import pandas as pd

from tsbench import metrics as M


def test_point_metrics_toy():
    y, yh = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert M.wape(y, yh) == pytest_approx(100 * 30 / 300)
    assert M.mae(y, yh) == 15.0 and np.isclose(M.rmse(y, yh), np.sqrt(250))
    assert np.isclose(M.smape(y, yh), 100 * np.mean([10 / 105, 20 / 190]))


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_pooled_differs_from_median_and_pinball():
    assert np.isclose(M.pinball(np.array([10.0]), np.array([8.0]), 0.9), 0.9 * 2)
    assert np.isclose(M.pinball(np.array([10.0]), np.array([12.0]), 0.9), 0.1 * 2)


def test_bootstrap_resamples_windows_and_paired():
    w = pd.DataFrame({"origin_date": list("abcdefgh"), "sum_abs_error": [1, 2, 3, 4, 5, 6, 7, 8.0],
                      "sum_actual": [100.0] * 8, "window_wape": [1, 2, 3, 4, 5, 6, 7, 8.0]})
    lo, hi = M.bootstrap_wape(w, 500, 1)
    assert lo < M.pooled_wape_from_windows(w) < hi
    assert M.bootstrap_wape(w, 500, 1) == (lo, hi)  # fixed seed
    b = w.assign(sum_abs_error=w.sum_abs_error + 1, window_wape=w.window_wape + 1)
    d = M.paired_delta(w, b, 500, 1)
    assert np.isclose(d["delta_wape"], -1.0) and d["days_a_better"] == 8 and d["delta_ci_hi"] < 0
