"""Metrics. Point metrics use P50 and are POOLED over all scored points unless labelled otherwise."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum() * 100)


def mae(y, yhat) -> float:
    return float(np.abs(y - yhat).mean())


def rmse(y, yhat) -> float:
    return float(np.sqrt(((y - yhat) ** 2).mean()))


def smape(y, yhat) -> float:
    return float(100 * np.mean(np.abs(y - yhat) / ((np.abs(y) + np.abs(yhat)) / 2)))


def pinball(y: np.ndarray, qhat: np.ndarray, tau: float) -> float:
    d = y - qhat
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def window_table(points: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-window sums of scored points. `meta` has origin_date, is_holiday, is_weekend per window."""
    s = points[points["is_scored"]].copy()
    s["abs_err"] = (s["actual"] - s["P50"]).abs()
    g = s.groupby("origin_date").agg(
        n_points=("actual", "size"), sum_abs_error=("abs_err", "sum"), sum_actual=("actual", lambda x: x.abs().sum())
    )
    g["window_wape"] = g["sum_abs_error"] / g["sum_actual"] * 100
    return meta.join(g, how="inner").reset_index()


def pooled_wape_from_windows(w: pd.DataFrame) -> float:
    return float(w["sum_abs_error"].sum() / w["sum_actual"].sum() * 100)


def bootstrap_wape(w: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI of pooled WAPE, resampling WINDOWS (days)."""
    rng = np.random.default_rng(seed)
    e, a = w["sum_abs_error"].to_numpy(), w["sum_actual"].to_numpy()
    idx = rng.integers(0, len(w), size=(n_boot, len(w)))
    boots = e[idx].sum(1) / a[idx].sum(1) * 100
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_delta(w_a: pd.DataFrame, w_b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """WAPE(a) - WAPE(b) on the same windows (negative = a better), bootstrap CI + sign test on days."""
    m = w_a.merge(w_b, on="origin_date", suffixes=("_a", "_b"))
    assert len(m) == len(w_a) == len(w_b), "paired comparison needs identical windows"
    ea, eb = m["sum_abs_error_a"].to_numpy(), m["sum_abs_error_b"].to_numpy()
    act = m["sum_actual_a"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    d = (ea[idx].sum(1) - eb[idx].sum(1)) / act[idx].sum(1) * 100
    wins = int((m["window_wape_a"] < m["window_wape_b"]).sum())
    n_ties = int((m["window_wape_a"] == m["window_wape_b"]).sum())
    n = len(m) - n_ties
    return {
        "delta_wape": float((ea.sum() - eb.sum()) / act.sum() * 100),
        "delta_ci_lo": float(np.percentile(d, 2.5)),
        "delta_ci_hi": float(np.percentile(d, 97.5)),
        "days_a_better": wins,
        "n_days": len(m),
        "sign_test_p": float(binomtest(wins, n, 0.5).pvalue) if n else float("nan"),
    }


def slice_wape(w: pd.DataFrame) -> dict[str, float]:
    """Pooled WAPE within slices. weekday = Mon-Fri non-holiday; weekend = Sat/Sun;
    holiday_weekday = public holiday on Mon-Fri; non_holiday = all non-holiday days."""
    dow = pd.to_datetime(w["origin_date"]).dt.dayofweek  # origin = day before scored day
    scored = pd.to_datetime(w["scored_date"])
    sdow = scored.dt.dayofweek
    masks = {
        "holiday_weekday": w["is_holiday"] & (sdow < 5),
        "non_holiday": ~w["is_holiday"],
        "weekday": (~w["is_holiday"]) & (sdow < 5),
        "weekend": w["is_weekend"],
    }
    for m in range(1, 13):
        masks[f"month_{m:02d}"] = scored.dt.month == m
    del dow
    return {k: pooled_wape_from_windows(w[v]) for k, v in masks.items() if v.any()}


def prob_metrics(points: pd.DataFrame, eval_q: list[float], interval: tuple[float, float]) -> dict:
    s = points[points["is_scored"]]
    y = s["actual"].to_numpy()
    out: dict = {}
    lo, hi = (f"P{round(q * 100):02d}" for q in interval)
    if lo in s and hi in s:
        l, h = s[lo].to_numpy(), s[hi].to_numpy()
        out["coverage_p10_p90_pct"] = float(((y >= l) & (y <= h)).mean() * 100)
        out["interval_width_mw"] = float((h - l).mean())
    avail = [q for q in eval_q if f"P{round(q * 100):02d}" in s and s[f"P{round(q * 100):02d}"].notna().all()]
    if avail:
        pb = {q: pinball(y, s[f"P{round(q * 100):02d}"].to_numpy(), q) for q in avail}
        out["pinball_mean_mw"] = float(np.mean(list(pb.values())))
        out["pinball_levels_used"] = len(avail)
        common3 = [q for q in (0.1, 0.5, 0.9) if q in pb]
        out["pinball_p10_p50_p90_mw"] = float(np.mean([pb[q] for q in common3])) if len(common3) == 3 else None
        # CRPS ~ 2 * mean_tau pinball_tau, only when the full eval set is available
        out["crps_approx_mw"] = float(2 * np.mean(list(pb.values()))) if len(avail) == len(eval_q) else None
    return out


def compute_metrics(points: pd.DataFrame, windows: pd.DataFrame, eval_q, interval, n_boot, seed) -> dict:
    s = points[points["is_scored"]]
    y, yh = s["actual"].to_numpy(), s["P50"].to_numpy()
    lo, hi = bootstrap_wape(windows, n_boot, seed)
    return {
        "wape_pct": wape(y, yh),
        "wape_ci95_lo": lo,
        "wape_ci95_hi": hi,
        "mae_mw": mae(y, yh),
        "rmse_mw": rmse(y, yh),
        "smape_pct": smape(y, yh),
        "median_window_wape_pct": float(windows["window_wape"].median()),
        "n_windows": len(windows),
        "n_scored_points": len(s),
        **prob_metrics(points, eval_q, interval),
    }


def crps_window_table(points: pd.DataFrame, eval_q: list[float]) -> pd.DataFrame:
    """Per window: sum over scored points of CRPS≈ (= 2 x mean pinball over eval_q) and point count."""
    s = points[points["is_scored"]]
    y = s["actual"].to_numpy()
    cols = [f"P{round(q * 100):02d}" for q in eval_q]
    missing = [c for c in cols if c not in s]
    if missing:
        raise ValueError(f"CRPS≈ needs all eval quantiles; missing {missing}")
    pb = np.zeros(len(s))
    for q, c in zip(eval_q, cols):
        d = y - s[c].to_numpy()
        pb += np.maximum(q * d, (q - 1) * d)
    crps = 2 * pb / len(eval_q)
    return (pd.DataFrame({"origin_date": s["origin_date"].to_numpy(), "crps": crps})
            .groupby("origin_date").agg(crps_sum=("crps", "sum"), n_points=("crps", "size")).reset_index())


def paired_crps_delta(c_a: pd.DataFrame, c_b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """CRPS≈(a) - CRPS≈(b) in MW, pooled over points, bootstrap over days (negative = a better)."""
    m = c_a.merge(c_b, on="origin_date", suffixes=("_a", "_b"))
    assert len(m) == len(c_a) == len(c_b), "paired comparison needs identical windows"
    sa, sb, n = m["crps_sum_a"].to_numpy(), m["crps_sum_b"].to_numpy(), m["n_points_a"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    win_a, win_b = sa / n, sb / n
    wins = int((win_a < win_b).sum())
    ties = int((win_a == win_b).sum())
    return {
        "delta_crps_mw": float((sa.sum() - sb.sum()) / n.sum()),
        "delta_ci_lo": float(np.percentile(d, 2.5)),
        "delta_ci_hi": float(np.percentile(d, 97.5)),
        "days_a_better": wins,
        "n_days": int(len(m)),
        "sign_test_p": float(binomtest(wins, len(m) - ties, 0.5).pvalue) if len(m) > ties else float("nan"),
    }
