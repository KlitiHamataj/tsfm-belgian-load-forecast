"""Two 1200x1200 PNGs for social media, built from the committed dashboard export (no raw data needed).

    uv run bench export-dashboard          # only if the runs changed
    uv run python scripts/plot_linkedin.py

docs/img/linkedin_week.png : the dashboard's default week (rule-based "typical week": holiday-free Monday-Sunday whose
                             WAPE is closest to the full-year WAPE), actual vs timestral-123m (with its P10-P90 band)
                             vs Elia's operational forecast, week WAPE per series in the legend.
docs/img/linkedin_year.png : daily WAPE through 2025 for the same two forecasts, weekday holidays labelled.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsbench.config import ROOT

DASH = ROOT / "docs" / "dashboard"
OUT = ROOT / "docs" / "img"
MODEL, REF = "timestral-123m", "elia_reference"
ACTUAL_COLOR, MUTED, HOLIDAY, NEAR_BLACK = "#8c8c8c", "#555b66", "#b0485a", "#1a1a1a"
SHORT = {"New Year's Day": "New Year", "Easter Monday": "Easter Mon", "Labor Day": "Labour Day",
         "Ascension Day": "Ascension", "Pentecost Monday": "Whit Monday", "National Day": "National Day",
         "Assumption Day": "Assumption", "Armistice Day": "Armistice", "Christmas Day": "Christmas"}
FOOTNOTE = ("Zero-shot (no fine-tuning), CPU. One forecast per day at 18:00 Brussels for the next day, 15-min,\n"
            "every day of 2025. Calendar + temperature covariates; best context length on 2025 (slightly optimistic).\n"
            "† Weights released after 2025 and training data undisclosed: contamination is not ruled out.\n"
            "Data: Elia Open Data (ods001). Weather data by Open-Meteo.com (CC BY 4.0).")

plt.rcParams.update({"font.size": 20, "axes.titlesize": 26, "axes.labelsize": 20, "xtick.labelsize": 18,
                     "ytick.labelsize": 18, "legend.fontsize": 18, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888888", "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 1})


def color(series: dict) -> str:
    return NEAR_BLACK if series["color"].lower() == "#000000" else series["color"]


def wape(a: np.ndarray, f: np.ndarray) -> float:
    return float(np.abs(f - a).sum() / np.abs(a).sum() * 100)


def load():
    data = json.loads((DASH / "data.json").read_text())
    daily = json.loads((DASH / "daily.json").read_text())
    utc = pd.Timestamp(data["start_utc"]) + pd.to_timedelta(np.arange(data["n"]) * data["step_min"], "min")
    # Naive local wall-clock time for plotting (no DST transition inside the plotted week).
    local = utc.tz_convert(data["timezone"]).tz_localize(None)
    return data, daily, local


def frame(fig, title: str, subtitle: str) -> None:
    fig.text(0.05, 0.965, title, fontsize=30, fontweight="bold", va="top", linespacing=1.15)
    fig.text(0.05, 0.858, subtitle, fontsize=19, color=MUTED, va="top", linespacing=1.3)
    fig.text(0.05, 0.022, FOOTNOTE, fontsize=12.5, color=MUTED, va="bottom", linespacing=1.45)


def week_png(data, daily, local) -> None:
    week = (data["default_view"]["start"], data["default_view"]["end"])
    lo, hi = pd.Timestamp(week[0]), pd.Timestamp(week[1]) + pd.Timedelta(days=1)
    m = (local >= lo) & (local < hi)
    t, a = local[m], np.asarray(data["actual"])[m]
    s, r = data["series"][MODEL], data["series"][REF]
    p10, p50, p90 = (np.asarray(s[q])[m] for q in ("P10", "P50", "P90"))
    ref = np.asarray(r["P50"])[m]

    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_axes((0.11, 0.33, 0.85, 0.44))
    for d, name, wk in zip(daily["dates"], daily["holiday_name"], daily["is_holiday"]):
        if wk and week[0] <= d <= week[1]:  # none by construction for the typical week
            ax.axvspan(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1), color=HOLIDAY, alpha=0.10, lw=0)
            ax.text(pd.Timestamp(d) + pd.Timedelta(hours=12), 0.985, f"{SHORT.get(name, name)}\n(public holiday)",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=16, color=HOLIDAY)
    ax.fill_between(t, p10, p90, color=color(s), alpha=0.18, lw=0, label=f"{s['label']}: P10-P90 band")
    ax.plot(t, a, color=ACTUAL_COLOR, lw=4.5, label="Actual load", solid_capstyle="round")
    ax.plot(t, ref, color=color(r), lw=2, label=f"{r['label']}  ·  week WAPE {wape(a, ref):.2f}%")
    ax.plot(t, p50, color=color(s), lw=2.4, label=f"{s['label']}  ·  week WAPE {wape(a, p50):.2f}%")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(mdates.DayLocator())  # gridlines at midnight, labels centred on each day
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=12))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.tick_params(axis="x", which="minor", length=0, labelsize=17)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_ylabel("Belgian grid load (MW)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.12)
    h, lab = ax.get_legend_handles_labels()
    order = [1, 3, 0, 2]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left", bbox_to_anchor=(0.04, 0.255),
               frameon=False, ncol=1, handlelength=2.2, fontsize=17, labelspacing=0.45)
    span = f"{lo:%a %d} - {hi - pd.Timedelta(days=1):%a %d %b %Y}"
    frame(fig, "A typical week of 2025: Belgian day-ahead\nload, zero-shot forecast vs actual",
          f"{span}: the holiday-free week whose WAPE is closest\nto the full-year WAPE (chosen by rule).")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "linkedin_week.png", facecolor="white")
    plt.close(fig)


def year_png(data, daily) -> None:
    dates = pd.to_datetime(daily["dates"])
    full = {k: sum(daily["sum_abs_err"][k]) / sum(daily["sum_actual"]) * 100 for k in (MODEL, REF)}
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_axes((0.11, 0.30, 0.85, 0.48))
    for d, name, hol, we in zip(dates, daily["holiday_name"], daily["is_holiday"], daily["is_weekend"]):
        if hol and not we:
            ax.axvline(d, color=HOLIDAY, lw=1.2, alpha=0.5, zorder=0)
            first = d == dates[0]  # 1 Jan sits on the left edge: label inside the axes
            ax.text(d, 0.99, SHORT.get(name, name), transform=ax.get_xaxis_transform(), rotation=90,
                    ha="left" if first else "right", va="top", fontsize=13, color=HOLIDAY)
    for k, lw in ((REF, 1.6), (MODEL, 1.8)):
        sr = data["series"][k]
        ax.plot(dates, daily["wape"][k], color=color(sr), lw=lw, label=f"{sr['label']}  ·  2025 WAPE {full[k]:.2f}%")
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for lab in ax.get_xticklabels():
        lab.set_fontsize(15)
    peak = max(max(daily["wape"][k]) for k in (MODEL, REF))
    ax.set_ylim(0, peak * 1.38)  # headroom so the holiday labels sit above the data
    ax.set_ylabel("Daily WAPE (%) - lower is better")
    fig.legend(loc="upper left", bbox_to_anchor=(0.04, 0.225), frameon=False, handlelength=2.2, fontsize=17)
    frame(fig, "Day by day through 2025: where zero-shot\nforecasting struggles",
          "Daily error of each day-ahead forecast (15-min, next local day);\nred lines mark weekday public holidays.")
    fig.savefig(OUT / "linkedin_year.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    data, daily, local = load()
    week_png(data, daily, local)
    year_png(data, daily)
    print(f"wrote {OUT.relative_to(ROOT)}/linkedin_week.png and linkedin_year.png")


if __name__ == "__main__":
    main()
