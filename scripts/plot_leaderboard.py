"""Plot pooled WAPE (best configuration per model) from the leaderboard results.

Reads results/leaderboard/results_wide.csv (written by `bench summarize`) and saves
docs/img/wape_by_model.png. Best configuration and the † contamination flag follow the same rules
as the README results table. Bars use each model card's `color`; † models are hatched.

    uv run python scripts/plot_leaderboard.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tsbench.cards import load_all_cards
from tsbench.config import ROOT, load_protocol

RESULTS = ROOT / "results" / "leaderboard" / "results_wide.csv"
OUT = ROOT / "docs" / "img" / "wape_by_model.png"
# Omitted from the chart only (still in the README table): its ~12% WAPE would squash the axis.
EXCLUDE = {"last_value"}


def main() -> None:
    protocol = load_protocol()
    cards = load_all_cards()
    df = pd.read_csv(RESULTS)
    dropped = df[df.model.isin(EXCLUDE)].groupby("model")["wape_pct"].min()
    df = df[~df.model.isin(EXCLUDE)]
    best = df.loc[df.groupby("model")["wape_pct"].idxmin()].sort_values("wape_pct", ascending=False)

    def flagged(model: str) -> bool:
        c = cards.get(model)
        return bool(
            c and c.weights_date and c.weights_date > protocol.scored_end
            and c.training_data_disclosed is False
        )

    def label(r: pd.Series) -> str:
        name = f"{r.model} †" if flagged(r.model) else r.model
        if r.model == protocol.reference_model or r.family == "baseline":
            return name
        return f"{name}  ({r.covariates}, {r['ctx (days)']}d)"

    ref_wape = float(best.loc[best.model == protocol.reference_model, "wape_pct"].iloc[0])
    colors = [cards[m].color for m in best.model]
    hatches = ["////" if flagged(m) else "" for m in best.model]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = range(len(best))
    err = [best.wape_pct - best.wape_ci95_lo, best.wape_ci95_hi - best.wape_pct]
    plt.rcParams["hatch.linewidth"] = 1.2
    bars = ax.barh(y, best.wape_pct, xerr=err, color=colors, height=0.65, edgecolor="white", linewidth=0,
                   error_kw={"ecolor": "black", "elinewidth": 1, "capsize": 3})
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_yticks(list(y), [label(r) for _, r in best.iterrows()], fontsize=9)
    for yi, w, hi in zip(y, best.wape_pct, best.wape_ci95_hi):
        ax.text(hi + 0.05, yi, f"{w:.2f}", va="center", ha="left", fontsize=8, color="black")
    ax.axvline(ref_wape, color="black", linestyle="--", linewidth=1)
    ax.text(ref_wape, len(best) - 0.4, f" {protocol.reference_model} {ref_wape:.2f}%",
            va="bottom", ha="left", fontsize=8)
    ax.set_xlabel("Pooled WAPE on 2025 (%), lower is better; whiskers = 95% CI over days")
    ax.set_ylim(-0.6, len(best) - 0.1)
    ax.set_title("Best configuration per model, 365 daily windows", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    note = "† hatched: weights released after 2025, training data undisclosed: contamination not ruled out."
    if len(dropped):
        omitted = ", ".join(f"{m} ({w:.1f}%)" for m, w in dropped.items())
        note += f" Not shown: {omitted}; see the table."
    fig.text(0.01, 0.01, note, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
