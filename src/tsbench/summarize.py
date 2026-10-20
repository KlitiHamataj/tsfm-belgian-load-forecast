"""Aggregate runs of one experiment: results_long/wide, summary.md, figures."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tsbench import metrics as M
from tsbench.cards import ModelCard
from tsbench.config import ROOT, Experiment, Protocol
from tsbench.data.prepare import data_hash
from tsbench.leakage import load_gates
from tsbench.registry import expand_models
from tsbench.runner import Cell, expand_cells, make_run_id, resolve_params
from tsbench.windows import window_specs

log = logging.getLogger(__name__)
ARROW = "↓"
ID_COLS = ["model", "covariates", "ctx (days)"]
README_EXPERIMENT = "leaderboard"  # only the main experiment writes the README results block
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


def _fmt_time(s: float) -> str:
    return f"{s:.1f}" if s < 1 else f"{s:.0f}"


def _load_runs(exp, protocol, cards, ids):
    n_expected = len(window_specs(protocol))
    dh = data_hash(protocol)
    runs, skipped = [], []
    for c in expand_cells(exp, protocol, cards, ids):
        if c.skip_reason:
            skipped.append((c.label, c.skip_reason))
            continue
        f = c.folder
        try:
            meta = json.loads((f / "run_meta.json").read_text())
            mets = json.loads((f / "metrics.json").read_text())
            win = pd.read_csv(f / "windows.csv")
        except FileNotFoundError:
            skipped.append((c.label, "no outputs (not run or failed)"))
            continue
        card = cards[c.model_id]
        rid = make_run_id(card, c, resolve_params(card, exp), exp, protocol, dh)
        if meta["run_id"] != rid:
            skipped.append((c.label, "stale run_id (config/adapter/data changed): re-run"))
            continue
        if mets["n_windows"] != n_expected or len(win) != n_expected:
            raise RuntimeError(f"{c.label}: incomplete windows ({len(win)}/{n_expected}); refusing to aggregate")
        runs.append({"cell": c, "meta": meta, "metrics": mets, "windows": win, "card": card})
    return runs, skipped


def _ids(r) -> dict:
    """Run identity as separate columns (no '|' inside labels)."""
    c: Cell = r["cell"]
    base = c.covariate_set == "default"
    return {"model": c.model_id, "covariates": "-" if base else c.covariate_set,
            "ctx (days)": "-" if base else c.context_days}


def _label(r) -> str:
    """Short label for figures and the 'vs' column."""
    i = _ids(r)
    return i["model"] if i["covariates"] == "-" else f"{i['model']} / {i['covariates']} / {i['ctx (days)']}d"


def _params_m(card: ModelCard) -> float | None:
    """Card `params` ("5.21M", "~123M", "0.2B") -> millions."""
    s = card.params.strip().lstrip("~")
    try:
        return float(s[:-1]) * (1000 if s.endswith("B") else 1) if s[-1] in "MB" else None
    except ValueError:
        return None


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if len(df) else "_none_"


def _gates_section(protocol: Protocol, cards: dict[str, ModelCard], ids: list[str], runs: list) -> list[str]:
    md = ["## Acceptance gates\n"]
    ref = next((r for r in runs if r["cell"].model_id == protocol.reference_model), None)
    target, tol = protocol.metrics.elia_reference_wape_pct, protocol.metrics.elia_reference_tolerance
    if ref:
        w = ref["metrics"]["wape_pct"]
        verdict = "PASS" if abs(w - target) <= tol else "FAIL"
        md.append(f"- **{protocol.reference_model} pooled WAPE on 2025**: {w:.4f}% (target {target}% ± {tol}) → **{verdict}**")
    else:
        md.append(f"- **{protocol.reference_model} WAPE gate**: not evaluated ({protocol.reference_model} not in this experiment)")
    n_exp = len(window_specs(protocol))
    md.append(f"- **Complete windows**: every aggregated run has {n_exp}/{n_exp} windows (aggregation refuses otherwise)\n")

    rows = []
    for mid in ids:
        g, card = load_gates(mid), cards[mid]

        def cell(key: str) -> tuple[str, str]:
            if key not in g:
                return "not run", ""
            rec = g[key]
            stale = rec["revision"] != card.revision or rec["adapter_version"] != card.adapter_version
            res = ("PASS" if rec["passed"] else "FAIL") + (" (stale: re-run)" if stale else "")
            setting = f"{rec['covariate_set']}, ctx {rec['context_days'] or '-'}"
            return res, setting

        lk, lk_set = cell("leakage")
        dt, dt_set = cell("determinism")
        extra = ""
        if "determinism" in g and not g["determinism"]["passed"]:
            extra = f"max |Δ| {g['determinism']['max_abs_diff_mw']:.3g} MW"
        rows.append({"model": mid, "leakage (noise after 17:45)": lk, "leakage setting": lk_set,
                     "determinism (2 runs, bit-identical)": dt, "determinism setting": dt_set, "note": extra,
                     "recorded (UTC)": max((g[k]["recorded_utc"][:16] for k in g), default="")})
    md += ["Leakage dates: 2025-03-15 (normal), 2025-03-30 and 2025-10-26 (DST), 2025-05-01 and 2025-12-25 (holidays). "
           "Determinism: 8 windows run twice, all stored quantiles compared. "
           "Re-run with `bench gates --config <experiment>`; stale = revision or adapter_version changed since recording.\n",
           _md(pd.DataFrame(rows)), ""]
    return md


def summarize(exp: Experiment, protocol: Protocol, cards: dict[str, ModelCard]) -> None:
    ids = expand_models(exp, cards)
    runs, skipped = _load_runs(exp, protocol, cards, ids)
    if not runs:
        raise RuntimeError("no complete runs to aggregate for this config")
    out = ROOT / "outputs" / "experiments" / exp.name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    nb, seed = protocol.metrics.n_boot, protocol.metrics.bootstrap_seed

    wide = []
    for r in runs:
        m, mt, c = r["metrics"], r["meta"], r["cell"]
        wide.append(
            {
                **_ids(r), "family": r["card"].family,
                "context_days_configured": mt["context_days_configured"], "context_days_effective": mt["context_days_effective"],
                "context_steps_effective": mt["context_steps_effective_max"], **m,
                "load_seconds": mt["timings"]["load_seconds"], "inference_seconds": mt["timings"]["inference_seconds"],
                "per_window_seconds": mt["timings"]["per_window_seconds"], "device": mt["device"],
                "num_threads": mt["num_threads"], "batch_size": mt["batch_size"], "run_id": mt["run_id"],
                "covariate_set": c.covariate_set,
            }
        )
    W = pd.DataFrame(wide)
    W.to_csv(out / "results_wide.csv", index=False)
    W.melt(id_vars=[*ID_COLS, "covariate_set", "context_days_configured", "run_id"], var_name="metric") \
        .dropna(subset=["value"]).to_csv(out / "results_long.csv", index=False)

    ds = protocol.dataset
    md = [
        f"# {exp.name}\n",
        "| data | SHA-256 |\n|:--|:--|\n"
        f"| processed dataset `{ds.processed}` (input of every run) | `{data_hash(protocol)}` |\n"
        f"| raw load `{ds.raw_load}` (Elia ods001) | `{ds.raw_load_sha256}` |\n"
        f"| raw weather `{ds.raw_weather}` (Open-Meteo) | `{ds.raw_weather_sha256}` |\n",
        f"Windows per run: {int(W.n_windows.iloc[0])} (every day of 2025)  |  CPU threads: "
        f"{sorted(int(x) for x in W.num_threads.unique())}  |  bootstrap: {nb} resamples of days, seed {seed}\n",
        "Caveat: Open-Meteo has no issue timestamps; availability of the temperature covariate at 18:00 the day before "
        "cannot be proven.\n",
    ]
    md += _gates_section(protocol, cards, ids, runs)

    # Leaderboard
    L = W.sort_values("wape_pct")[[*ID_COLS, "context_days_effective", "wape_pct", "wape_ci95_lo", "wape_ci95_hi",
                                   "median_window_wape_pct", "mae_mw", "rmse_mw", "smape_pct", "coverage_p10_p90_pct",
                                   "interval_width_mw", "pinball_mean_mw", "pinball_p10_p50_p90_mw", "crps_approx_mw"]].copy()
    L.columns = [*ID_COLS, "eff. ctx (days)", f"WAPE (pooled) {ARROW} %", "CI95 lo %", "CI95 hi %",
                 f"median-window WAPE {ARROW} %", f"MAE {ARROW}", f"RMSE {ARROW}", f"sMAPE {ARROW} %",
                 "P10-P90 coverage % (target 80)", f"width {ARROW}", f"pinball mean(0.1..0.9) {ARROW}",
                 f"pinball P10/50/90 {ARROW}", f"CRPS≈ {ARROW}"]
    md += ["## Leaderboard\nMW for MAE, RMSE, width, pinball, CRPS≈. Point metrics use P50, pooled over all scored points. "
           "eff. ctx = context actually fed to the model (baselines: days their definition needs). "
           "CRPS≈ = 2 × mean pinball over the common quantile set 0.1..0.9 (only where all 9 levels exist; the Elia reference "
           "has only 0.1/0.5/0.9, so compare it via the P10/50/90 pinball column).\n", _md(L.round(2)), ""]

    # Size curve
    size_ctx = 60
    sz = W[(W["ctx (days)"] == size_ctx) & (W.family.isin(["timestral", "chronos"]))].copy()
    if len(sz):
        sz["params_m"] = [_params_m(cards[m]) for m in sz.model]
        ref_rows = {cs: g for cs, g in sz[sz.model == protocol.primary_comparison_model].groupby("covariates")}
        rows = []
        for r in sz.sort_values(["covariates", "params_m"]).itertuples():
            ref = ref_rows.get(r.covariates)
            d = (r.wape_pct - ref.wape_pct.iloc[0]) if ref is not None and r.model != protocol.primary_comparison_model else None
            rows.append({"model": r.model, "params (M)": _params_m(cards[r.model]), "covariates": r.covariates,
                         f"WAPE {ARROW} %": round(r.wape_pct, 2),
                         f"Δ WAPE vs {protocol.primary_comparison_model} (pp)": "ref" if r.model == protocol.primary_comparison_model
                         else ("-" if d is None else round(d, 2)),
                         f"CRPS≈ {ARROW} MW": round(r.crps_approx_mw, 1), f"per window {ARROW} s": _fmt_time(r.per_window_seconds),
                         f"total inference {ARROW} s": _fmt_time(r.inference_seconds)})
        missing = [m for m in cards if cards[m].family == "timestral" and m in ids and not (sz.model == m).any()]
        md += [f"## Size curve (Timestral sizes at {size_ctx}d context, {protocol.primary_comparison_model} as reference)\n"
               "CPU latency on this machine (threads in the header). Mistral claims Timestral-28M beats Chronos-2: compare the "
               "Δ column at matching covariates. Δ < 0 = better than the reference.\n",
               _md(pd.DataFrame(rows)),
               "" if not missing else f"\nNot yet run at {size_ctx}d: {', '.join(missing)}.",
               "\n![Accuracy vs latency](figures/accuracy_latency_bubble.png)\n", ""]

    # Context sweep
    sweep = W[W.covariates != "-"].pivot_table(index=["model", "covariates"], columns="context_days_configured",
                                               values="wape_pct").round(2)
    sweep.columns = [f"{int(c)}d" for c in sweep.columns]
    md += [f"## Context sweep (pooled WAPE {ARROW} %, by configured context)\n"
           "Where effective context differs from configured (e.g. a model maximum), the leaderboard's eff. ctx column shows it.\n",
           sweep.reset_index().to_markdown(index=False) if len(sweep) else "_none_", ""]

    # Covariate effect
    if (W.covariates == "none").any() and (W.covariates == "full").any():
        p = W[W.covariates.isin(["none", "full"])].pivot_table(index=["model", "ctx (days)"], columns="covariates",
                                                               values="wape_pct")
        if {"none", "full"} <= set(p.columns):
            p["full − none (pp)"] = p["full"] - p["none"]
            md += [f"## Covariate effect (WAPE {ARROW} %; negative last column = covariates help)\n",
                   p.round(2).reset_index().to_markdown(index=False), ""]

    # Slices
    S = pd.DataFrame([{**_ids(r), **M.slice_wape(r["windows"])} for r in runs]).sort_values(["model", "covariates"])
    md += [f"## Slices (pooled WAPE {ARROW} %)\nweekday = Mon–Fri non-holiday; weekend = Sat/Sun; "
           "holiday_weekday = public holiday on Mon–Fri; non_holiday = all non-holiday days.\n", _md(S.round(2)), ""]

    # Paired comparisons
    def paired(a, b):
        d = M.paired_delta(a["windows"], b["windows"], nb, seed)
        return {**_ids(a), "vs": _label(b), f"ΔWAPE {ARROW} (pp)": round(d["delta_wape"], 2),
                "CI95": f"[{d['delta_ci_lo']:.2f}, {d['delta_ci_hi']:.2f}]", "days better": f"{d['days_a_better']}/{d['n_days']}",
                "sign-test p": f"{d['sign_test_p']:.3g}"}

    prow = []
    ref = next((r for r in runs if r["cell"].model_id == protocol.reference_model), None)
    if ref:
        prow += [paired(r, ref) for r in runs if r is not ref]
    cname = protocol.primary_comparison_model
    chron = [r for r in runs if r["cell"].model_id == cname]
    if chron:
        best = min(chron, key=lambda r: r["metrics"]["wape_pct"])
        for r in runs:
            if r["cell"].model_id == cname:
                continue
            same = next((x for x in chron if x["cell"].covariate_set == r["cell"].covariate_set
                         and x["cell"].context_days == r["cell"].context_days), None)
            prow.append(paired(r, same or best))
    md += ["## Paired comparisons (negative ΔWAPE = row is better; bootstrap over days; sign test over days)\n"
           f"Versus {protocol.reference_model} for every run; versus {cname} at the same covariates/context where that "
           f"run exists, else its best run" + ("" if chron else f" ({cname} not in this experiment)") + ".\n",
           _md(pd.DataFrame(prow)), ""]

    # Latency
    lat = W[[*ID_COLS, "device", "num_threads", "batch_size", "load_seconds", "inference_seconds", "per_window_seconds"]].copy()
    for c in ["load_seconds", "inference_seconds", "per_window_seconds"]:
        lat[c] = lat[c].map(_fmt_time)
    lat.columns = [*ID_COLS, "device", "threads", "batch", f"load {ARROW} s", f"total inference {ARROW} s",
                   f"per window {ARROW} s"]
    md += ["## Latency\n", _md(lat), ""]

    if skipped:
        md += ["## Not aggregated (skipped / missing)\n", _md(pd.DataFrame(skipped, columns=["cell", "reason"])), ""]
    (out / "summary.md").write_text("\n".join(md))
    _figures(W, runs, cards, out)
    if exp.name == README_EXPERIMENT:
        _write_readme(runs, protocol, cards, nb, seed, skipped)
    log.info("wrote %s", out)


def _col(cards, mid):
    return cards[mid].color


def _figures(W, runs, cards, out):
    fig_dir = out / "figures"
    sw = W[W.covariates != "-"]
    if len(sw):
        n = sw.covariates.nunique()
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, squeeze=False)
        for ax, (cs, g) in zip(axes[0], sw.groupby("covariates")):
            for mid, gm in g.groupby("model"):
                gm = gm.sort_values("context_days_effective")
                ax.plot(gm.context_days_effective, gm.wape_pct, "o-", color=_col(cards, mid), label=mid)
            for r in W[W.covariates == "-"].itertuples():
                if r.model in ("elia_reference", "weekly_persistence"):
                    ax.axhline(r.wape_pct, ls="--", color=_col(cards, r.model), lw=1, label=r.model)
            ax.set(title=f"covariates: {cs}", xlabel="effective context (days)", ylabel=f"pooled WAPE {ARROW} (%)")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "wape_vs_context.png", dpi=130)
        plt.close(fig)

    best = {}
    for r in runs:
        k = r["cell"].model_id
        if k not in best or r["metrics"]["wape_pct"] < best[k]["metrics"]["wape_pct"]:
            best[k] = r
    bl = list(best.values())

    fig, ax = plt.subplots(figsize=(9, 4))
    for r in bl:
        w = r["windows"]
        mo = pd.to_datetime(w.scored_date).dt.month
        s = w.groupby(mo).apply(lambda x: x.sum_abs_error.sum() / x.sum_actual.sum() * 100)
        ax.plot(s.index, s.values, "o-", color=_col(cards, r["cell"].model_id), label=_label(r), lw=1)
    ax.set(xlabel="month", ylabel=f"pooled WAPE {ARROW} (%)", title="Monthly WAPE (best run per model)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "monthly_wape.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, r in enumerate(bl):
        w = r["windows"]
        h = w[w.is_holiday & (pd.to_datetime(w.scored_date).dt.dayofweek < 5)]
        n = w[~w.is_holiday]
        ax.bar(i - 0.2, M.pooled_wape_from_windows(h), 0.4, color=_col(cards, r["cell"].model_id), alpha=0.5)
        ax.bar(i + 0.2, M.pooled_wape_from_windows(n), 0.4, color=_col(cards, r["cell"].model_id))
    ax.set_xticks(range(len(bl)), [r["cell"].model_id for r in bl], rotation=60, ha="right", fontsize=7)
    ax.set(ylabel=f"pooled WAPE {ARROW} (%)", title="Weekday holidays (light) vs non-holidays (solid)")
    fig.tight_layout()
    fig.savefig(fig_dir / "holiday_vs_nonholiday.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    markers = {"none": "o", "full": "s"}
    for r in runs:
        mid = r["cell"].model_id
        if cards[mid].kind == "baseline" or r["cell"].context_days != 60:
            continue
        cs = r["cell"].covariate_set
        size = _params_m(cards[mid]) or 50
        x, y = r["meta"]["timings"]["per_window_seconds"], r["metrics"]["wape_pct"]
        ax.scatter(x, y, s=20 + size * 3, color=_col(cards, mid), alpha=0.7, marker=markers.get(cs, "^"), edgecolor="k", lw=0.5)
        ax.annotate(f"{mid} ({cs})", (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ref = next((r for r in runs if r["cell"].model_id == "elia_reference"), None)
    if ref:
        ax.axhline(ref["metrics"]["wape_pct"], ls="--", color=_col(cards, "elia_reference"), lw=1, label="elia_reference")
        ax.legend(fontsize=7)
    ax.set(xscale="log", xlabel=f"inference seconds per window {ARROW} (CPU, log)", ylabel=f"pooled WAPE {ARROW} (%)",
           title="Accuracy vs latency at 60d (circle = none, square = full; area ∝ params)")
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_latency_bubble.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([r["windows"].window_wape for r in bl], showfliers=False)
    ax.set_xticks(range(1, len(bl) + 1), [_label(r) for r in bl], rotation=60, ha="right", fontsize=6)
    ax.set(ylabel=f"per-window WAPE {ARROW} (%)", title="Per-window WAPE (best run per model)")
    fig.tight_layout()
    fig.savefig(fig_dir / "window_wape_boxplot.png", dpi=130)
    plt.close(fig)


def _write_readme(runs, protocol: Protocol, cards: dict[str, ModelCard], nb: int, seed: int, skipped: list) -> None:
    """Replace the README block between the results markers with the best run per model."""
    readme = ROOT / "README.md"
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        log.warning("README has no results markers; skipping")
        return
    best: dict[str, dict] = {}
    for r in runs:
        k = r["cell"].model_id
        if k not in best or r["metrics"]["wape_pct"] < best[k]["metrics"]["wape_pct"]:
            best[k] = r
    ref = best.get(protocol.reference_model)
    rows = []
    for r in sorted(best.values(), key=lambda r: r["metrics"]["wape_pct"]):
        m = r["metrics"]
        row = {**_ids(r), f"WAPE {ARROW} %": f"{m['wape_pct']:.2f}", "CI95 %": f"[{m['wape_ci95_lo']:.2f}, {m['wape_ci95_hi']:.2f}]",
               f"median-window WAPE {ARROW} %": f"{m['median_window_wape_pct']:.2f}"}
        if ref is not None and r is not ref:
            d = M.paired_delta(r["windows"], ref["windows"], nb, seed)
            row[f"Δ vs {protocol.reference_model} (pp) [CI95]"] = f"{d['delta_wape']:+.2f} [{d['delta_ci_lo']:+.2f}, {d['delta_ci_hi']:+.2f}]"
            row["days better"] = f"{d['days_a_better']}/{d['n_days']}"
        else:
            row[f"Δ vs {protocol.reference_model} (pp) [CI95]"] = "ref"
            row["days better"] = "-"
        crps = m.get("crps_approx_mw")
        row[f"CRPS≈ {ARROW} MW"] = f"{crps:.0f}" if crps == crps and crps is not None else "-"
        row[f"per window {ARROW} s"] = _fmt_time(r["meta"]["timings"]["per_window_seconds"])
        rows.append(row)
    gate = ""
    if ref is not None:
        w, tgt, tol = ref["metrics"]["wape_pct"], protocol.metrics.elia_reference_wape_pct, protocol.metrics.elia_reference_tolerance
        gate = f"Gate: {protocol.reference_model} WAPE {w:.3f}% (target {tgt}% ± {tol}) → {'PASS' if abs(w - tgt) <= tol else 'FAIL'}. "
    failing = [mid for mid in best if load_gates(mid) and not all(load_gates(mid)[k]["passed"] for k in load_gates(mid))]
    gate += "Leakage and determinism: " + ("all recorded gates pass." if not failing else f"FAILING for {', '.join(failing)}.")
    block = "\n".join([
        README_START,
        f"_Auto-filled by `bench summarize` ({pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC). Best configuration per model on "
        f"2025 (365 daily windows), pooled over all scored points; Δ is paired over the same days (negative = better than "
        f"{protocol.reference_model}). Context and covariates were selected on 2025 itself, so the best-per-model WAPE is "
        "optimistic; see the full context sweep in the experiment summary._",
        "",
        _md(pd.DataFrame(rows)),
        "",
        gate + (f" {len(skipped)} configured cell(s) not aggregated." if skipped else ""),
        README_END,
    ])
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + block + text[end:])
    log.info("updated README results")
