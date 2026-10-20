"""One-off diagnostic analyses. Results go to outputs/experiments/<name>/analyses/ and are appended to summary.md."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsbench import metrics as M
from tsbench.cards import load_all_cards
from tsbench.config import ROOT, Experiment, load_protocol
from tsbench.data.prepare import load_processed
from tsbench.runner import resolve_params, set_threads
from tsbench.windows import build_task, scored_mask, window_specs

log = logging.getLogger(__name__)
TIMING_DIR = ROOT / "outputs" / "timing"


def _analyses_dir(experiment: str) -> Path:
    d = ROOT / "outputs" / "experiments" / experiment / "analyses"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mirrored_pass_test(experiment: str, model_id: str, cov_set: str, ctx_days: int) -> dict:
    """Score Timestral's direct pass and its mirrored pass separately on all windows.

    direct   = _forecast_raw(context)                      (what symmetric_averaging=False returns)
    mirrored = -_forecast_raw(-context).flip(quantiles)    (the second pass symmetric_averaging adds)
    average  = 0.5 * (direct + mirrored)                   (what symmetric_averaging=True returns)
    clip_nonnegative is applied to each, as the library does after averaging.
    """
    from timestral.forecaster import nonnegative_context

    protocol = load_protocol()
    card = load_all_cards()[model_id]
    if card.family != "timestral":
        raise ValueError("mirrored-pass test reads Timestral internals; family must be timestral")
    df = load_processed(protocol)
    specs = window_specs(protocol)
    cov_cols = protocol.covariate_sets[cov_set]
    set_threads(None)
    adapter = card.adapter_class()(card)
    adapter.load(resolve_params(card, Experiment(name="analysis", models=[model_id])), "cpu")
    fc = adapter.fc
    levels = list(fc.quantile_levels)
    steps = ctx_days * protocol.steps_per_day
    rows = {k: [] for k in ("direct", "mirrored", "average")}
    groups: dict[int, list] = {}
    for sp in specs:
        groups.setdefault(sp.horizon, []).append(sp)
    t0 = time.perf_counter()
    for h, sps in groups.items():
        for s in range(0, len(sps), 16):
            sub = sps[s : s + 16]
            tasks = [build_task(df, sp, protocol, steps, cov_cols, levels) for sp in sub]
            ctx = np.stack([t.context.to_numpy(np.float32) for t in tasks])
            known = None
            if cov_cols:
                known = np.stack([np.concatenate([t.past_covariates.to_numpy(np.float32),
                                                  t.future_covariates.to_numpy(np.float32)]).T for t in tasks])
            with torch.inference_mode():
                inputs, mask = fc._as_context(ctx, None)
                kc = None if known is None else fc._as_covariates(known, len(sub), ctx.shape[-1] + h, "known_covariates")
                direct = fc._forecast_raw(inputs, mask, h, known_covariates=kc)
                mirr = fc._forecast_raw(-inputs, mask, h, known_covariates=None if kc is None else -kc)
                mirrored = (-mirr).flip(-1)
                average = 0.5 * (direct + mirrored)
                eligible = nonnegative_context(inputs, mask)
                outs = {k: torch.where(eligible[..., None, None], v.clamp_min(0), v)
                        for k, v in (("direct", direct), ("mirrored", mirrored), ("average", average))}
            for j, (sp, t) in enumerate(zip(sub, tasks)):
                m = scored_mask(sp, protocol).to_numpy()
                y = df.loc[t.horizon_index, "load"].to_numpy()
                od = str(sp.origin.tz_convert(protocol.timezone).date())
                for k, v in outs.items():
                    q = v[j, 0].float().numpy()  # (H, Q)
                    fr = pd.DataFrame({"origin_date": od, "step": np.arange(h), "is_scored": m, "actual": y})
                    for li, lv in enumerate(levels):
                        fr[f"P{round(lv * 100):02d}"] = q[:, li]
                    rows[k].append(fr)
    elapsed = time.perf_counter() - t0
    pts = {k: pd.concat(v, ignore_index=True) for k, v in rows.items()}
    res: dict = {"model": model_id, "covariate_set": cov_set, "context_days": ctx_days, "n_windows": len(specs),
                 "elapsed_seconds": elapsed, "passes": {}}
    for k, p in pts.items():
        s = p[p.is_scored]
        res["passes"][k] = {
            "wape_pct": M.wape(s.actual.to_numpy(), s.P50.to_numpy()),
            "crps_approx_mw": M.prob_metrics(p, protocol.eval_quantiles, protocol.interval)["crps_approx_mw"],
            "coverage_p10_p90_pct": M.prob_metrics(p, protocol.eval_quantiles, protocol.interval)["coverage_p10_p90_pct"],
            "interval_width_mw": M.prob_metrics(p, protocol.eval_quantiles, protocol.interval)["interval_width_mw"],
        }
    e = {k: (p[p.is_scored].actual - p[p.is_scored].P50).to_numpy() for k, p in pts.items()}
    res["error_correlation_direct_vs_mirrored"] = float(np.corrcoef(e["direct"], e["mirrored"])[0, 1])
    res["p50_abs_diff_direct_vs_mirrored_mw"] = float(np.mean(np.abs(e["direct"] - e["mirrored"])))
    res["p50_rmse_direct_mw"] = float(np.sqrt(np.mean(e["direct"] ** 2)))
    # paired over days: mirrored vs direct
    wins = {k: M.window_table(p, pd.DataFrame({"origin_date": sorted(p.origin_date.unique())}).set_index("origin_date"))
            for k, p in pts.items()}
    nb, seed = protocol.metrics.n_boot, protocol.metrics.bootstrap_seed
    res["paired_mirrored_vs_direct"] = M.paired_delta(wins["mirrored"], wins["direct"], nb, seed)
    res["paired_crps_mirrored_vs_direct"] = M.paired_crps_delta(
        M.crps_window_table(pts["mirrored"], protocol.eval_quantiles),
        M.crps_window_table(pts["direct"], protocol.eval_quantiles), nb, seed)
    # sanity: direct must reproduce the -nosym run, average the default run (same model, cov, ctx)
    checks = {}
    for k, run_id in (("direct", f"{model_id}-nosym"), ("average", model_id)):
        f = ROOT / "outputs" / "runs" / run_id / cov_set / f"ctx{ctx_days:02d}d" / "points.parquet"
        if f.exists():
            ref = pd.read_parquet(f, columns=["origin_date", "step", "P50"])
            j = ref.merge(pts[k][["origin_date", "step", "P50"]], on=["origin_date", "step"], suffixes=("_ref", ""))
            assert len(j) == len(ref) == len(pts[k]), "row alignment failed"
            checks[f"{k}_vs_{run_id}_max_abs_diff_P50_mw"] = float(np.abs(j.P50_ref - j.P50).max())
    res["reproduction_checks"] = checks
    out = _analyses_dir(experiment)
    (out / "mirrored_pass.json").write_text(json.dumps(res, indent=2))
    _write_mirrored_md(res, out / "mirrored_pass.md")
    return res


def _write_mirrored_md(r: dict, path: Path) -> None:
    p = r["passes"]
    tab = pd.DataFrame([{"pass": k, "WAPE ↓ %": round(v["wape_pct"], 3), "CRPS≈ ↓ MW": round(v["crps_approx_mw"], 2),
                         "P10-P90 coverage %": round(v["coverage_p10_p90_pct"], 1), "width ↓ MW": round(v["interval_width_mw"], 0)}
                        for k, v in p.items()])
    pw, pc = r["paired_mirrored_vs_direct"], r["paired_crps_mirrored_vs_direct"]
    lines = [
        f"## Mirrored-pass test ({r['model']}, {r['covariate_set']}, {r['context_days']}d, {r['n_windows']} windows)\n",
        "direct = the pass `symmetric_averaging=False` returns; mirrored = forecast of −context, negated back and "
        "quantile-flipped; average = what `symmetric_averaging=True` returns.\n",
        tab.to_markdown(index=False), "",
        f"- Mirrored vs direct, paired over days: ΔWAPE {pw['delta_wape']:+.3f} pp [{pw['delta_ci_lo']:+.3f}, "
        f"{pw['delta_ci_hi']:+.3f}], mirrored better on {pw['days_a_better']}/{pw['n_days']} days (sign p {pw['sign_test_p']:.3g}); "
        f"ΔCRPS≈ {pc['delta_crps_mw']:+.2f} MW [{pc['delta_ci_lo']:+.2f}, {pc['delta_ci_hi']:+.2f}].",
        f"- Correlation of per-point P50 errors, direct vs mirrored: **{r['error_correlation_direct_vs_mirrored']:.4f}** "
        f"(mean |P50 direct − P50 mirrored| = {r['p50_abs_diff_direct_vs_mirrored_mw']:.1f} MW; direct RMSE "
        f"{r['p50_rmse_direct_mw']:.0f} MW).",
        "- Reproduction checks (max |ΔP50| vs the stored runs, MW): "
        + (", ".join(f"{k} = {v:.3g}" for k, v in r["reproduction_checks"].items()) or "none available"),
        "",
    ]
    path.write_text("\n".join(lines))


def retime(model_ids: list[str], cov_sets: list[str], ctx_days: int, n_windows: int, repeats: int = 2) -> list[dict]:
    """Controlled latency: same windows, same session, models interleaved; best of `repeats` per model."""
    protocol = load_protocol()
    cards = load_all_cards()
    df = load_processed(protocol)
    specs = window_specs(protocol)[100 : 100 + n_windows]
    threads = set_threads(None)
    load_avg = Path("/proc/loadavg").read_text().split()[:3] if Path("/proc/loadavg").exists() else []
    adapters = {}
    for mid in model_ids:
        a = cards[mid].adapter_class()(cards[mid])
        a.load(resolve_params(cards[mid], Experiment(name="retime", models=[mid])), "cpu")
        adapters[mid] = a
    recs = []
    for cs in cov_sets:
        tasks = {mid: [build_task(df, s, protocol, ctx_days * protocol.steps_per_day, protocol.covariate_sets[cs],
                                  cards[mid].outputs.levels) for s in specs] for mid in model_ids}
        best = {mid: float("inf") for mid in model_ids}
        for mid in model_ids:
            adapters[mid].predict(tasks[mid][:1])  # warm-up
        for _ in range(repeats):
            for mid in model_ids:  # interleave so drift hits both equally
                t0 = time.perf_counter()
                adapters[mid].predict(tasks[mid])
                best[mid] = min(best[mid], (time.perf_counter() - t0) / n_windows)
        for mid in model_ids:
            recs.append({"model": mid, "covariate_set": cs, "context_days": ctx_days, "per_window_seconds": best[mid],
                         "n_windows": n_windows, "repeats": repeats, "threads": threads, "loadavg_before": load_avg,
                         "recorded_utc": pd.Timestamp.now("UTC").isoformat()})
            log.info("retime %s %s %dd: %.3f s/window", mid, cs, ctx_days, best[mid])
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    f = TIMING_DIR / "retime.json"
    old = json.loads(f.read_text()) if f.exists() else []
    keep = [o for o in old if (o["model"], o["covariate_set"], o["context_days"]) not in
            {(r["model"], r["covariate_set"], r["context_days"]) for r in recs}]
    f.write_text(json.dumps(keep + recs, indent=2))
    return recs


def load_retime() -> dict[tuple[str, str, int], dict]:
    f = TIMING_DIR / "retime.json"
    if not f.exists():
        return {}
    return {(r["model"], r["covariate_set"], r["context_days"]): r for r in json.loads(f.read_text())}
