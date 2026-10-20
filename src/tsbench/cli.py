"""bench CLI."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

from tsbench.cards import load_all_cards
from tsbench.config import ROOT, load_experiment, load_protocol
from tsbench.registry import expand_models
from tsbench.runner import Cell, expand_cells, run_model_cells

app = typer.Typer(no_args_is_help=True, add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bench")


@app.command()
def prepare() -> None:
    """Raw CSVs -> data/processed parquet + manifest."""
    from tsbench.data.prepare import prepare as prep

    typer.echo(prep(load_protocol()))


@app.command("list-models")
def list_models() -> None:
    cards = load_all_cards()
    typer.echo(f"{'id':26}{'kind':10}{'max_ctx':>8} {'cov(p/f/c)':12}{'quantiles':>10}  device  adapter_v")
    for c in cards.values():
        cv = "/".join(str(int(x)) for x in (c.covariates.past, c.covariates.future, c.covariates.categorical))
        typer.echo(
            f"{c.id:26}{c.kind:10}{c.max_context!s:>8} {cv:12}{len(c.outputs.levels):>10}  "
            f"{','.join(c.device):7} {c.adapter_version}"
        )
    from tsbench.config import load_groups

    typer.echo("\ngroups: " + json.dumps(load_groups()))


def _resolve(config: Path, models: str | None, model: str | None, context: int | None, cov: str | None) -> tuple:
    exp = load_experiment(config)
    protocol = load_protocol()
    cards = load_all_cards()
    if models:
        exp.models, exp.exclude = [m.strip() for m in models.split(",")], []
    if model:
        exp.models, exp.exclude = [model], []
    if context is not None:
        exp.contexts_days = [context]
    if cov:
        exp.covariate_sets = [cov]
    ids = expand_models(exp, cards)
    return exp, protocol, cards, ids


@app.command()
def run(
    config: Path = typer.Option(..., "--config"),
    models: str = typer.Option(None, "--models", help="comma-separated ids; overrides config"),
    model: str = typer.Option(None, "--model"),
    context: int = typer.Option(None, "--context"),
    covariates: str = typer.Option(None, "--covariates"),
) -> None:
    """Run the matrix (each model in its own subprocess, cells in series)."""
    exp, protocol, cards, ids = _resolve(config, models, model, context, covariates)
    exp_dir = ROOT / "outputs" / "experiments" / exp.name
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    # persist the (possibly CLI-overridden) config as run
    import yaml

    (exp_dir / "experiment_config.yaml").write_text(yaml.safe_dump(exp.model_dump()))
    cfg_path = exp_dir / "experiment_config.yaml"
    status_file = exp_dir / "status.jsonl"
    status_file.write_text("")
    for mid in ids:
        cmd = [sys.executable, "-m", "tsbench.cli", "worker", "--config", str(cfg_path), "--model", mid,
               "--status-file", str(status_file)]
        rc = _run_worker(cmd, exp_dir / "logs" / f"{mid}.log")
        if rc != 0:  # worker crashed: mark its cells that have no status yet as failed
            done = {json.loads(l)["cell"] for l in status_file.read_text().splitlines() if l}
            for c in expand_cells(exp, protocol, cards, [mid]):
                if c.label not in done:
                    _write_status(status_file, c, "", "failed", f"worker exit code {rc}", 0.0)
        typer.echo(f"finished {mid}")
    rows = [json.loads(l) for l in status_file.read_text().splitlines() if l]
    st = pd.DataFrame(rows)
    st.to_csv(exp_dir / "status.csv", index=False)
    typer.echo(st.groupby("status").size().to_string())
    typer.echo(st[st.status == "failed"].to_string() if (st.status == "failed").any() else "no failures")


def _run_worker(cmd: list[str], log_path: Path) -> int:
    """Run a worker, streaming its output (incl. tqdm's carriage returns) to the terminal AND the log file."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "wb") as lg:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, env=env, bufsize=0)
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while chunk := os.read(fd, 4096):
            lg.write(chunk)
            lg.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return proc.wait()


def _write_status(path: Path, c: Cell, rid: str, status: str, reason: str, dur: float) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"cell": c.label, "run_id": rid, "status": status, "reason": reason, "duration_s": round(dur, 1)}) + "\n")


@app.command(hidden=True)
def worker(config: Path = typer.Option(...), model: str = typer.Option(...), status_file: Path = typer.Option(...)) -> None:
    exp = load_experiment(config)
    protocol = load_protocol()
    cards = load_all_cards()
    cells = expand_cells(exp, protocol, cards, [model])
    run_model_cells(exp, protocol, cards[model], cells, lambda c, rid, st, why, dur: _write_status(status_file, c, rid, st, why, dur))


@app.command()
def summarize(config: Path = typer.Option(..., "--config")) -> None:
    from tsbench.summarize import summarize as summ

    exp = load_experiment(config)
    summ(exp, load_protocol(), load_all_cards())


@app.command("leakage-test")
def leakage_test(model: str = typer.Option(..., "--model"), covariates: str = "full", context: int = 30) -> None:
    """Noise after 17:45 must not change predictions. Records outputs/gates/<model>.json."""
    from tsbench.leakage import leakage_test as lt

    raise typer.Exit(0 if lt(model, covariates, context) else 1)


@app.command("determinism-test")
def determinism_test(model: str = typer.Option(..., "--model"), covariates: str = "full", context: int = 15,
                     windows: int = 8) -> None:
    """Same cell twice -> bit-identical quantiles. Records outputs/gates/<model>.json."""
    from tsbench.leakage import determinism_test as dt

    raise typer.Exit(0 if dt(model, covariates, context, windows) else 1)


@app.command()
def gates(config: Path = typer.Option(..., "--config"), force: bool = typer.Option(False, "--force")) -> None:
    """Leakage + determinism for every model in the config (each in its own subprocess).
    Skips a model whose recorded gates match its current revision and adapter_version, unless --force."""
    from tsbench.leakage import load_gates

    exp = load_experiment(config)
    cards = load_all_cards()
    for mid in expand_models(exp, cards):
        rec = load_gates(mid)
        fresh = all(
            k in rec and rec[k]["revision"] == cards[mid].revision and rec[k]["adapter_version"] == cards[mid].adapter_version
            for k in ("leakage", "determinism")
        )
        if fresh and not force:
            typer.echo(f"{mid}: gates up to date (leakage {rec['leakage']['passed']}, determinism {rec['determinism']['passed']})")
            continue
        for cmd in ("leakage-test", "determinism-test"):
            rc = subprocess.run([sys.executable, "-m", "tsbench.cli", cmd, "--model", mid], cwd=ROOT).returncode
            typer.echo(f"{mid} {cmd}: {'PASS' if rc == 0 else 'FAIL'}")


if __name__ == "__main__":
    app()
