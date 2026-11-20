"""Download the two raw input files and check them against the hashes in configs/protocol.yaml.

    uv run python scripts/download_data.py            # writes data/load.csv and data/weather_cov.csv
    uv run python scripts/download_data.py --force    # overwrite existing files

Sources (exact requests below):
- data/load.csv: Elia Open Data ods001 ("Measured and forecasted total load on the Belgian grid"), CSV export with
  column labels, ';' delimiter, Europe/Brussels timestamps, 2024-01-01 to 2025-12-31 local time (70,176 rows).
- data/weather_cov.csv: Open-Meteo Previous Runs API, grid cell near Charleroi (50.458 N, 4.437 E),
  hourly `temperature_2m_previous_day1`, 2024-01-01 to 2025-12-31, GMT, CSV.

A hash mismatch is reported as a WARNING, not an error: Elia's export does not return rows in a stable order, and
Elia or Open-Meteo may revise historical values. For the load file the script therefore also compares a
row-order-independent content hash, which tells "same data, reordered" apart from "data changed".
`bench prepare` accepts a load file whose content hash matches protocol.yaml (`raw_load_content_sha256`) and refuses
any other mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from tsbench.data.prepare import content_sha256

ROOT = Path(__file__).resolve().parents[1]

ELIA_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/exports/csv?" + urllib.parse.urlencode(
    {
        "lang": "en",
        "timezone": "Europe/Brussels",
        "use_labels": "true",
        "delimiter": ";",
        "where": "datetime >= date'2024-01-01T00:00:00+01:00' and datetime < date'2026-01-01T00:00:00+01:00'",
    }
)
OPEN_METEO_URL = "https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
    {
        "latitude": "50.458",
        "longitude": "4.437",
        "hourly": "temperature_2m_previous_day1",
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "timezone": "GMT",
        "format": "csv",
    }
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tsbench-download/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite existing files in data/")
    args = ap.parse_args()

    ds = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())["dataset"]
    jobs = [
        ("load", ROOT / ds["raw_load"], ELIA_URL, ds["raw_load_sha256"]),
        ("weather", ROOT / ds["raw_weather"], OPEN_METEO_URL, ds["raw_weather_sha256"]),
    ]
    mismatches = 0
    for name, path, url, expected in jobs:
        if path.exists() and not args.force:
            print(f"{path.relative_to(ROOT)} exists; skipping download (use --force to overwrite)")
            data = path.read_bytes()
        else:
            print(f"downloading {name}: {url}")
            data = fetch(url)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"wrote {path.relative_to(ROOT)} ({len(data):,} bytes)")
        got = sha256(data)
        if got == expected:
            print(f"  OK: SHA-256 matches protocol.yaml ({got})")
            continue
        mismatches += 1
        print(f"  WARNING: SHA-256 mismatch for {path.relative_to(ROOT)}\n"
              f"    expected {expected}\n    got      {got}", file=sys.stderr)
        if name == "load":
            if content_sha256(path) == ds["raw_load_content_sha256"]:
                print("    Same rows and values as the benchmark file, in a different order (Elia's export order is not "
                      "stable). `bench prepare` accepts it.", file=sys.stderr)
            else:
                print("    Content differs from the benchmark file: Elia may have revised historical data.",
                      file=sys.stderr)
    if mismatches:
        print(f"WARNING: {mismatches} file(s) differ byte-for-byte from configs/protocol.yaml. `bench prepare` accepts "
              "a load file whose content hash matches (row order only) and refuses anything else. Results computed on "
              "different data are not comparable with the published ones.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
