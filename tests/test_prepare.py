"""bench prepare accepts a reordered load.csv (same content) and refuses changed content."""

import json
import random
import shutil

import pytest

from tsbench.config import ROOT
from tsbench.data.prepare import content_sha256, prepare


def test_content_hash_ignores_row_order_but_not_values(tmp_path):
    rows = [f"2025-01-01T{h:02d}:00:00+01:00;PT15M;{9000 + h}" for h in range(24)]
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    a.write_text("﻿Datetime;Resolution code;Total Load\n" + "\n".join(rows) + "\n", encoding="utf-8")
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    b.write_text("Datetime;Resolution code;Total Load\r\n" + "\r\n".join(shuffled) + "\r\n", encoding="utf-8")
    changed = rows[:]
    changed[5] = changed[5].replace("9005", "9006")
    c.write_text("﻿Datetime;Resolution code;Total Load\n" + "\n".join(changed) + "\n", encoding="utf-8")
    assert content_sha256(a) == content_sha256(b) != content_sha256(c)


def _real_data(protocol):
    if not all((ROOT / p).exists() for p in (protocol.dataset.raw_load, protocol.dataset.raw_weather,
                                             protocol.dataset.manifest)):
        pytest.skip("raw data not present (not committed): run scripts/download_data.py and `bench prepare`")


def _copy_inputs(protocol, root, load_lines):
    for rel in (protocol.dataset.raw_load, protocol.dataset.raw_weather):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / protocol.dataset.raw_weather, root / protocol.dataset.raw_weather)
    (root / protocol.dataset.raw_load).write_text("\n".join(load_lines) + "\n", encoding="utf-8-sig")


def test_prepare_accepts_shuffled_load_with_identical_output(protocol, tmp_path):
    _real_data(protocol)
    lines = (ROOT / protocol.dataset.raw_load).read_bytes().decode("utf-8-sig").splitlines()
    body = lines[1:]
    random.Random(42).shuffle(body)
    _copy_inputs(protocol, tmp_path, [lines[0], *body])
    assert content_sha256(tmp_path / protocol.dataset.raw_load) == protocol.dataset.raw_load_content_sha256
    prepare(protocol, root=tmp_path)
    got = json.loads((tmp_path / protocol.dataset.manifest).read_text())["processed_sha256"]
    want = json.loads((ROOT / protocol.dataset.manifest).read_text())["processed_sha256"]
    assert got == want  # same processed data, so every run_id stays valid


def test_prepare_refuses_changed_load_content(protocol, tmp_path):
    _real_data(protocol)
    lines = (ROOT / protocol.dataset.raw_load).read_bytes().decode("utf-8-sig").splitlines()
    fields = lines[1].split(";")
    fields[2] = f"{float(fields[2]) + 1:.2f}"  # one Total Load value changed by 1 MW
    _copy_inputs(protocol, tmp_path, [lines[0], ";".join(fields), *lines[2:]])
    with pytest.raises(RuntimeError, match="load content differs"):
        prepare(protocol, root=tmp_path)
