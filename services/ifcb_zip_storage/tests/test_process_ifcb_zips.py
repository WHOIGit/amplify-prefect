"""Tests for the ifcbkit-based ZIP builder.

The fixtures are two real bins: an I-style bin with stitched pairs and a
D-style bin. pyifcb_zip_manifest.json holds the SHA-256 of every entry that
pyifcb v1.2.1's bin2zip_stream produced for them, so these tests check that
the ZIPs are byte-identical to the ones made before the switch to ifcbkit.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "ifcb"
MANIFEST = json.loads((HERE / "fixtures" / "pyifcb_zip_manifest.json").read_text())["bins"]

spec = importlib.util.spec_from_file_location(
    "process_ifcb_zips", HERE.parent / "src" / "process_ifcb_zips.py"
)
process_ifcb_zips = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_ifcb_zips)


def test_lists_fixture_bins():
    filesets = process_ifcb_zips.list_filesets(str(FIXTURES))
    assert sorted(fs["pid"] for fs in filesets) == sorted(MANIFEST)
    for fs in filesets:
        for ext in ("hdr", "adc", "roi"):
            assert fs[ext] == str(FIXTURES / f"{fs['pid']}.{ext}")


@pytest.mark.parametrize("pid", sorted(MANIFEST))
def test_zip_matches_pyifcb(pid):
    fileset = {ext: str(FIXTURES / f"{pid}.{ext}") for ext in ("hdr", "adc", "roi")}
    zf = zipfile.ZipFile(process_ifcb_zips.bin2zip_stream(pid, fileset))
    actual = [[name, hashlib.sha256(zf.read(name)).hexdigest()] for name in zf.namelist()]
    expected = MANIFEST[pid]
    assert [name for name, _ in actual] == [name for name, _ in expected]
    mismatched = [a[0] for a, e in zip(actual, expected) if a[1] != e[1]]
    assert mismatched == []


def test_adc_csv_matches_pyifcb():
    # The first 120 lines of a 2026 Oleander ADC. Unlike the bins above, it has
    # values that pandas 3 parses differently, so this guards the pandas<3 pin.
    adc = (HERE / "fixtures" / "D20260709T105717_IFCB188_head.adc").read_bytes()
    expected = (HERE / "fixtures" / "D20260709T105717_IFCB188_head.pyifcb.csv").read_text()
    assert process_ifcb_zips.adc_to_csv(adc, schema=2) == expected


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any directory")
def test_skips_unreadable_directories(tmp_path):
    good = tmp_path / "D20160601"
    good.mkdir()
    pid = "D20160601T092459_IFCB107"
    for ext in ("hdr", "adc", "roi"):
        shutil.copy(FIXTURES / f"{pid}.{ext}", good)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        filesets = process_ifcb_zips.list_filesets(str(tmp_path))
    finally:
        locked.chmod(0o755)
    assert [fs["pid"] for fs in filesets] == [pid]
