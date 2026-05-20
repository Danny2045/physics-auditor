"""Regression test for benchmark/results/pocket_clashscore_comparison.json.

Locks the committed snapshot against silent drift from changes to shared
infrastructure (parser, topology, clash kernel, bonded-mask construction).

Background — Finding 11 / PR-DET (closed):
    Commit 1e2d7a6 (May 9 2026) added (C, OXT) to BACKBONE_BONDS and
    introduced disulfide-bond inference for CYS-SG pairs at 1.9-2.4 Å.
    Both fixes correctly enlarged the bonded-pair set, shrinking the
    non-bonded mask and reducing clash counts. The commit's claim that
    'clashscore numbers are unchanged; this affects LJ-energy decomposition
    only' was incorrect because the bonded mask is shared infrastructure
    between the LJ kernel and the clash kernel. Nobody regenerated
    pocket_clashscore_comparison.json, so it froze at bug-inflated values
    until PR-DET. This test exists to make that class of silent
    propagation impossible: any future topology- or kernel-affecting
    change must either regenerate the JSON or fail this test.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_JSON = REPO_ROOT / "benchmark" / "results" / "pocket_clashscore_comparison.json"

# Floats in the committed JSON are stored as round(value, 2) by
# benchmark_pocket_clashscore.run_benchmark, so equality at 1e-3 is the
# strictest meaningful tolerance.
FLOAT_ABS_TOL = 1e-3

FLOAT_FIELDS = (
    "exp_whole_clashscore",
    "exp_pocket_clashscore",
    "af_whole_clashscore",
    "af_pocket_clashscore",
    "whole_ratio",
    "pocket_ratio",
)
INT_FIELDS = (
    "pocket_n_residues",
    "pocket_n_atoms_exp",
    "pocket_n_atoms_af",
)
STRING_FIELDS = (
    "name",
    "experimental_pdb",
    "predicted_pdb",
)

EXPECTED_PAIR_COUNT = 11


def test_benchmark_clashscore_regenerates_identically(monkeypatch):
    """Rerunning the 11-pair benchmark must reproduce the committed JSON."""
    # The benchmark module references structure files via cwd-relative
    # paths (benchmark/structures/...), so the test runs from the repo root.
    monkeypatch.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    assert COMMITTED_JSON.exists(), (
        f"Committed JSON missing at {COMMITTED_JSON}. "
        "PR-DET should have created it."
    )
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)

    from benchmark.benchmark_pocket_clashscore import run_benchmark

    with tempfile.TemporaryDirectory() as tmpdir:
        fresh_path = Path(tmpdir) / "fresh_pocket_clashscore.json"
        fresh = run_benchmark(output_path=fresh_path)

    assert len(committed) == EXPECTED_PAIR_COUNT, (
        f"Committed JSON has {len(committed)} pairs; expected "
        f"{EXPECTED_PAIR_COUNT}. PR-A1 schema requires exactly 11."
    )
    assert len(fresh) == EXPECTED_PAIR_COUNT, (
        f"Fresh run produced {len(fresh)} pairs; expected "
        f"{EXPECTED_PAIR_COUNT}. A BENCHMARK_PAIRS edit needs to be "
        "reflected in the committed JSON and this test."
    )

    by_name_committed = {p["name"]: p for p in committed}
    by_name_fresh = {p["name"]: p for p in fresh}
    assert set(by_name_committed.keys()) == set(by_name_fresh.keys()), (
        f"Pair-name drift: committed={sorted(by_name_committed)} "
        f"fresh={sorted(by_name_fresh)}"
    )

    for name, c in by_name_committed.items():
        f = by_name_fresh[name]
        for field in STRING_FIELDS:
            assert c[field] == f[field], (
                f"{name}.{field}: committed={c[field]!r} != fresh={f[field]!r}"
            )
        for field in INT_FIELDS:
            assert c[field] == f[field], (
                f"{name}.{field}: committed={c[field]} != fresh={f[field]}"
            )
        for field in FLOAT_FIELDS:
            assert math.isclose(
                c[field], f[field], abs_tol=FLOAT_ABS_TOL
            ), (
                f"{name}.{field}: committed={c[field]} fresh={f[field]} "
                f"(abs diff {abs(c[field] - f[field]):.6f} > {FLOAT_ABS_TOL}). "
                "If this drift is intentional, regenerate the JSON "
                "(python benchmark/benchmark_pocket_clashscore.py) and "
                "commit it together with the topology/kernel change."
            )
