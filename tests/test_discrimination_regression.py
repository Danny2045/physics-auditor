"""Regression test for the clash-discrimination benchmark (charter v0.1).

Locks the committed ``benchmark/discrimination/discrimination_results.json``
headline AUROCs and integrity invariants against silent drift, and proves the
runner's scoring orchestration reproduces the library ``audit()`` clashscore
(the "committed clash check, unmodified" invariant) on a committed fixture.

Layers:

1. **Always-on JSON lock** — load the committed results and assert the headline
   AUROCs, bars, recognition rate, near-native control, counts, frozen
   ClashConfig, and §6 mapping. CI-safe (no 3DRobot data needed).
2. **AUROC unit lock** — the dependency-free AUROC matches known-answer cases.
3. **Fidelity (committed fixture)** — ``run_discrimination.clashscore_of_structure``
   equals ``audit()``'s clashscore, proving identical use of the committed check.
4. **Recompute (opt-in)** — with ``PHYSICS_AUDITOR_RUN_3DROBOT=1`` and the
   dataset present, re-run the full benchmark and confirm the headline AUROCs
   reproduce within tolerance. Skipped otherwise (mirrors the ESM divergence
   test's external-data skip).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DISCRIM_DIR = REPO_ROOT / "benchmark" / "discrimination"
RESULTS_JSON = DISCRIM_DIR / "discrimination_results.json"
RUNNER_PY = DISCRIM_DIR / "run_discrimination.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "clashing.pdb"

# Committed headline values (locked; charter v0.1 bars 0.95 / 0.65).
EXPECTED_TIER1_AUROC_SIGMA2 = 1.0
EXPECTED_TIER2_POOLED_AUROC = 0.999996
EXPECTED_TIER2_RECOGNITION = 1.0
EXPECTED_NEARNATIVE_EXCEEDS = 30


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_discrimination", RUNNER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so @dataclass can resolve the module
    spec.loader.exec_module(mod)
    return mod


def _load_results() -> dict:
    assert RESULTS_JSON.exists(), f"committed results missing: {RESULTS_JSON}"
    return json.loads(RESULTS_JSON.read_text())


# --------------------------------------------------------------------------- #
# 1. Always-on JSON lock                                                       #
# --------------------------------------------------------------------------- #
def test_committed_headline_auroc_locked():
    r = _load_results()
    # Tier 1 — positive control PASS, monotone, every sigma AUROC 1.0
    assert r["tier1"]["auroc_sigma_2.0"] == EXPECTED_TIER1_AUROC_SIGMA2
    assert r["tier1"]["bar"] == 0.95
    assert r["tier1"]["passed"] is True
    for sig in ("0.5", "1.0", "1.5", "2.0"):
        assert r["tier1"]["per_sigma"][sig]["auroc"] == 1.0
    means = [r["tier1"]["per_sigma"][s]["mean_decoy_clashscore"] for s in ("0.5", "1.0", "1.5", "2.0")]
    assert means == sorted(means), "Tier-1 dose-response must be monotone in sigma"

    # Tier 2 — surprising-positive result, locked
    assert r["tier2"]["auroc_pooled"] == EXPECTED_TIER2_POOLED_AUROC
    assert r["tier2"]["bar"] == 0.65
    assert r["tier2"]["meets_bar"] is True
    assert r["tier2"]["native_recognition_rate"] == EXPECTED_TIER2_RECOGNITION


def test_integrity_invariants_locked():
    r = _load_results()
    # pre-registration constants
    assert r["seed"] == 20260613
    assert r["tier1"]["design"]["k_p_per_native_per_sigma"] == 50
    assert r["tier1"]["design"]["sigmas"] == [0.5, 1.0, 1.5, 2.0]
    assert r["native_draw_lock_commit"] == "575a4af"
    # committed clash check, unmodified; frozen config
    assert "check_clashes" in r["clashscore_source"]
    assert "unmodified" in r["clashscore_source"]
    assert r["clash_config"] == {
        "vdw_tolerance": 0.4,
        "ignore_bonded_neighbors": 3,
        "severe_clash_threshold": 0.2,
    }
    # counts: all 30 targets scored, nothing silently dropped
    assert r["n_targets_requested"] == 30
    assert r["n_targets_scored"] == 30
    assert r["excluded_targets"] == []
    assert r["tier1"]["n_decoys_total"] == 6000
    assert r["tier1"]["n_decoys_excluded"] == 0
    assert r["tier2"]["n_decoys_scored"] == 9000
    assert r["tier2"]["n_decoys_excluded"] == 0


def test_nearnative_confound_locked():
    """The near-native control: lowest-RMSD decoys still out-clash natives 30/30
    — the Tier-2 separation is a preparation confound, not fold recognition."""
    r = _load_results()
    nn = r["tier2"]["nearnative_control"]
    assert nn["n_targets"] == 30
    assert nn["n_targets_nearnative_exceeds_native"] == EXPECTED_NEARNATIVE_EXCEEDS
    # near-native decoys are geometrically close to the native fold yet clash more
    assert nn["max_min_rmsd_angstrom"] <= 2.0
    assert nn["mean_nearnative_decoy_clashscore"] > nn["mean_native_clashscore"]


def test_charter_section6_mapping():
    r = _load_results()
    assert "row 1" in r["outcome_table_mapping"]["tier1_row"]
    assert "row 4" in r["outcome_table_mapping"]["tier2_row"]
    assert r.get("stopped_after_tier1") is False


# --------------------------------------------------------------------------- #
# 2. AUROC unit lock                                                           #
# --------------------------------------------------------------------------- #
def test_auroc_known_answers():
    mod = _load_runner()
    import numpy as np

    # perfect: positives (decoys, label 1) all score higher than negatives
    assert mod.auroc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) == 1.0
    # reversed: positives all lower
    assert mod.auroc(np.array([4.0, 3.0, 2.0, 1.0]), np.array([0, 0, 1, 1])) == 0.0
    # all tied → 0.5
    assert mod.auroc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 0, 1, 1])) == 0.5


# --------------------------------------------------------------------------- #
# 3. Fidelity — runner uses the committed clash check, unmodified              #
# --------------------------------------------------------------------------- #
def test_runner_matches_audit_clashscore():
    from physics_auditor.audit import StructureAuditRequest, audit
    from physics_auditor.core.parser import parse_pdb

    mod = _load_runner()
    my_cs = mod.clashscore_of_structure(parse_pdb(FIXTURE)).clashscore
    audit_cs = audit(StructureAuditRequest(pdb_path=FIXTURE)).checks["steric_clashes"]["clashscore"]
    # audit() rounds the clashscore to 2 dp; match at that precision.
    assert round(my_cs, 2) == audit_cs


# --------------------------------------------------------------------------- #
# 4. Recompute (opt-in; needs the uncommitted 3DRobot dataset)                 #
# --------------------------------------------------------------------------- #
def test_recompute_reproduces_headline():
    if os.environ.get("PHYSICS_AUDITOR_RUN_3DROBOT") != "1":
        pytest.skip("set PHYSICS_AUDITOR_RUN_3DROBOT=1 to re-run the full benchmark")
    mod = _load_runner()
    if not mod.DEFAULT_DATA_DIR.exists():
        pytest.skip(f"3DRobot data not present at {mod.DEFAULT_DATA_DIR}")
    res = mod.run_benchmark(mod.DEFAULT_DATA_DIR, mod.load_native_ids())
    assert abs(res["tier1"]["auroc_sigma_2.0"] - EXPECTED_TIER1_AUROC_SIGMA2) < 1e-9
    assert abs(res["tier2"]["auroc_pooled"] - EXPECTED_TIER2_POOLED_AUROC) < 1e-4
