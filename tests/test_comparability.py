"""Tests for the composable opt-in comparability gate.

Three sub-criteria, each independently enabled by providing its data:

  - anchor RMSD ≤ cutoff
  - pocket-size parity ≤ tolerance
  - shared cofactor atom-name count ≥ min

Plus a **behavior-equivalence pin**: for every combination of inputs, the
gate's ``passed`` must equal the legacy inline boolean that lived in
``run_w107f_selectivity.py`` before the extraction:

    anchor_rmsd <= 1.0
    and abs(target_pocket_size - ortholog_pocket_size) <= 4
    and len(shared_atom_names) >= 3

This pin is the load-bearing assurance that the w107f verdict cannot
silently drift away from the legacy expression even if the regression
dossier ever loosens — it tests the LOGIC, not the recorded numbers.
"""

from __future__ import annotations

import json

import pytest

from physics_auditor.causality.comparability import (
    ComparabilityCriterion,
    ComparabilityResult,
    comparability_gate,
)

# --- Per-criterion boundary tests --------------------------------------------


def test_anchor_rmsd_passes_at_or_below_cutoff():
    """Equal to cutoff is PASS (`<=` semantics)."""
    r = comparability_gate(anchor_rmsd=1.0, anchor_rmsd_cutoff=1.0)
    assert r.verdict == "PASS"
    assert r.passed is True
    assert len(r.criteria) == 1
    assert r.criteria[0].name == "anchor_rmsd"
    assert r.criteria[0].observed == 1.0
    assert r.criteria[0].threshold == 1.0
    assert r.criteria[0].passed is True


def test_anchor_rmsd_fails_above_cutoff():
    r = comparability_gate(anchor_rmsd=1.0001, anchor_rmsd_cutoff=1.0)
    assert r.verdict == "FAIL"
    assert r.passed is False
    assert r.criteria[0].passed is False


def test_pocket_parity_passes_at_or_below_tolerance():
    """|Δn| equal to tolerance is PASS."""
    r = comparability_gate(
        target_pocket_size=11,
        ortholog_pocket_size=7,
        pocket_parity_tolerance=4,
    )
    assert r.passed is True
    assert r.criteria[0].name == "pocket_parity"
    assert r.criteria[0].observed == 4
    assert r.criteria[0].threshold == 4


def test_pocket_parity_fails_above_tolerance():
    r = comparability_gate(
        target_pocket_size=11,
        ortholog_pocket_size=5,
        pocket_parity_tolerance=4,
    )
    assert r.passed is False
    assert r.criteria[0].observed == 6


def test_shared_atoms_passes_at_or_above_min():
    r = comparability_gate(
        shared_atom_names=["P", "O5'", "C5'"], min_shared_atoms=3
    )
    assert r.passed is True
    assert r.criteria[0].name == "shared_atom_names"
    assert r.criteria[0].observed == 3
    assert r.criteria[0].threshold == 3


def test_shared_atoms_fails_below_min():
    r = comparability_gate(
        shared_atom_names=["P", "O5'"], min_shared_atoms=3
    )
    assert r.passed is False
    assert r.criteria[0].observed == 2


# --- Composition semantics ---------------------------------------------------


def test_no_criteria_enabled_returns_not_assessed():
    """An honest empty gate refuses to mint a PASS."""
    r = comparability_gate()
    assert r.verdict == "NOT_ASSESSED"
    assert r.passed is False
    assert r.criteria == []
    assert "no comparability criteria were enabled" in r.reason


def test_pass_requires_every_enabled_criterion_to_pass():
    """If any enabled criterion fails, the overall verdict is FAIL."""
    r = comparability_gate(
        anchor_rmsd=0.5, anchor_rmsd_cutoff=1.0,        # PASS
        target_pocket_size=11, ortholog_pocket_size=11,
        pocket_parity_tolerance=4,                      # PASS
        shared_atom_names=["P"], min_shared_atoms=3,    # FAIL (1 < 3)
    )
    assert r.verdict == "FAIL"
    assert r.passed is False
    failed = [c for c in r.criteria if not c.passed]
    assert len(failed) == 1
    assert failed[0].name == "shared_atom_names"
    assert "shared_atom_names" in r.reason


def test_all_three_enabled_passing():
    """The w107f live values today."""
    r = comparability_gate(
        anchor_rmsd=0.8652, anchor_rmsd_cutoff=1.0,
        target_pocket_size=11, ortholog_pocket_size=9,
        pocket_parity_tolerance=4,
        shared_atom_names=["A"] * 44, min_shared_atoms=3,
    )
    assert r.verdict == "PASS"
    assert [c.name for c in r.criteria] == [
        "anchor_rmsd",
        "pocket_parity",
        "shared_atom_names",
    ]
    assert all(c.passed for c in r.criteria)


def test_criteria_are_ordered_anchor_then_parity_then_shared():
    """Documented criterion order is stable for downstream serializers."""
    r = comparability_gate(
        shared_atom_names=["A"] * 5, min_shared_atoms=3,
        target_pocket_size=10, ortholog_pocket_size=10,
        pocket_parity_tolerance=4,
        anchor_rmsd=0.5, anchor_rmsd_cutoff=1.0,
    )
    assert [c.name for c in r.criteria] == [
        "anchor_rmsd",
        "pocket_parity",
        "shared_atom_names",
    ]


# --- Partial / mismatched args are programmer errors -------------------------


def test_partial_anchor_args_raises():
    with pytest.raises(ValueError, match="anchor_rmsd_cutoff"):
        comparability_gate(anchor_rmsd=0.5)
    with pytest.raises(ValueError, match="anchor_rmsd"):
        comparability_gate(anchor_rmsd_cutoff=1.0)


def test_partial_pocket_parity_args_raises():
    """Any 1 or 2 of the 3 parity args (without all 3) is a programming error."""
    with pytest.raises(ValueError, match="ALL of target_pocket_size"):
        comparability_gate(target_pocket_size=11)
    with pytest.raises(ValueError, match="ALL of target_pocket_size"):
        comparability_gate(target_pocket_size=11, ortholog_pocket_size=9)


def test_partial_shared_atoms_args_raises():
    with pytest.raises(ValueError, match="min_shared_atoms"):
        comparability_gate(shared_atom_names=["A"])
    with pytest.raises(ValueError, match="shared_atom_names"):
        comparability_gate(min_shared_atoms=3)


# --- Serialization -----------------------------------------------------------


def test_to_dict_is_json_serializable_and_includes_provenance():
    r = comparability_gate(
        anchor_rmsd=0.8652, anchor_rmsd_cutoff=1.0,
        target_pocket_size=11, ortholog_pocket_size=9,
        pocket_parity_tolerance=4,
        shared_atom_names=["P", "O5'", "C5'", "C4'"], min_shared_atoms=3,
    )
    d = r.to_dict()
    json.dumps(d)
    assert d["verdict"] == "PASS"
    assert d["passed"] is True
    assert len(d["criteria"]) == 3
    assert d["shared_atom_names"] == ["P", "O5'", "C5'", "C4'"]


def test_dataclass_types_are_concrete():
    """ComparabilityCriterion / ComparabilityResult are real dataclasses."""
    r = comparability_gate(anchor_rmsd=0.5, anchor_rmsd_cutoff=1.0)
    assert isinstance(r, ComparabilityResult)
    assert isinstance(r.criteria[0], ComparabilityCriterion)


# --- Behavior-equivalence pin (load-bearing) ---------------------------------


def _legacy_w107f_expression(
    anchor_rmsd: float,
    n_target: int,
    n_ortholog: int,
    shared_atom_names: list[str],
) -> bool:
    """The exact inline boolean that lived in run_w107f_selectivity.py
    before the extraction. This is the ground truth the new gate must
    reproduce for the w107f thresholds.
    """
    return (
        anchor_rmsd <= 1.0
        and abs(n_target - n_ortholog) <= 4
        and len(shared_atom_names) >= 3
    )


@pytest.mark.parametrize(
    "anchor_rmsd, n_target, n_ortholog, shared_atom_names",
    [
        # The live w107f numbers — must PASS.
        (0.8652, 11, 9, ["X"] * 44),
        # All three thresholds at the exact PASS boundary.
        (1.0, 11, 7, ["X"] * 3),
        # Anchor RMSD just over → FAIL.
        (1.0001, 11, 9, ["X"] * 44),
        # Pocket-size parity just over → FAIL.
        (0.5, 11, 6, ["X"] * 44),
        # Shared-atoms just under → FAIL.
        (0.5, 11, 9, ["X"] * 2),
        # Combinations: 2 of 3 fail.
        (1.5, 11, 6, ["X"] * 44),
        (0.5, 11, 6, ["X"] * 2),
        # All three fail.
        (2.0, 20, 5, ["X"]),
        # Edge: identical pockets, zero RMSD, lots of atoms.
        (0.0, 10, 10, ["X"] * 10),
        # Edge: ortholog larger than target (parity is absolute).
        (0.5, 9, 11, ["X"] * 5),
    ],
)
def test_gate_passed_equals_legacy_w107f_boolean(
    anchor_rmsd, n_target, n_ortholog, shared_atom_names
):
    """For every w107f-thresholds case, the gate's `passed` must equal the
    legacy inline boolean. This pin survives even if the dossier values
    drift — it locks the LOGIC, not the recorded numbers.
    """
    expected = _legacy_w107f_expression(
        anchor_rmsd, n_target, n_ortholog, shared_atom_names
    )
    r = comparability_gate(
        anchor_rmsd=anchor_rmsd, anchor_rmsd_cutoff=1.0,
        target_pocket_size=n_target,
        ortholog_pocket_size=n_ortholog,
        pocket_parity_tolerance=4,
        shared_atom_names=shared_atom_names, min_shared_atoms=3,
    )
    assert r.passed is expected, (
        f"gate.passed={r.passed} but legacy expression={expected} for "
        f"anchor_rmsd={anchor_rmsd}, n_target={n_target}, "
        f"n_ortholog={n_ortholog}, |shared|={len(shared_atom_names)}; "
        f"reason: {r.reason}"
    )
