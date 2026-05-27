"""Tests for the honest partial composite (Step 0: score honesty).

Verifies that the composite is computed over ONLY the scored checks, that its
provenance truthfully reports partial-ness and the unscored / not-implemented
breakdown, and that no unbuilt check can be silently weighted into the score.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from physics_auditor.cli import app
from physics_auditor.composite import (
    CHECK_REGISTRY,
    N_TOTAL_CHECKS,
    CheckStatus,
    compute_composite,
)
from physics_auditor.config import CompositeConfig

FIXTURES = Path(__file__).parent / "fixtures"


def test_registry_matches_documented_status():
    """Exactly 1 scored, 2 computed-but-unscored, 5 not-implemented, of 8."""
    assert N_TOTAL_CHECKS == 8
    scored = [n for n, (s, _) in CHECK_REGISTRY.items() if s is CheckStatus.SCORED]
    unscored = [
        n for n, (s, _) in CHECK_REGISTRY.items() if s is CheckStatus.COMPUTED_UNSCORED
    ]
    not_impl = [
        n for n, (s, _) in CHECK_REGISTRY.items() if s is CheckStatus.NOT_IMPLEMENTED
    ]
    assert scored == ["steric_clashes"]
    assert set(unscored) == {"lennard_jones", "ramachandran"}
    assert set(not_impl) == {
        "bond_geometry",
        "peptide_planarity",
        "chirality",
        "rotamers",
        "disulfides",
    }


def test_single_scored_check_composite_equals_subscore():
    """With one scored check the normalized weight is 1.0 → composite == subscore."""
    prov = compute_composite({"steric_clashes": 0.42})
    assert prov.score == pytest.approx(0.42)
    assert len(prov.contributing_checks) == 1
    c = prov.contributing_checks[0]
    assert c.check == "steric_clashes"
    assert c.normalized_weight == pytest.approx(1.0)
    assert c.subscore == pytest.approx(0.42)


def test_provenance_reports_partial_and_breakdown():
    prov = compute_composite({"steric_clashes": 1.0})
    assert prov.is_partial is True
    assert prov.n_scored_checks == 1
    assert prov.n_total_checks == 8
    assert set(prov.computed_but_unscored) == {"lennard_jones", "ramachandran"}
    assert set(prov.not_implemented) == {
        "bond_geometry",
        "peptide_planarity",
        "chirality",
        "rotamers",
        "disulfides",
    }


def test_unbuilt_check_cannot_be_weighted_into_score():
    """Feeding any non-scored check is a hard error — the core honesty guarantee."""
    with pytest.raises(ValueError):
        compute_composite({"steric_clashes": 1.0, "rotamers": 1.0})
    with pytest.raises(ValueError):
        compute_composite({"lennard_jones": 1.0})  # computed-but-unscored, not scored
    with pytest.raises(ValueError):
        compute_composite({})  # missing the scored check


def test_only_scored_checks_appear_in_contributions():
    prov = compute_composite({"steric_clashes": 0.9})
    contributing_names = {c.check for c in prov.contributing_checks}
    not_impl = {
        n for n, (s, _) in CHECK_REGISTRY.items() if s is CheckStatus.NOT_IMPLEMENTED
    }
    unscored = {
        n for n, (s, _) in CHECK_REGISTRY.items() if s is CheckStatus.COMPUTED_UNSCORED
    }
    assert contributing_names.isdisjoint(not_impl)
    assert contributing_names.isdisjoint(unscored)


def test_normalized_weights_sum_to_one():
    prov = compute_composite({"steric_clashes": 0.5})
    assert sum(c.normalized_weight for c in prov.contributing_checks) == pytest.approx(
        1.0
    )


def test_to_dict_is_json_serializable_and_complete():
    prov = compute_composite({"steric_clashes": 0.7})
    d = prov.to_dict()
    json.dumps(d)  # must not raise
    assert d["is_partial"] is True
    assert d["n_scored_checks"] == 1
    assert d["n_total_checks"] == 8
    assert {"lennard_jones", "ramachandran"} == set(d["computed_but_unscored"])
    assert len(d["not_implemented"]) == 5


def test_custom_config_weights_still_normalize():
    """A different clash weight cannot change a single-scored-check composite."""
    cfg = CompositeConfig(weight_clashes=0.99)
    prov = compute_composite({"steric_clashes": 0.3}, cfg)
    assert prov.score == pytest.approx(0.3)


# --- CLI integration: the audit report carries honest composite provenance ---


def test_cli_report_carries_partial_composite():
    runner = CliRunner()
    result = runner.invoke(
        app, ["validate", str(FIXTURES / "tri_ala.pdb"), "--json"]
    )
    assert result.exit_code == 0
    report = json.loads(result.stdout)

    # global_score still present and equals the clash subscore (composite==clashes)
    assert report["global_score"] == report["checks"]["steric_clashes"]["subscore"]

    comp = report["composite"]
    assert comp["is_partial"] is True
    assert comp["n_scored_checks"] == 1
    assert comp["n_total_checks"] == 8
    assert [c["check"] for c in comp["contributing_checks"]] == ["steric_clashes"]
    assert comp["score"] == report["global_score"]

    # honesty flags on the unscored checks
    assert report["checks"]["steric_clashes"]["scored"] is True
    assert report["checks"]["lennard_jones"]["scored"] is False
    assert report["checks"]["backbone_dihedrals"]["scored"] is False
