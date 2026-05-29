"""Tests for the library-level audit() function and its readiness gate.

The audit() function is the bridge: typed StructureAuditRequest in,
typed StructureAuditResult out, with the three-state readiness gate enforced
at the front door.

Key claims tested here:
  - A bare PDB audit (no ligand) still runs the structure-level gate.
  - Ready structures get a real composite/global_score/recommendation.
  - Unready structures get composite/global_score/recommendation = None
    AND still get the raw per-check results populated.
  - The JSON serialization (to_dict) is byte-stable for ready fixtures and
    matches the CLI's prior layout plus the additive readiness block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from physics_auditor.audit import (
    StructureAuditRequest,
    StructureAuditResult,
    audit,
)
from physics_auditor.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
EXPERIMENTAL = (
    Path(__file__).parent.parent / "benchmark" / "structures" / "experimental"
)


def test_audit_returns_typed_result_for_ready_fixture():
    """tri_ala has no SEQRES; the SEQRES-less fallback walks the resolved
    range, finds no gaps, and the structure gate reports COMPLETE."""
    request = StructureAuditRequest(pdb_path=FIXTURES / "tri_ala.pdb")
    result = audit(request)

    assert isinstance(result, StructureAuditResult)
    assert result.readiness.ready is True
    assert result.readiness.structure_level == "COMPLETE"
    assert result.readiness.pocket_level == "NOT_ASSESSED"
    assert result.composite is not None
    assert result.global_score is not None
    assert result.recommendation in {"accept", "short_md", "discard"}
    # Raw check facts always populated.
    assert result.checks["steric_clashes"]["scored"] is True
    assert "clashscore" in result.checks["steric_clashes"]


def test_audit_withholds_composite_on_unready_4plz():
    """The W107f failure mode: 4PLZ has a disordered substrate-specificity
    loop. The always-on structure-level gate must flip the audit to unready
    WITHOUT a ligand supplied; composite/global_score/recommendation are
    withheld; raw per-check facts are still computed.
    """
    request = StructureAuditRequest(pdb_path=EXPERIMENTAL / "4PLZ.pdb")
    result = audit(request)

    assert result.readiness.ready is False
    assert result.readiness.structure_level == "INCOMPLETE"
    assert result.readiness.pocket_level == "NOT_ASSESSED"
    # The 12 internal-gap residues (82-93) — the disordered loop.
    gap_seqs = sorted(m.res_seq for m in result.readiness.structure_internal_gaps)
    assert gap_seqs == list(range(82, 94))
    # Withheld confident-verdict artifacts.
    assert result.composite is None
    assert result.global_score is None
    assert result.recommendation is None
    # Raw facts still produced.
    assert result.checks["steric_clashes"]["scored"] is True
    assert result.checks["steric_clashes"]["n_clashes"] > 0
    assert result.checks["lennard_jones"]["n_hot_pairs"] >= 0


def test_audit_ready_when_ligand_supplied_and_pocket_complete():
    """If a ligand is supplied and the pocket is fully resolved, the audit
    reports both structure_level AND pocket_level as COMPLETE."""
    # 1UBQ has no gaps; supply HOH as a stand-in ligand resname so the
    # pocket gate runs (it's just exercising the path).
    request = StructureAuditRequest(
        pdb_path=EXPERIMENTAL / "1UBQ.pdb",
        ligand_resname="HOH",
        pocket_cutoff=5.0,
    )
    result = audit(request)

    assert result.readiness.structure_level == "COMPLETE"
    assert result.readiness.pocket_level == "COMPLETE"
    assert result.readiness.ready is True
    assert result.global_score is not None


def test_audit_unready_when_pocket_incomplete_even_if_structure_were_complete():
    """If the ligand-gated pocket check returns INCOMPLETE, the audit is
    unready even if structure_level were COMPLETE (logically: composite
    requires every applicable check to pass).

    4PLZ has its native ligand OXM bound at the active site of the
    substrate-specificity loop. With OXM supplied, pocket_completeness_gate
    flags the disordered loop residues as in-pocket; this redundantly
    confirms unreadiness on top of the structure-level signal.
    """
    request = StructureAuditRequest(
        pdb_path=EXPERIMENTAL / "4PLZ.pdb",
        ligand_resname="OXM",
        pocket_cutoff=5.0,
    )
    result = audit(request)

    assert result.readiness.ready is False
    # Both levels INCOMPLETE on 4PLZ + OXM.
    assert result.readiness.structure_level == "INCOMPLETE"
    assert result.readiness.pocket_level == "INCOMPLETE"
    assert len(result.readiness.pocket_missing) > 0
    assert result.composite is None


def test_audit_to_dict_layout_matches_expected_schema():
    """The to_dict() report has exactly the top-level keys the CLI emits."""
    request = StructureAuditRequest(pdb_path=FIXTURES / "tri_ala.pdb")
    d = audit(request).to_dict()

    assert set(d) == {
        "file",
        "global_score",
        "recommendation",
        "readiness",
        "composite",
        "checks",
        "metadata",
    }
    # readiness sub-schema
    assert set(d["readiness"]) == {
        "ready",
        "structure_level",
        "pocket_level",
        "structure_internal_gaps",
        "structure_unresolved_termini",
        "pocket_missing",
        "reason",
    }


def test_audit_to_dict_is_json_serializable():
    """The result must round-trip through JSON without losing structure."""
    request = StructureAuditRequest(pdb_path=FIXTURES / "tri_ala.pdb")
    d = audit(request).to_dict()
    parsed = json.loads(json.dumps(d))
    assert parsed["readiness"]["ready"] is True
    assert parsed["global_score"] == d["global_score"]


# --- CLI integration ---


def test_cli_unready_4plz_returns_null_composite_and_readiness_block():
    """The CLI report for 4PLZ (no ligand) must show composite=null,
    global_score=null, recommendation=null, and readiness.ready=False.
    """
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["validate", str(EXPERIMENTAL / "4PLZ.pdb"), "--json"],
    )
    assert result.exit_code == 0, result.stdout
    report = json.loads(result.stdout)

    assert report["composite"] is None
    assert report["global_score"] is None
    assert report["recommendation"] is None
    assert report["readiness"]["ready"] is False
    assert report["readiness"]["structure_level"] == "INCOMPLETE"
    assert report["readiness"]["pocket_level"] == "NOT_ASSESSED"

    gap_seqs = sorted(
        g["res_seq"] for g in report["readiness"]["structure_internal_gaps"]
    )
    assert gap_seqs == list(range(82, 94))
    # Raw check facts still present.
    assert report["checks"]["steric_clashes"]["n_clashes"] > 0


def test_cli_ready_tri_ala_preserves_global_score_and_adds_readiness_only():
    """The CLI report for tri_ala (ready) must keep global_score == clash
    subscore (the Step-0 honest-partial-composite invariant) AND add the
    new readiness block with ready=True.
    """
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["validate", str(FIXTURES / "tri_ala.pdb"), "--json"],
    )
    assert result.exit_code == 0
    report = json.loads(result.stdout)

    # Step-0 invariant still holds.
    assert report["global_score"] == report["checks"]["steric_clashes"]["subscore"]
    assert report["composite"]["score"] == report["global_score"]

    # Step-1 additive readiness block.
    assert report["readiness"]["ready"] is True
    assert report["readiness"]["structure_level"] == "COMPLETE"
    assert report["readiness"]["pocket_level"] == "NOT_ASSESSED"


@pytest.mark.parametrize("fixture", ["tri_ala", "clashing"])
def test_cli_ready_fixtures_keep_partial_composite_honesty(fixture):
    """The Step-0 partial-composite provenance must survive the audit refactor:
    each ready fixture still carries is_partial=True, 1 of 8 scored, and
    the same contributing-checks list.
    """
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["validate", str(FIXTURES / f"{fixture}.pdb"), "--json"],
    )
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    comp = report["composite"]
    assert comp is not None
    assert comp["is_partial"] is True
    assert comp["n_scored_checks"] == 1
    assert comp["n_total_checks"] == 8
    assert [c["check"] for c in comp["contributing_checks"]] == ["steric_clashes"]
