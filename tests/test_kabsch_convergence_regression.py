"""Regression test for the Kabsch convergent-validation dossier.

Locks the committed PfDHODH_vs_HsDHODH_kabsch_convergence.json against
silent drift from changes to the Kabsch superposition engine
(superposition.py), the cofactor extraction, the pocket-correspondence
selection logic, or the underlying parser. Modeled on
test_realigned_dossier_regression.py (PR-DET rule: every committed
result artifact gets a regression test that locks both numbers and the
finding).

This test locks THREE things:

1. The seed-quality numbers — FMN and ORO anchor RMSDs within
   abs_tol=1e-3.

2. The finding — three cluster pairings (HIS185 -> HIS56,
   PHE188 -> ALA59, ARG265 -> ARG136) must agree with v1 sequence
   pairings under BOTH the FMN anchor and the ORO orthogonal
   cross-check, and the headline "3 of 3 ... 3 of 3" must hold. If
   a future change shifts any pairing or breaks the cross-anchor
   agreement, the conserved-residue convergent verdict is no longer
   supported and this test flags it.

3. The per-residue FMN nearest-CA distances within abs_tol=1e-2 (the
   tight-regime claim the dossier rests on — drift here would mean
   the rigid-body assumption no longer holds at the same level).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_JSON = (
    REPO_ROOT
    / "benchmark"
    / "results"
    / "selectivity_maps"
    / "PfDHODH_vs_HsDHODH_kabsch_convergence.json"
)

FLOAT_ABS_TOL = 1e-3
DISTANCE_ABS_TOL = 1e-2

EXPECTED_ANCHORS = {
    "FMN": {
        "n_anchor_atoms": 31,
        "anchor_rmsd_expected": 0.2597,
        "n_pocket_target": 21,
        "n_pocket_ortholog": 19,
        "n_spatially_matched": 21,
        "n_unmatched_target": 0,
    },
    "ORO": {
        "n_anchor_atoms": 11,
        "anchor_rmsd_expected": 0.0762,
        "n_pocket_target": 21,
        "n_pocket_ortholog": 19,
        "n_spatially_matched": 20,
        "n_unmatched_target": 1,
    },
}

# (target_res_seq, target_res_name, ortholog_res_name, ortholog_res_seq,
#  FMN_nearest_distance_expected)
EXPECTED_CLUSTER = [
    (185, "HIS", "HIS", 56, 1.393),
    (188, "PHE", "ALA", 59, 1.764),
    (265, "ARG", "ARG", 136, 1.427),
]


def _rerun_in_process() -> dict:
    """Reproduce the convergent-validation in-process without touching disk."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from physics_auditor.causality.binding_site import extract_binding_site
    from physics_auditor.causality.selectivity_map import find_ligand_atoms_by_resname
    from physics_auditor.causality.superposition import superpose_and_correspond
    from physics_auditor.core.parser import parse_pdb
    from physics_auditor.utils.pdb_provenance import verify_pdb_source

    pf_pdb = REPO_ROOT / "benchmark" / "structures" / "experimental" / "4ORM.pdb"
    hs_pdb = REPO_ROOT / "benchmark" / "structures" / "experimental" / "1D3H.pdb"

    verify_pdb_source(pf_pdb, "PLASMODIUM FALCIPARUM")
    verify_pdb_source(hs_pdb, "HOMO SAPIENS")

    pf = parse_pdb(pf_pdb)
    hs = parse_pdb(hs_pdb)
    pf_lig_idx, _ = find_ligand_atoms_by_resname(pf, "2V6")
    hs_lig_idx, _ = find_ligand_atoms_by_resname(hs, "A26")
    pf_pocket = extract_binding_site(pf, pf.coords[pf_lig_idx], cutoff=5.0)
    hs_pocket = extract_binding_site(hs, hs.coords[hs_lig_idx], cutoff=5.0)

    out: dict[str, dict] = {}
    for anchor in ("FMN", "ORO"):
        sp = superpose_and_correspond(
            target_structure=pf,
            ortholog_structure=hs,
            target_pocket=pf_pocket,
            ortholog_pocket=hs_pocket,
            anchor_resname=anchor,
            match_cutoff=4.0,
            target_chain="A",
            ortholog_chain="A",
        )
        rid_list_t = list(pf.residues.keys())
        rid_list_o = list(hs.residues.keys())
        cluster_pairings: dict[int, dict] = {}
        for record in sp.match_details:
            t_rid = rid_list_t[record.target_engine_idx]
            t_res = pf.residues[t_rid]
            if t_res.res_seq not in (185, 188, 265):
                continue
            partner = None
            if record.ortholog_engine_idx is not None:
                o_rid = rid_list_o[record.ortholog_engine_idx]
                o_res = hs.residues[o_rid]
                partner = (o_res.res_name, o_res.res_seq)
            cluster_pairings[t_res.res_seq] = {
                "target_res_name": t_res.res_name,
                "partner": partner,
                "nearest_distance": record.nearest_distance,
                "accepted": record.accepted,
            }
        out[anchor] = {
            "n_anchor_atoms": sp.n_anchor_atoms,
            "anchor_rmsd": sp.anchor_rmsd,
            "n_pocket_target": sp.n_pocket_target,
            "n_pocket_ortholog": sp.n_pocket_ortholog,
            "n_spatially_matched": sp.n_spatially_matched,
            "n_unmatched_target": sp.n_unmatched_target,
            "cluster_pairings": cluster_pairings,
        }
    return out


def test_kabsch_dossier_anchor_rmsd_locked():
    """FMN and ORO anchor RMSDs and pocket-match counts must reproduce."""
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)

    fresh = _rerun_in_process()

    for anchor, expected in EXPECTED_ANCHORS.items():
        committed_anchor = committed["anchors"][anchor]
        fresh_anchor = fresh[anchor]

        for field in (
            "n_anchor_atoms",
            "n_pocket_target",
            "n_pocket_ortholog",
            "n_spatially_matched",
            "n_unmatched_target",
        ):
            assert committed_anchor[field] == expected[field], (
                f"{anchor}.{field}: committed={committed_anchor[field]} "
                f"!= expected={expected[field]}"
            )
            assert fresh_anchor[field] == expected[field], (
                f"{anchor}.{field}: fresh={fresh_anchor[field]} "
                f"!= expected={expected[field]}"
            )

        assert math.isclose(
            committed_anchor["anchor_rmsd"],
            expected["anchor_rmsd_expected"],
            abs_tol=FLOAT_ABS_TOL,
        ), (
            f"committed {anchor} anchor_rmsd={committed_anchor['anchor_rmsd']} "
            f"drifted from expected {expected['anchor_rmsd_expected']}"
        )
        assert math.isclose(
            fresh_anchor["anchor_rmsd"],
            expected["anchor_rmsd_expected"],
            abs_tol=FLOAT_ABS_TOL,
        ), (
            f"fresh {anchor} anchor_rmsd={fresh_anchor['anchor_rmsd']} "
            f"drifted from expected {expected['anchor_rmsd_expected']}"
        )
        assert math.isclose(
            committed_anchor["anchor_rmsd"],
            fresh_anchor["anchor_rmsd"],
            abs_tol=FLOAT_ABS_TOL,
        ), (
            f"{anchor} anchor_rmsd: committed vs fresh drift "
            f"(committed={committed_anchor['anchor_rmsd']}, "
            f"fresh={fresh_anchor['anchor_rmsd']})"
        )


def test_kabsch_cluster_pairings_locked():
    """All three cluster pairings must agree under both anchors AND v1 sequence."""
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)
    fresh = _rerun_in_process()

    per_res_committed = {
        row["target_res_seq"]: row
        for row in committed["convergence_verdict"]["per_residue"]
    }

    for target_res_seq, target_name, ortho_name, ortho_seq, _fmn_d in EXPECTED_CLUSTER:
        committed_row = per_res_committed[target_res_seq]
        assert committed_row["target_res_name"] == target_name
        assert committed_row["v1_sequence_partner"] == {
            "res_name": ortho_name,
            "res_seq": ortho_seq,
        }
        assert committed_row["v2_structural_partner_FMN"] == {
            "res_name": ortho_name,
            "res_seq": ortho_seq,
        }, (
            f"FMN structural partner for {target_name}{target_res_seq} drifted; "
            f"committed={committed_row['v2_structural_partner_FMN']}, "
            f"expected={{'res_name': {ortho_name!r}, 'res_seq': {ortho_seq}}}"
        )
        assert committed_row["v2_structural_partner_ORO"] == {
            "res_name": ortho_name,
            "res_seq": ortho_seq,
        }, (
            f"ORO structural partner for {target_name}{target_res_seq} drifted"
        )
        assert committed_row["agrees_v1_vs_FMN"] is True
        assert committed_row["agrees_v1_vs_ORO"] is True
        assert committed_row["agrees_FMN_vs_ORO"] is True

        fresh_anchor_FMN = fresh["FMN"]["cluster_pairings"][target_res_seq]
        fresh_anchor_ORO = fresh["ORO"]["cluster_pairings"][target_res_seq]
        assert fresh_anchor_FMN["partner"] == (ortho_name, ortho_seq), (
            f"fresh FMN partner for {target_name}{target_res_seq}: "
            f"{fresh_anchor_FMN['partner']} != ({ortho_name!r}, {ortho_seq})"
        )
        assert fresh_anchor_ORO["partner"] == (ortho_name, ortho_seq), (
            f"fresh ORO partner for {target_name}{target_res_seq}: "
            f"{fresh_anchor_ORO['partner']} != ({ortho_name!r}, {ortho_seq})"
        )


def test_kabsch_fmn_per_residue_distances_locked():
    """Per-residue FMN nearest-CA distances for the cluster must reproduce."""
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)
    fresh = _rerun_in_process()

    per_res_committed = {
        row["target_res_seq"]: row
        for row in committed["convergence_verdict"]["per_residue"]
    }

    for target_res_seq, target_name, _o_name, _o_seq, expected_fmn_d in EXPECTED_CLUSTER:
        committed_d = per_res_committed[target_res_seq]["v2_FMN_nearest_distance"]
        fresh_d = fresh["FMN"]["cluster_pairings"][target_res_seq]["nearest_distance"]

        assert math.isclose(committed_d, expected_fmn_d, abs_tol=DISTANCE_ABS_TOL), (
            f"committed FMN nearest distance for {target_name}{target_res_seq}: "
            f"{committed_d} drifted from expected {expected_fmn_d}"
        )
        assert math.isclose(fresh_d, expected_fmn_d, abs_tol=DISTANCE_ABS_TOL), (
            f"fresh FMN nearest distance for {target_name}{target_res_seq}: "
            f"{fresh_d} drifted from expected {expected_fmn_d}. If intentional, "
            f"regenerate the dossier via "
            f"`python benchmark/run_kabsch_convergent_validation.py` and commit "
            f"alongside the engine change."
        )
        assert math.isclose(committed_d, fresh_d, abs_tol=DISTANCE_ABS_TOL), (
            f"FMN nearest distance for {target_name}{target_res_seq}: "
            f"committed={committed_d} vs fresh={fresh_d}"
        )


def test_kabsch_headline_agreement_locked():
    """Headline must be '3 of 3' under both FMN and ORO anchors."""
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)

    verdict = committed["convergence_verdict"]
    assert verdict["primary_anchor"] == "FMN"
    assert verdict["cross_check_anchor"] == "ORO"

    headline = verdict["headline_agreement"]
    assert "3 of 3" in headline, (
        f"headline_agreement no longer reports 3-of-3: {headline!r}"
    )
    assert "FMN-anchored" in headline
    assert "ORO-anchored" in headline


def test_kabsch_pocket_match_counts_locked():
    """FMN must match 21/21 pocket residues; ORO 20/21."""
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)

    fmn = committed["anchors"]["FMN"]
    oro = committed["anchors"]["ORO"]

    assert fmn["n_pocket_target"] == 21
    assert fmn["n_spatially_matched"] == 21
    assert fmn["n_unmatched_target"] == 0

    assert oro["n_pocket_target"] == 21
    assert oro["n_spatially_matched"] == 20
    assert oro["n_unmatched_target"] == 1
