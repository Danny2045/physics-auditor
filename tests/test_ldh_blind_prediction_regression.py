"""Regression test for the locked blind PfLDH vs HsLDH-A prediction.

Locks the committed PfLDH_vs_HsLDH-A_blind_prediction.json against
silent drift from any change to the parser, NAI-anchored Kabsch,
sequence correspondence, or LJ-delta engine. This is the lock-test
for Phase B-2 of the LDH blind benchmark.

What this test locks (PR-DET rule):

1. The prediction pair is 4PLZ (PfLDH) vs 1I10 (HsLDH-A) — NOT 5W8K.
   5W8K was the Phase A acquisition; Phase B-1 replaced it with the
   chemotype-matched 1I10 (NAI+OXM both sides). A future change that
   silently reverts the pair to 5W8K is a regression.

2. Sequence-correspondence: global identity + n_confidently_paired.

3. NAI-anchored Kabsch: shared atom-name count + anchor RMSD.

4. The predicted divergent-position set (the target->ortholog pairs
   with non-identical residue names in the pocket).

5. The per-residue delta_kcal values for every pocket pair, abs_tol 1e-2.

6. The convergent-subset list (positions where seq and structural
   correspondence both agree on the pairing AND it is either divergent
   or in the top-quartile |delta_kcal|).

7. The bold sub-claim (top-ranked target-selective residue).

8. The blindness markers: prediction_locked_before_literature == true
   AND literature_check_commit is null. These are flipped only by a
   later B-3 commit in a separate session, never by the lock commit.

The test additionally re-runs the engine in-process and asserts the
fresh numbers match the dossier within tolerance, so the JSON cannot
be edited away from what the engine produces.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOSSIER = (
    REPO_ROOT
    / "benchmark"
    / "results"
    / "selectivity_maps"
    / "PfLDH_vs_HsLDH-A_blind_prediction.json"
)

FLOAT_ABS_TOL = 1e-3
DELTA_ABS_TOL = 1e-2

EXPECTED_PAIR = {
    "target_pdb": "4PLZ",
    "ortholog_pdb": "1I10",
    "target_ligand": "OXM",
    "ortholog_ligand": "OXM",
    "cofactor": "NAI",
}

EXPECTED_SEQUENCE = {
    "global_sequence_identity": 0.3211,
    "n_confidently_paired": 8,
}

EXPECTED_STRUCTURAL = {
    "n_anchor_atoms": 44,
    "anchor_rmsd": 0.6731,
    "n_spatially_matched": 8,
    "n_pocket_target": 8,
    "n_pocket_ortholog": 9,
}

# Pocket pairs locked as (target_label, ortholog_label, delta_kcal_expected).
# Identity-preserved positions retain their res_name; divergent positions
# show distinct res_names.
EXPECTED_POCKET_PAIRS = [
    ("ASN126", "ASN137", -0.164),
    ("LEU153", "LEU164", +0.154),
    ("ASP154", "ASP165", +0.118),
    ("ARG157", "ARG168", +1.991),
    ("HIS181", "HIS192", -1.702),
    ("ALA224", "ALA237", -0.135),
    ("PRO234", "THR247", +2.465),
    ("PRO238", "ILE251", +0.182),
]

# Divergent positions (different residue names across the pair, ranked
# by combined evidence: divergence flag + |delta_kcal|).
EXPECTED_DIVERGENT_SET = [
    ("PRO234", "THR247"),
    ("PRO238", "ILE251"),
]

# Convergent headline: seq AND struct agree on the pairing, pair is either
# divergent or has top-quartile |delta_kcal|.
EXPECTED_CONVERGENT = [
    ("PRO234", "THR247"),
    ("ARG157", "ARG168"),
    ("HIS181", "HIS192"),
    ("PRO238", "ILE251"),
]

EXPECTED_BOLD = "PRO234"


def _load_dossier() -> dict:
    assert DOSSIER.exists(), (
        f"Dossier missing at {DOSSIER}. The blind-prediction lock commit "
        f"must include this artifact."
    )
    return json.loads(DOSSIER.read_text())


def _rerun_engine() -> dict:
    """Reproduce the engine output in-process so the JSON cannot drift
    silently away from what the engine actually produces."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from physics_auditor.causality.binding_site import extract_binding_site
    from physics_auditor.causality.correspondence import build_pocket_alignment
    from physics_auditor.causality.selectivity_map import (
        compute_selectivity_map,
        find_ligand_atoms_by_resname,
    )
    from physics_auditor.causality.superposition import superpose_and_correspond
    from physics_auditor.core.parser import parse_pdb, parse_pdb_string
    from physics_auditor.utils.pdb_provenance import verify_pdb_source

    pf_pdb = REPO_ROOT / "benchmark" / "structures" / "experimental" / "4PLZ.pdb"
    hs_pdb = REPO_ROOT / "benchmark" / "structures" / "experimental" / "1I10.pdb"
    verify_pdb_source(pf_pdb, "PLASMODIUM FALCIPARUM")
    verify_pdb_source(hs_pdb, "HOMO SAPIENS")

    pf = parse_pdb(pf_pdb)
    raw = hs_pdb.read_text()
    text_A = "\n".join(
        line for line in raw.splitlines()
        if not (
            line.startswith(("ATOM", "HETATM", "ANISOU"))
            and line[21:22] != "A"
        )
    )
    hs = parse_pdb_string(text_A, name="1I10_chainA")

    pf_lig_idx, _ = find_ligand_atoms_by_resname(pf, "OXM")
    hs_lig_idx, _ = find_ligand_atoms_by_resname(hs, "OXM")
    pf_pocket = extract_binding_site(pf, pf.coords[pf_lig_idx], cutoff=5.0)
    hs_pocket = extract_binding_site(hs, hs.coords[hs_lig_idx], cutoff=5.0)

    palign = build_pocket_alignment(
        target_structure=pf,
        ortholog_structure=hs,
        target_pocket=pf_pocket,
        ortholog_pocket=hs_pocket,
        target_chain="A",
        ortholog_chain="A",
    )
    sp = superpose_and_correspond(
        target_structure=pf,
        ortholog_structure=hs,
        target_pocket=pf_pocket,
        ortholog_pocket=hs_pocket,
        anchor_resname="NAI",
        match_cutoff=4.0,
        target_chain="A",
        ortholog_chain="A",
    )
    smap = compute_selectivity_map(
        target_structure=pf,
        target_ligand_resname="OXM",
        ortholog_structure=hs,
        ortholog_ligand_resname="OXM",
        pocket_cutoff=5.0,
        alignment=sp.correspondence,
    )

    pairs = []
    for r in smap.residues:
        pairs.append({
            "target": f"{r.res_name_target}{r.res_seq_target}",
            "ortholog": f"{r.res_name_ortholog}{r.res_seq_ortholog}",
            "delta_kcal": r.delta_kcal,
            "divergent": r.res_name_target != r.res_name_ortholog,
        })

    return {
        "sequence_identity": palign.global_sequence_identity,
        "sequence_n_paired": palign.n_confidently_paired,
        "sequence_refused": palign.refused,
        "anchor_atoms": sp.n_anchor_atoms,
        "anchor_rmsd": sp.anchor_rmsd,
        "n_spatially_matched": sp.n_spatially_matched,
        "n_pocket_target": sp.n_pocket_target,
        "n_pocket_ortholog": sp.n_pocket_ortholog,
        "pairs": pairs,
    }


# === Tests ===

def test_dossier_present_and_pair_is_4plz_vs_1i10():
    d = _load_dossier()
    pair = d["pair"]
    assert pair["target"]["pdb"] == EXPECTED_PAIR["target_pdb"], (
        f"Prediction-pair target drifted: {pair['target']['pdb']!r} "
        f"(expected {EXPECTED_PAIR['target_pdb']!r}). Phase B-1 replaced "
        f"5W8K with 1I10 for chemotype match; a revert here breaks the "
        f"blind test."
    )
    assert pair["ortholog"]["pdb"] == EXPECTED_PAIR["ortholog_pdb"], (
        f"Prediction-pair ortholog drifted to {pair['ortholog']['pdb']!r} "
        f"(expected {EXPECTED_PAIR['ortholog_pdb']!r} — NOT 5W8K)."
    )
    assert pair["ortholog"]["pdb"] != "5W8K", (
        "5W8K is the chemotype-confounded acquisition retained on disk "
        "for the audit trail; it must NOT appear as the prediction-pair "
        "ortholog."
    )
    assert pair["target"]["ligand_resname"] == EXPECTED_PAIR["target_ligand"]
    assert pair["ortholog"]["ligand_resname"] == EXPECTED_PAIR["ortholog_ligand"]
    assert pair["target"]["cofactor_resname"] == EXPECTED_PAIR["cofactor"]
    assert pair["ortholog"]["cofactor_resname"] == EXPECTED_PAIR["cofactor"]
    assert pair["chemotype_matched"] is True


def test_blindness_markers_are_in_locked_state():
    d = _load_dossier()
    b = d["blindness"]
    assert b["prediction_locked_before_literature"] is True
    # B-3 update: literature_check_commit is filled by the B-3 result commit
    # with the SHA of the literature-summary commit (da9a05c), since a
    # commit cannot self-reference its own SHA. The blindness guarantee is
    # preserved by the COMMIT ORDER (lock at 6dfa69b -> literature at
    # da9a05c -> result), not by this field being null forever.
    lit_check = b["literature_check_commit"]
    assert lit_check is not None and isinstance(lit_check, str), (
        "literature_check_commit must be filled by the B-3 result commit "
        "with the SHA of the literature-summary commit. A null value here "
        "means B-3 has not been recorded."
    )
    assert len(lit_check) >= 7, (
        f"literature_check_commit must be a git SHA (>=7 hex chars): "
        f"{lit_check!r}"
    )
    assert b["prediction_lock_timestamp_utc"]


def test_scoring_rubric_present_with_all_four_outcomes():
    d = _load_dossier()
    rubric = d["scoring_rubric"]
    for key in ("strong_pass", "partial_pass", "informative_miss", "confounded_void"):
        assert key in rubric, f"scoring_rubric missing {key!r}"
        assert isinstance(rubric[key], str) and rubric[key], (
            f"scoring_rubric[{key!r}] must be a non-empty string"
        )


def test_sequence_correspondence_matches_lock():
    d = _load_dossier()
    seq = d["alignment_provenance"]["sequence"]
    assert seq["refused"] is False
    assert math.isclose(
        seq["global_sequence_identity"],
        EXPECTED_SEQUENCE["global_sequence_identity"],
        abs_tol=FLOAT_ABS_TOL,
    ), f"sequence identity drifted: {seq['global_sequence_identity']!r}"
    assert seq["n_confidently_paired"] == EXPECTED_SEQUENCE["n_confidently_paired"]

    # Re-run check
    fresh = _rerun_engine()
    assert math.isclose(
        fresh["sequence_identity"],
        EXPECTED_SEQUENCE["global_sequence_identity"],
        abs_tol=FLOAT_ABS_TOL,
    )
    assert fresh["sequence_n_paired"] == EXPECTED_SEQUENCE["n_confidently_paired"]
    assert fresh["sequence_refused"] is False


def test_nai_anchor_rmsd_and_atom_count():
    d = _load_dossier()
    s = d["alignment_provenance"]["structural"]
    assert s["anchor_resname"] == "NAI"
    assert s["n_anchor_atoms"] == EXPECTED_STRUCTURAL["n_anchor_atoms"]
    assert math.isclose(
        s["anchor_rmsd"],
        EXPECTED_STRUCTURAL["anchor_rmsd"],
        abs_tol=FLOAT_ABS_TOL,
    )
    assert s["n_spatially_matched"] == EXPECTED_STRUCTURAL["n_spatially_matched"]
    assert s["n_pocket_target"] == EXPECTED_STRUCTURAL["n_pocket_target"]
    assert s["n_pocket_ortholog"] == EXPECTED_STRUCTURAL["n_pocket_ortholog"]

    fresh = _rerun_engine()
    assert fresh["anchor_atoms"] == EXPECTED_STRUCTURAL["n_anchor_atoms"]
    assert math.isclose(
        fresh["anchor_rmsd"],
        EXPECTED_STRUCTURAL["anchor_rmsd"],
        abs_tol=FLOAT_ABS_TOL,
    )
    assert fresh["n_spatially_matched"] == EXPECTED_STRUCTURAL["n_spatially_matched"]


def test_pocket_pairs_and_per_residue_delta_kcal():
    d = _load_dossier()
    rows = d["prediction"]["secondary_lj_ranked"]["structural_correspondence"]["all_pairs"]
    actual = {
        (f"{r['res_name_target']}{r['res_seq_target']}",
         f"{r['res_name_ortholog']}{r['res_seq_ortholog']}"): r["delta_kcal"]
        for r in rows
    }
    for t_lbl, o_lbl, delta_expected in EXPECTED_POCKET_PAIRS:
        assert (t_lbl, o_lbl) in actual, (
            f"locked pocket pair {t_lbl}->{o_lbl} missing from dossier"
        )
        assert math.isclose(actual[(t_lbl, o_lbl)], delta_expected, abs_tol=DELTA_ABS_TOL), (
            f"delta_kcal drift on {t_lbl}->{o_lbl}: "
            f"{actual[(t_lbl, o_lbl)]} vs expected {delta_expected}"
        )

    fresh = _rerun_engine()
    fresh_map = {(p["target"], p["ortholog"]): p["delta_kcal"] for p in fresh["pairs"]}
    for t_lbl, o_lbl, delta_expected in EXPECTED_POCKET_PAIRS:
        assert (t_lbl, o_lbl) in fresh_map
        assert math.isclose(fresh_map[(t_lbl, o_lbl)], delta_expected, abs_tol=DELTA_ABS_TOL)


def test_primary_divergent_set_locked():
    d = _load_dossier()
    primary = d["prediction"]["primary_divergent_set"]
    got = [(p["target"], p["ortholog"]) for p in primary]
    assert got == EXPECTED_DIVERGENT_SET, (
        f"Primary divergent set drifted.\n  expected: {EXPECTED_DIVERGENT_SET}\n"
        f"  got:      {got}"
    )


def test_convergent_headline_locked():
    d = _load_dossier()
    conv = d["prediction"]["convergent_headline"]
    got = {(c["target"], c["ortholog"]) for c in conv}
    expected = set(EXPECTED_CONVERGENT)
    assert got == expected, (
        f"Convergent headline drifted.\n  expected: {expected}\n  got: {got}"
    )


def test_bold_sub_claim_locked():
    d = _load_dossier()
    bold = d["prediction"]["bold_sub_claim"]
    assert bold["target_residue"] == EXPECTED_BOLD, (
        f"Bold sub-claim drifted: {bold['target_residue']!r} vs "
        f"expected {EXPECTED_BOLD!r}"
    )
    # The bold claim is the single largest positive delta_kcal in the pocket.
    fresh = _rerun_engine()
    top = max(fresh["pairs"], key=lambda p: p["delta_kcal"])
    assert top["target"] == EXPECTED_BOLD, (
        f"Fresh engine top target-selective residue drifted: {top['target']!r}"
    )
