"""Regression test for the realigned PfDHODH-vs-HsDHODH dossier.

Locks the committed PfDHODH_vs_HsDHODH_selectivity_REALIGNED.json against
silent drift from changes to the correspondence engine, the selectivity
machinery, the pocket extractor, or the LJ kernel. Modeled on
test_benchmark_clashscore_regression.py (PR-DET pattern: every committed
result artifact gets a regression test).

This test locks TWO things:

1. The numbers — every numeric field on the dossier (pocket sizes,
   per-residue energies, deltas, identity, alignment provenance) must
   reproduce within abs_tol=1e-3.

2. The finding — the literature selectivity cluster (HIS185, PHE188,
   ARG265 of PfDHODH) must appear in top_n_target_selective with their
   v1-realigned ortholog partners (HIS56, ALA59, ARG136 of HsDHODH).
   If a future correspondence-engine change shifts those pairings, the
   conserved-residue interpretation in REALIGNED_FINDINGS.md is no
   longer supported and this test will flag it.

Also asserts the alignment was not refused and the global sequence
identity lies in the catalytic-domain ortholog band (0.35-0.50).
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
    / "PfDHODH_vs_HsDHODH_selectivity_REALIGNED.json"
)

FLOAT_ABS_TOL = 1e-3

PAIR_FLOAT_FIELDS = ("energy_target_kcal", "energy_ortholog_kcal", "delta_kcal")
PAIR_INT_FIELDS = ("res_seq_target", "res_seq_ortholog")
PAIR_STRING_FIELDS = ("res_name_target", "res_name_ortholog")

# v1 realigned partners for the literature selectivity cluster.
# Format: PfDHODH res_seq -> (target_res_name, ortholog_res_name, ortholog_res_seq).
LITERATURE_CLUSTER_REALIGNED = {
    185: ("HIS", "HIS", 56),
    188: ("PHE", "ALA", 59),
    265: ("ARG", "ARG", 136),
}


def _rerun_in_process() -> dict:
    """Reproduce the dossier in-process without touching disk."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from physics_auditor.causality.binding_site import extract_binding_site
    from physics_auditor.causality.correspondence import build_pocket_alignment
    from physics_auditor.causality.selectivity_map import (
        compute_selectivity_map,
        find_ligand_atoms_by_resname,
        selectivity_map_to_dict,
    )
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

    palign = build_pocket_alignment(
        target_structure=pf,
        ortholog_structure=hs,
        target_pocket=pf_pocket,
        ortholog_pocket=hs_pocket,
        min_global_identity=0.25,
    )
    assert not palign.refused, (
        f"Realignment unexpectedly refused: {palign.refusal_reason}"
    )

    smap = compute_selectivity_map(
        target_structure=pf,
        target_ligand_resname="2V6",
        ortholog_structure=hs,
        ortholog_ligand_resname="A26",
        pocket_cutoff=5.0,
        alignment=palign.correspondence,
    )

    alignment_provenance = {
        "method": palign.method,
        "global_sequence_identity": round(palign.global_sequence_identity, 4),
        "min_identity_threshold": palign.min_identity_threshold,
        "n_pocket_target": palign.n_pocket_target,
        "n_pocket_ortholog": palign.n_pocket_ortholog,
        "n_confidently_paired": palign.n_confidently_paired,
        "n_unpaired_target": palign.n_unpaired_target,
        "refused": palign.refused,
        "refusal_reason": palign.refusal_reason,
    }
    return {
        "alignment_provenance": alignment_provenance,
        **selectivity_map_to_dict(smap, top_n=15),
    }


def _assert_pair_lists_match(committed: list[dict], fresh: list[dict], label: str) -> None:
    assert len(committed) == len(fresh), (
        f"{label}: committed has {len(committed)} entries, fresh has {len(fresh)}"
    )
    for i, (c, f) in enumerate(zip(committed, fresh)):
        for field in PAIR_STRING_FIELDS:
            assert c[field] == f[field], (
                f"{label}[{i}].{field}: committed={c[field]!r} != fresh={f[field]!r}"
            )
        for field in PAIR_INT_FIELDS:
            assert c[field] == f[field], (
                f"{label}[{i}].{field}: committed={c[field]} != fresh={f[field]}"
            )
        for field in PAIR_FLOAT_FIELDS:
            assert math.isclose(c[field], f[field], abs_tol=FLOAT_ABS_TOL), (
                f"{label}[{i}].{field}: committed={c[field]} fresh={f[field]} "
                f"(abs diff {abs(c[field] - f[field]):.6f} > {FLOAT_ABS_TOL}). "
                "If intentional, regenerate the dossier via "
                "`python benchmark/run_pfdhodh_realigned_comparison.py` "
                "and commit it together with the engine change."
            )


def test_realigned_dossier_reproduces_byte_compatible():
    """Rerunning the realignment must reproduce the committed dossier."""
    assert COMMITTED_JSON.exists(), f"Committed dossier missing at {COMMITTED_JSON}"
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)

    fresh = _rerun_in_process()

    c_prov = committed["alignment_provenance"]
    f_prov = fresh["alignment_provenance"]

    for field in ("method", "refusal_reason"):
        assert c_prov[field] == f_prov[field], (
            f"alignment_provenance.{field}: committed={c_prov[field]!r} "
            f"!= fresh={f_prov[field]!r}"
        )
    for field in (
        "n_pocket_target",
        "n_pocket_ortholog",
        "n_confidently_paired",
        "n_unpaired_target",
    ):
        assert c_prov[field] == f_prov[field], (
            f"alignment_provenance.{field}: committed={c_prov[field]} "
            f"!= fresh={f_prov[field]}"
        )
    assert c_prov["refused"] == f_prov["refused"]
    for field in ("global_sequence_identity", "min_identity_threshold"):
        assert math.isclose(c_prov[field], f_prov[field], abs_tol=FLOAT_ABS_TOL), (
            f"alignment_provenance.{field}: committed={c_prov[field]} "
            f"!= fresh={f_prov[field]}"
        )

    for field in (
        "target_name",
        "ortholog_name",
        "target_ligand_name",
        "ortholog_ligand_name",
    ):
        assert committed[field] == fresh[field], (
            f"{field}: committed={committed[field]!r} != fresh={fresh[field]!r}"
        )
    assert committed["n_aligned_pocket_residues"] == fresh["n_aligned_pocket_residues"]
    for field in (
        "total_target_interaction_kcal",
        "total_ortholog_interaction_kcal",
        "pocket_target_interaction_kcal",
        "pocket_ortholog_interaction_kcal",
        "pocket_delta_kcal",
    ):
        assert math.isclose(committed[field], fresh[field], abs_tol=FLOAT_ABS_TOL), (
            f"{field}: committed={committed[field]} fresh={fresh[field]} "
            f"(abs diff {abs(committed[field] - fresh[field]):.6f})"
        )

    _assert_pair_lists_match(
        committed["top_n_target_selective"],
        fresh["top_n_target_selective"],
        "top_n_target_selective",
    )
    _assert_pair_lists_match(
        committed["top_n_ortholog_selective"],
        fresh["top_n_ortholog_selective"],
        "top_n_ortholog_selective",
    )
    _assert_pair_lists_match(
        committed["residues"], fresh["residues"], "residues"
    )


def test_realigned_dossier_alignment_provenance_sane():
    """Alignment must not be refused and identity must sit in the catalytic-ortholog band."""
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)
    prov = committed["alignment_provenance"]

    assert prov["refused"] is False, (
        f"Realigned dossier marked refused: {prov['refusal_reason']!r}"
    )
    assert prov["method"] == "global_sequence_nw"
    assert 0.35 < prov["global_sequence_identity"] < 0.50, (
        f"global_sequence_identity={prov['global_sequence_identity']} outside "
        f"plausible catalytic-domain ortholog band (0.35-0.50). PfDHODH/HsDHODH "
        f"should land near 0.41."
    )
    assert prov["n_confidently_paired"] == prov["n_pocket_target"], (
        "Every target-pocket residue should pair to a homologous counterpart."
    )
    assert prov["n_unpaired_target"] == 0


def test_realigned_literature_cluster_pairings_locked():
    """The HIS185/PHE188/ARG265 cluster must pair to (HIS56, ALA59, ARG136)."""
    with COMMITTED_JSON.open() as f:
        committed = json.load(f)
    top_target = committed["top_n_target_selective"]

    found: dict[int, dict] = {}
    for row in top_target:
        if row["res_seq_target"] in LITERATURE_CLUSTER_REALIGNED:
            found[row["res_seq_target"]] = row

    missing = sorted(set(LITERATURE_CLUSTER_REALIGNED) - set(found))
    assert not missing, (
        f"Literature-cluster residues missing from top_n_target_selective: "
        f"{missing}. Realignment may have demoted them below the top-15 cutoff; "
        f"if intentional, REALIGNED_FINDINGS.md must be revised."
    )

    for res_seq, (
        expected_target_name,
        expected_ortholog_name,
        expected_ortholog_seq,
    ) in LITERATURE_CLUSTER_REALIGNED.items():
        row = found[res_seq]
        assert row["res_name_target"] == expected_target_name, (
            f"Target residue {res_seq} expected {expected_target_name}, "
            f"got {row['res_name_target']}"
        )
        assert row["res_name_ortholog"] == expected_ortholog_name, (
            f"Target residue {res_seq} ({expected_target_name}) expected "
            f"ortholog partner {expected_ortholog_name}, got "
            f"{row['res_name_ortholog']}. The conserved-pairing finding in "
            f"REALIGNED_FINDINGS.md depends on this — if the engine now "
            f"reports a different partner, the finding must be reinterpreted."
        )
        assert row["res_seq_ortholog"] == expected_ortholog_seq, (
            f"Target residue {res_seq} expected ortholog res_seq "
            f"{expected_ortholog_seq}, got {row['res_seq_ortholog']}"
        )
