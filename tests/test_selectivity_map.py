"""Tests for the selectivity-attribution module.

Two layers of testing:

1. Synthetic ground truth — two toy "proteins" each carrying a ligand,
   where the only difference between them is a single residue's
   ligand contact. The selectivity map must attribute the gap to that
   residue.

2. Real-data integration on the 1D3H / 1MVS pair with their bound
   inhibitors A26 and DTM. Verifies the module runs end-to-end on
   actual co-crystal data, with sensible energy magnitudes and
   non-empty pocket residue lists.

   NB: 1D3H is HsDHODH (catalytic construct of Q02127) and 1MVS is
   HsDHFR — two different human enzymes. The integration test asserts
   on per-residue *attribution* mechanics only; it does NOT assert
   parasite-vs-human selectivity. See SELECTIVITY_FINDINGS.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_auditor.causality.selectivity_map import (
    compute_selectivity_map,
    find_ligand_atoms_by_resname,
    selectivity_map_to_dict,
)
from physics_auditor.core.parser import parse_pdb, parse_pdb_string

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Synthetic protein-with-ligand structures.
#
# Each "protein" is two alanines plus one HETATM "ligand" atom labelled LIG.
# Target structure: ligand is 3.5 A from residue 1's CB (favorable LJ
# attractive well — both are carbon, so sigma ≈ 3.4 A and we're near the
# minimum).
# Ortholog structure: same backbone, but residue 1's CB is moved so it is
# 5.5 A from the ligand (LJ contact is now negligible).
# Expected: per-residue ligand interaction is much larger (more negative)
# in target than in ortholog, and the selectivity map attributes the gap
# to residue index 0.
# ---------------------------------------------------------------------------

_TARGET_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   1.500   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.500   1.500   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.000   2.600   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       0.000   2.200   1.400  1.00  0.00           C
ATOM      6  N   ALA A   2       2.200   0.400   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.700   0.300   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       4.300   1.700   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       5.500   1.900   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       4.200  -0.600   1.300  1.00  0.00           C
HETATM   11  C1  LIG A 999       0.000   2.200   4.900  1.00  0.00           C
END
"""

# Ortholog: same backbone, residue 1's CB moved farther from where the
# ligand would be. We replace the CB coordinate so the C-LIG distance
# is ~5.5 A instead of ~3.5 A.
_ORTHOLOG_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   1.500   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.500   1.500   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.000   2.600   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       0.000   2.200  -0.600  1.00  0.00           C
ATOM      6  N   ALA A   2       2.200   0.400   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.700   0.300   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       4.300   1.700   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       5.500   1.900   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       4.200  -0.600   1.300  1.00  0.00           C
HETATM   11  C1  LIG A 999       0.000   2.200   4.900  1.00  0.00           C
END
"""


class TestFindLigandAtomsByResname:
    """The ligand-atom-by-residue-name helper."""

    def test_finds_lig_atom(self):
        s = parse_pdb_string(_TARGET_PDB, name="target")
        idx, n = find_ligand_atoms_by_resname(s, "LIG")
        assert n == 1
        assert len(idx) == 1
        # The LIG atom should be index 10 (last of 11 atoms, 0-indexed).
        assert idx[0] == 10

    def test_missing_resname_returns_empty(self):
        s = parse_pdb_string(_TARGET_PDB, name="target")
        idx, n = find_ligand_atoms_by_resname(s, "ZZZ")
        assert n == 0
        assert len(idx) == 0


class TestSyntheticGroundTruth:
    """The synthetic two-protein pair: residue 1's CB is close to ligand
    in target, far in ortholog. The selectivity map must attribute the
    gap to residue index 0."""

    def test_target_more_favorable_than_ortholog(self):
        target = parse_pdb_string(_TARGET_PDB, name="target")
        ortholog = parse_pdb_string(_ORTHOLOG_PDB, name="ortholog")

        smap = compute_selectivity_map(
            target_structure=target,
            target_ligand_resname="LIG",
            ortholog_structure=ortholog,
            ortholog_ligand_resname="LIG",
            pocket_cutoff=6.0,
        )

        # Target should have MORE FAVORABLE (more negative) total ligand
        # interaction than ortholog (the close CB contact in target).
        assert smap.total_target_interaction_kcal < smap.total_ortholog_interaction_kcal, (
            f"Target should be more favorable than ortholog. "
            f"Target={smap.total_target_interaction_kcal:.3f}, "
            f"Ortholog={smap.total_ortholog_interaction_kcal:.3f}"
        )

    def test_top_target_selective_residue_is_residue_0(self):
        target = parse_pdb_string(_TARGET_PDB, name="target")
        ortholog = parse_pdb_string(_ORTHOLOG_PDB, name="ortholog")

        smap = compute_selectivity_map(
            target_structure=target,
            target_ligand_resname="LIG",
            ortholog_structure=ortholog,
            ortholog_ligand_resname="LIG",
            pocket_cutoff=6.0,
        )

        # Residue index 0 should be the top target-selective residue
        top = smap.top_n_target_selective(n=1)
        assert len(top) >= 1
        assert top[0].residue_index_target == 0, (
            f"Top target-selective residue should be index 0 (where the "
            f"close-contact CB lives), got {top[0].residue_index_target}"
        )
        # And the delta should be substantially positive
        # (positive delta = target preferred)
        assert top[0].delta_kcal > 0, (
            f"Top target-selective residue should have positive delta, "
            f"got {top[0].delta_kcal}"
        )

    def test_residue_1_has_near_zero_delta(self):
        """Residue 2 (index 1) is identical in both structures, so its
        delta should be near zero."""
        target = parse_pdb_string(_TARGET_PDB, name="target")
        ortholog = parse_pdb_string(_ORTHOLOG_PDB, name="ortholog")
        smap = compute_selectivity_map(
            target_structure=target,
            target_ligand_resname="LIG",
            ortholog_structure=ortholog,
            ortholog_ligand_resname="LIG",
            pocket_cutoff=6.0,
        )
        res1 = next((r for r in smap.residues if r.residue_index_target == 1), None)
        if res1 is not None:
            assert abs(res1.delta_kcal) < 0.01, (
                f"Residue 2 is identical in both structures; "
                f"delta_kcal should be ~0, got {res1.delta_kcal}"
            )

    def test_self_comparison_yields_zero_deltas(self):
        """Compare a structure to itself; all deltas should be exactly zero."""
        target = parse_pdb_string(_TARGET_PDB, name="target")
        # Re-parse to get an independent copy
        target2 = parse_pdb_string(_TARGET_PDB, name="target2")
        smap = compute_selectivity_map(
            target_structure=target,
            target_ligand_resname="LIG",
            ortholog_structure=target2,
            ortholog_ligand_resname="LIG",
            pocket_cutoff=6.0,
        )
        for r in smap.residues:
            assert abs(r.delta_kcal) < 1e-6, (
                f"Self-comparison delta should be 0, got {r.delta_kcal} "
                f"at residue {r.res_name_target}{r.res_seq_target}"
            )

    def test_missing_ligand_raises(self):
        target = parse_pdb_string(_TARGET_PDB, name="target")
        ortholog = parse_pdb_string(_ORTHOLOG_PDB, name="ortholog")
        with pytest.raises(ValueError, match="No atoms found"):
            compute_selectivity_map(
                target_structure=target,
                target_ligand_resname="ZZZ",
                ortholog_structure=ortholog,
                ortholog_ligand_resname="LIG",
            )


class TestSerialization:
    """JSON-readiness of the SelectivityMap."""

    def test_dict_round_trips(self):
        import json
        target = parse_pdb_string(_TARGET_PDB, name="target")
        ortholog = parse_pdb_string(_ORTHOLOG_PDB, name="ortholog")
        smap = compute_selectivity_map(
            target, "LIG", ortholog, "LIG", pocket_cutoff=6.0,
        )
        d = selectivity_map_to_dict(smap)
        # All keys present
        for key in ("target_name", "ortholog_name", "pocket_delta_kcal",
                    "top_n_target_selective", "top_n_ortholog_selective",
                    "residues"):
            assert key in d
        # JSON round-trip
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["target_name"] == "target"


class TestDHODHIntegration:
    """End-to-end on the 1D3H (HsDHODH catalytic construct + A26) / 1MVS
    (HsDHFR + DTM) pair. The structures are committed in the benchmark
    directory.

    This is a functional-paralog comparison between two human enzymes,
    not a parasite-vs-human selectivity comparison. The test verifies
    only that per-residue attribution mechanics work on real co-crystal
    data — sensible energies, non-empty pocket. Biology claims about
    selectivity are out of scope for this test.
    """

    def _hsdhodh_path(self) -> Path:
        return REPO_ROOT / "benchmark" / "structures" / "experimental" / "1D3H.pdb"

    def _hsdhfr_path(self) -> Path:
        return REPO_ROOT / "benchmark" / "structures" / "experimental" / "1MVS.pdb"

    def test_dhodh_pair_runs_end_to_end(self):
        if not self._hsdhodh_path().exists() or not self._hsdhfr_path().exists():
            pytest.skip("DHODH/DHFR benchmark structures not in this checkout")

        hs_dhodh = parse_pdb(self._hsdhodh_path())
        hs_dhfr = parse_pdb(self._hsdhfr_path())

        smap = compute_selectivity_map(
            target_structure=hs_dhodh,
            target_ligand_resname="A26",
            ortholog_structure=hs_dhfr,
            ortholog_ligand_resname="DTM",
            pocket_cutoff=5.0,
        )

        # The maps must have a reasonable number of aligned pocket residues
        assert smap.n_aligned_pocket_residues > 10, (
            f"Pocket alignment too sparse: only "
            f"{smap.n_aligned_pocket_residues} aligned residues. "
            f"Either active site should have at least 10 contact residues."
        )

        # Both totals should be substantially negative (favorable) — these
        # are real inhibitors in real binding modes; LJ interaction in the
        # tens of -kcal/mol is expected.
        assert smap.total_target_interaction_kcal < -5.0, (
            f"1D3H total LJ interaction not favorable: "
            f"{smap.total_target_interaction_kcal:.2f} kcal/mol. "
            f"Real inhibitor should give meaningful negative number."
        )
        assert smap.total_ortholog_interaction_kcal < -5.0, (
            f"1MVS total LJ interaction not favorable: "
            f"{smap.total_ortholog_interaction_kcal:.2f} kcal/mol."
        )

        # The ranked list must be non-empty. We do not assert the sign of
        # the top delta — the comparison is between two different human
        # enzymes binding two different chemotypes, so a "selective"
        # interpretation of the ranking is not warranted.
        top_target = smap.top_n_target_selective(n=5)
        assert len(top_target) > 0
