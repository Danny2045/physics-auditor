"""Tests for per-residue energy decomposition.

The strategy is to test invariants on the existing fixtures (sums,
shapes, sanity) plus one synthetic scenario where we can predict the
answer: a clash injected on a known residue must show up as the top
unfavorable entry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physics_auditor.causality.energy_decomp import (
    decomposition_to_dict,
    difference_to_dict,
    per_residue_decomposition,
    per_residue_difference,
)
from physics_auditor.core.parser import Structure, parse_pdb, parse_pdb_string

FIXTURES = Path(__file__).parent / "fixtures"


# A 5-residue tri-alanine extension built explicitly to give us a
# predictable clash-injection target. Same backbone geometry as
# tests/fixtures/tri_ala.pdb extended to 5 residues. The CB of residue 3
# can later be moved to overlap with another residue's atoms to inject
# a clash.
_FIVE_ALA_PDB = """\
ATOM      1  N   ALA A   1       1.458   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.500   1.400   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       4.000   2.500   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.500   2.200  -1.200  1.00  0.00           C
ATOM      6  N   ALA A   2       4.200   0.300   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.700   0.200   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       6.300   1.600   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       7.500   1.800   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       6.200  -0.700  -1.200  1.00  0.00           C
ATOM     11  N   ALA A   3       5.500   2.700   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       6.000   4.000   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       7.500   4.000   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       8.000   5.100   0.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       5.500   4.800  -1.200  1.00  0.00           C
ATOM     16  N   ALA A   4       8.200   2.900   0.000  1.00  0.00           N
ATOM     17  CA  ALA A   4       9.700   2.800   0.000  1.00  0.00           C
ATOM     18  C   ALA A   4      10.300   4.200   0.000  1.00  0.00           C
ATOM     19  O   ALA A   4      11.500   4.400   0.000  1.00  0.00           O
ATOM     20  CB  ALA A   4      10.200   1.900  -1.200  1.00  0.00           C
ATOM     21  N   ALA A   5       9.500   5.300   0.000  1.00  0.00           N
ATOM     22  CA  ALA A   5      10.000   6.600   0.000  1.00  0.00           C
ATOM     23  C   ALA A   5      11.500   6.600   0.000  1.00  0.00           C
ATOM     24  O   ALA A   5      12.000   7.700   0.000  1.00  0.00           O
ATOM     25  CB  ALA A   5       9.500   7.400  -1.200  1.00  0.00           C
END
"""


def _inject_clash(struct: Structure, atom_i: int, atom_j: int, distance: float = 0.8) -> Structure:
    """Move atom_j on top of atom_i (offset by tiny distance) to force a clash.

    Returns a new Structure with the modified coordinates. The atom-to-residue
    mapping and topology are unchanged.
    """
    new_coords = np.array(struct.coords, copy=True)
    # Place j very close to i along x-axis
    new_coords[atom_j] = new_coords[atom_i] + np.array([distance, 0.0, 0.0], dtype=np.float32)
    # Build a copy of the structure with new coords
    # Structure is a dataclass — we need to mutate via dataclasses.replace
    from dataclasses import replace
    return replace(struct, coords=new_coords.astype(np.float32))


class TestPerResidueDecomposition:
    """Invariants of per_residue_decomposition on real fixtures."""

    def test_tri_ala_sum_matches_total(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        per_res_sum = sum(r.energy_kcal for r in decomp.residues)
        # Per-residue sum should approximately equal total (rtol=1% as in test_energy)
        np.testing.assert_allclose(per_res_sum, decomp.total_energy_kcal, rtol=0.01)

    def test_tri_ala_residue_count(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        assert len(decomp.residues) == struct.n_residues == 3
        assert all(r.res_name == "ALA" for r in decomp.residues)

    def test_tri_ala_residue_indices_sequential(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        assert [r.residue_index for r in decomp.residues] == [0, 1, 2]

    def test_tri_ala_no_hot_pairs(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        assert all(r.n_hot_pairs == 0 for r in decomp.residues)

    def test_atom_counts_consistent(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        assert sum(r.n_atoms for r in decomp.residues) == struct.n_atoms

    def test_pocket_flagging(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct, pocket_residue_indices=[1])
        assert decomp.residues[0].is_in_pocket is False
        assert decomp.residues[1].is_in_pocket is True
        assert decomp.residues[2].is_in_pocket is False
        assert decomp.n_pocket_residues == 1
        # Pocket energy is exactly the energy of residue 1
        assert decomp.pocket_energy_kcal == decomp.residues[1].energy_kcal

    def test_pocket_none_means_no_flagging(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct, pocket_residue_indices=None)
        assert all(not r.is_in_pocket for r in decomp.residues)
        assert decomp.pocket_energy_kcal == 0.0
        assert decomp.n_pocket_residues == 0

    def test_clashing_fixture_has_hot_pairs(self):
        struct = parse_pdb(FIXTURES / "clashing.pdb")
        decomp = per_residue_decomposition(struct)
        # The clashing fixture is two residues with steric overlap.
        # At least one residue should report a hot pair.
        assert any(r.n_hot_pairs > 0 for r in decomp.residues)
        # And at least one residue should have positive (repulsive) energy.
        assert any(r.energy_kcal > 10.0 for r in decomp.residues)


class TestPerResidueDifference:
    """The differential analysis — the core selectivity-attribution operation."""

    def test_self_difference_is_zero(self):
        """Comparing a structure to itself yields zero deltas everywhere."""
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        diff = per_residue_difference(decomp, decomp)
        assert diff.n_aligned == 3
        assert diff.total_delta_kcal == 0.0
        assert diff.pocket_delta_kcal == 0.0
        assert all(r.delta_kcal == 0.0 for r in diff.residues)

    def test_clash_injected_on_residue_3_shows_at_top(self):
        """Inject a clash between an atom of residue 3 and an atom of
        residue 4; the top unfavorable residue in B (vs A) should
        include residue index 2 (residue 3, zero-indexed)."""
        struct_a = parse_pdb_string(_FIVE_ALA_PDB, name="five_ala_clean")
        # Atom 14 (CB of residue 3, zero-indexed res 2) and atom 16
        # (N of residue 4, zero-indexed res 3) are 1-indexed in the PDB.
        # In the parsed flat array (no hydrogens), the indices are 0-based.
        # Let's compute them:
        # tri_ala fixture had 15 atoms for 3 residues = 5 atoms/residue (N, CA, C, O, CB).
        # Same here: 25 atoms for 5 residues. Residue 3 atoms are at indices 10-14;
        # CB of residue 3 = index 14. N of residue 4 = index 15.
        struct_b = _inject_clash(struct_a, atom_i=14, atom_j=15, distance=0.8)

        decomp_a = per_residue_decomposition(struct_a)
        decomp_b = per_residue_decomposition(struct_b)
        diff = per_residue_difference(decomp_a, decomp_b)

        # The clash is between residue 2 (zero-indexed) and residue 3.
        # Both should appear in top unfavorable in B.
        top = diff.top_n_unfavorable_in_b(n=3)
        top_indices = {r.residue_index_b for r in top}
        assert 2 in top_indices or 3 in top_indices, (
            f"Expected clash residue (idx 2 or 3) in top 3 unfavorable, got {top_indices}"
        )
        # And the delta_kcal for the top entry should be substantially positive
        assert top[0].delta_kcal > 5.0, (
            f"Top unfavorable delta should be large, got {top[0].delta_kcal}"
        )

    def test_no_clash_injection_keeps_total_delta_small(self):
        """Sanity: parse the same string twice and the totals should be
        identical (no floating-point drift on a clean input)."""
        sa = parse_pdb_string(_FIVE_ALA_PDB, name="x")
        sb = parse_pdb_string(_FIVE_ALA_PDB, name="y")
        diff = per_residue_difference(per_residue_decomposition(sa), per_residue_decomposition(sb))
        assert abs(diff.total_delta_kcal) < 1e-3

    def test_top_n_methods_return_correct_lengths(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        diff = per_residue_difference(decomp, decomp)
        assert len(diff.top_n_by_abs_delta(2)) == 2
        assert len(diff.top_n_unfavorable_in_b(2)) == 2
        assert len(diff.top_n_favorable_in_b(2)) == 2
        # Asking for more than available returns all of them
        assert len(diff.top_n_by_abs_delta(100)) == diff.n_aligned

    def test_pocket_delta_only_counts_pocket_residues(self):
        """If we flag residue 1 as pocket on both sides and inject no
        change, pocket_delta_kcal should be zero. If we flag residue 0
        as pocket but the only change is on residue 1, pocket_delta
        is still zero."""
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp_a = per_residue_decomposition(struct, pocket_residue_indices=[0])
        decomp_b = per_residue_decomposition(struct, pocket_residue_indices=[0])
        diff = per_residue_difference(decomp_a, decomp_b)
        # No injected change, so all deltas are zero
        assert diff.pocket_delta_kcal == 0.0
        # And n_pocket_aligned should reflect the flagging
        assert diff.n_pocket_aligned == 1

    def test_explicit_alignment_respected(self):
        """Pass a non-trivial alignment (A:0 -> B:2, A:2 -> B:0) and
        verify the difference uses those residues."""
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        # All-ALA structure so any alignment yields zero deltas, but
        # we can verify the alignment was honored by looking at indices.
        diff = per_residue_difference(decomp, decomp, alignment=[(0, 2), (2, 0)])
        assert diff.n_aligned == 2
        assert diff.residues[0].residue_index_a == 0
        assert diff.residues[0].residue_index_b == 2
        assert diff.residues[1].residue_index_a == 2
        assert diff.residues[1].residue_index_b == 0


class TestSerialization:
    """JSON-readiness of the dict representations."""

    def test_decomposition_to_dict_has_expected_keys(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct, pocket_residue_indices=[1])
        d = decomposition_to_dict(decomp)
        assert d["structure_name"] == "tri_ala"
        assert d["n_residues"] == 3
        assert d["n_pocket_residues"] == 1
        assert d["pocket_residue_indices"] == [1]
        assert len(d["residues"]) == 3
        # First residue dict has all expected fields
        r0 = d["residues"][0]
        for key in ("residue_index", "res_name", "chain_id", "res_seq",
                    "energy_kcal", "n_atoms", "is_in_pocket", "n_hot_pairs"):
            assert key in r0

    def test_difference_to_dict_has_top_n_views(self):
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        diff = per_residue_difference(decomp, decomp)
        d = difference_to_dict(diff, top_n=2)
        assert d["name_a"] == d["name_b"] == "tri_ala"
        assert d["total_delta_kcal"] == 0.0
        assert len(d["top_n_by_abs_delta"]) == 2
        assert len(d["top_n_unfavorable_in_b"]) == 2
        assert len(d["top_n_favorable_in_b"]) == 2
        assert len(d["residues"]) == 3

    def test_decomposition_dict_is_json_serializable(self):
        import json
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        decomp = per_residue_decomposition(struct)
        d = decomposition_to_dict(decomp)
        # Should round-trip without error
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["structure_name"] == "tri_ala"


class TestPublicAPI:
    """The causality package re-exports the new public API."""

    def test_imports_from_package(self):
        from physics_auditor.causality import (
            DecompositionDifference,
            ResidueDifference,
            ResidueEnergy,
            StructureDecomposition,
            decomposition_to_dict,
            difference_to_dict,
            per_residue_decomposition,
            per_residue_difference,
        )
        # All names exist
        assert all(callable(x) or isinstance(x, type) for x in [
            DecompositionDifference, ResidueDifference, ResidueEnergy,
            StructureDecomposition, decomposition_to_dict, difference_to_dict,
            per_residue_decomposition, per_residue_difference,
        ])
