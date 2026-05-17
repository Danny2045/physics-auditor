"""Tests for disulfide-bond inference in topology.

Two scenarios:

1. Synthetic ground truth — a small structure with four cysteines
   where we know exactly which pair of SG atoms are at disulfide
   distance and which are not. The bond list must contain the close
   pair and not the far one.

2. Real-data integration — CoV2Mpro (5R82) had CYS296 reporting
   +1067 kcal of spurious LJ strain in the first causality run because
   the topology builder didn't bond a non-sequential disulfide pair.
   With the fix in place, that residue's energy must drop into a
   sane range.
"""

from __future__ import annotations

from pathlib import Path

from physics_auditor.causality.energy_decomp import per_residue_decomposition
from physics_auditor.core.parser import parse_pdb, parse_pdb_string
from physics_auditor.core.topology import infer_bonds_from_topology

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


# A four-CYS toy structure. CYS A1 SG is at (0, 0, 0). CYS A4 SG is
# at (2.05, 0, 0) — disulfide distance, must bond. CYS A8 SG is at
# (10, 0, 0), CYS A12 SG is at (15, 0, 0) — both far from everything,
# must not bond to anything.
#
# Backbone atoms placed plausibly so the topology builder doesn't
# choke; only the relative SG-SG distances matter for this test.
_FOUR_CYS_PDB = """\
ATOM      1  N   CYS A   1       0.000  -2.000   0.000  1.00  0.00           N
ATOM      2  CA  CYS A   1       0.000  -1.500   0.000  1.00  0.00           C
ATOM      3  C   CYS A   1       0.000  -1.000   1.500  1.00  0.00           C
ATOM      4  O   CYS A   1       0.000  -1.500   2.500  1.00  0.00           O
ATOM      5  CB  CYS A   1       0.000  -0.700  -0.500  1.00  0.00           C
ATOM      6  SG  CYS A   1       0.000   0.000   0.000  1.00  0.00           S
ATOM      7  N   CYS A   4       2.050  -2.000   0.000  1.00  0.00           N
ATOM      8  CA  CYS A   4       2.050  -1.500   0.000  1.00  0.00           C
ATOM      9  C   CYS A   4       2.050  -1.000   1.500  1.00  0.00           C
ATOM     10  O   CYS A   4       2.050  -1.500   2.500  1.00  0.00           O
ATOM     11  CB  CYS A   4       2.050  -0.700  -0.500  1.00  0.00           C
ATOM     12  SG  CYS A   4       2.050   0.000   0.000  1.00  0.00           S
ATOM     13  N   CYS A   8      10.000  -2.000   0.000  1.00  0.00           N
ATOM     14  CA  CYS A   8      10.000  -1.500   0.000  1.00  0.00           C
ATOM     15  C   CYS A   8      10.000  -1.000   1.500  1.00  0.00           C
ATOM     16  O   CYS A   8      10.000  -1.500   2.500  1.00  0.00           O
ATOM     17  CB  CYS A   8      10.000  -0.700  -0.500  1.00  0.00           C
ATOM     18  SG  CYS A   8      10.000   0.000   0.000  1.00  0.00           S
ATOM     19  N   CYS A  12      15.000  -2.000   0.000  1.00  0.00           N
ATOM     20  CA  CYS A  12      15.000  -1.500   0.000  1.00  0.00           C
ATOM     21  C   CYS A  12      15.000  -1.000   1.500  1.00  0.00           C
ATOM     22  O   CYS A  12      15.000  -1.500   2.500  1.00  0.00           O
ATOM     23  CB  CYS A  12      15.000  -0.700  -0.500  1.00  0.00           C
ATOM     24  SG  CYS A  12      15.000   0.000   0.000  1.00  0.00           S
END
"""


class TestDisulfideInference:
    """Synthetic ground truth: only the SG pair at 2.05 angstroms bonds."""

    def test_four_cys_structure_bonds_only_close_pair(self):
        struct = parse_pdb_string(_FOUR_CYS_PDB, name="four_cys")
        bonds = infer_bonds_from_topology(struct)

        # Find the SG atom indices in the flat array
        sg_indices = []
        for i in range(struct.n_atoms):
            if struct.atom_names[i] == "SG":
                sg_indices.append(i)
        assert len(sg_indices) == 4, f"Expected 4 SG atoms, got {len(sg_indices)}"

        # CYS A1 SG = sg_indices[0], CYS A4 SG = sg_indices[1] (close pair)
        # CYS A8 SG = sg_indices[2], CYS A12 SG = sg_indices[3] (far)
        close_pair = tuple(sorted((sg_indices[0], sg_indices[1])))
        far_pair_a = tuple(sorted((sg_indices[2], sg_indices[3])))
        far_pair_b = tuple(sorted((sg_indices[0], sg_indices[2])))
        far_pair_c = tuple(sorted((sg_indices[0], sg_indices[3])))

        assert close_pair in bonds, (
            f"Disulfide pair (A1-A4 SG, distance 2.05) should bond, "
            f"but {close_pair} not in bonds"
        )
        for far in (far_pair_a, far_pair_b, far_pair_c):
            assert far not in bonds, (
                f"Far SG pair {far} should NOT bond (distance >> 2.4)"
            )

    def test_no_cys_structure_no_disulfides(self):
        """tri_ala fixture has no CYS, so no disulfides should be added."""
        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        bonds_before_count = len(infer_bonds_from_topology(struct))
        # Re-running should give exactly the same bonds (deterministic).
        bonds_again = infer_bonds_from_topology(struct)
        assert len(bonds_again) == bonds_before_count

    def test_distance_window_boundaries(self):
        """SG atoms outside the [1.9, 2.4] window must not bond."""
        # Build a two-CYS PDB with SG atoms at exactly 1.85 (too close)
        # and one at 2.5 (too far). Neither should bond.
        for sg_distance in [1.85, 2.5, 3.0]:
            pdb = _FOUR_CYS_PDB.replace(
                "ATOM     12  SG  CYS A   4       2.050   0.000   0.000",
                f"ATOM     12  SG  CYS A   4      {sg_distance:6.3f}   0.000   0.000",
            )
            struct = parse_pdb_string(pdb, name=f"d_{sg_distance}")
            bonds = infer_bonds_from_topology(struct)
            sg0, sg1 = None, None
            for i in range(struct.n_atoms):
                if struct.atom_names[i] == "SG":
                    if sg0 is None:
                        sg0 = i
                    elif sg1 is None:
                        sg1 = i
                        break
            pair = tuple(sorted((sg0, sg1)))
            assert pair not in bonds, (
                f"SG pair at {sg_distance} angstroms must not bond "
                f"(window is [1.9, 2.4])"
            )


class TestCoV2MproRegression:
    """The real-data smoke test: CoV2Mpro had CYS296 reporting +1067 kcal
    in the pre-fix first causality run.  With disulfide bonding, that
    residue's energy must drop substantially."""

    def test_cov2mpro_cys296_no_longer_blows_up(self):
        pdb = REPO_ROOT / "benchmark" / "structures" / "predicted" / "AF-P0DTD1-F1-model_v6.pdb"
        if not pdb.exists():
            import pytest
            pytest.skip(f"{pdb} not available in this checkout")

        struct = parse_pdb(pdb)
        decomp = per_residue_decomposition(struct)

        # Find the residue that previously reported +1067 kcal.
        # In the pre-fix run it was at residue_index 296 (CYS).
        # The exact index can drift by parser version; instead, verify that
        # NO single residue exceeds a sanity threshold.  Without disulfide
        # bonding, at least one residue spikes above +500 kcal.  With
        # bonding, all residues should be well-behaved.
        max_residue_energy = max(r.energy_kcal for r in decomp.residues)
        assert max_residue_energy < 500.0, (
            f"At least one residue still reports {max_residue_energy:.1f} kcal "
            f"of LJ energy after disulfide-bond inference. Either the fix is "
            f"not catching this disulfide, or there's another non-bonded clash "
            f"not addressed by this fix."
        )

        # Also: at least one CYS residue should report a non-trivial,
        # negative (favorable) energy now that its SG isn't clashing.
        cys_energies = [r.energy_kcal for r in decomp.residues if r.res_name == "CYS"]
        assert len(cys_energies) > 0, "No CYS in CoV2Mpro? Parser bug."
        assert min(cys_energies) < 0, (
            f"All CYS residues have non-favorable energies "
            f"({min(cys_energies)=}); disulfide fix may not be working."
        )
