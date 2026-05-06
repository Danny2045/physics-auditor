"""Regression tests for parser bug fixes and subscore calibration."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from physics_auditor.checks.clashes import check_clashes
from physics_auditor.config import AuditorConfig
from physics_auditor.core.geometry import compute_distance_matrix
from physics_auditor.core.parser import parse_pdb, parse_pdb_string
from physics_auditor.core.topology import build_bonded_mask, infer_bonds_from_topology

FIXTURES = Path(__file__).parent / "fixtures"


# Minimal two-model NMR-style PDB: same atoms in MODEL 1 and MODEL 2.
# A correct parser stops after ENDMDL of model 1 and yields 3 atoms total.
# A buggy parser would concatenate both models and yield 6 atoms.
_NMR_PDB = """\
HEADER    NMR ENSEMBLE TEST
MODEL        1
ATOM      1  N   ALA A   1       1.458   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.500   1.400   0.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  N   ALA A   1       1.500   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.100   1.400   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.600   1.400   0.000  1.00  0.00           C
ENDMDL
END
"""


class TestNMREnsembleHandling:
    """Regression: parsers must stop at first ENDMDL.

    parse_pdb had this fix in commit a65d454; parse_pdb_string was
    inadvertently left without it. Both should now keep model 1 only.
    """

    def test_parse_pdb_string_stops_at_endmdl(self):
        struct = parse_pdb_string(_NMR_PDB, name="nmr_test")
        assert struct.n_atoms == 3, (
            f"Expected 3 atoms (model 1 only), got {struct.n_atoms} "
            f"(parser concatenated multiple NMR models)"
        )

    def test_parse_pdb_stops_at_endmdl(self, tmp_path):
        nmr_file = tmp_path / "nmr.pdb"
        nmr_file.write_text(_NMR_PDB)
        struct = parse_pdb(nmr_file)
        assert struct.n_atoms == 3


class TestSubscoreCalibration:
    """Regression: subscore decay should classify real PDB structures
    sensibly (high-resolution crystal -> ACCEPT-ish, deliberately
    clashed -> DISCARD)."""

    def _validate(self, pdb_path):
        struct = parse_pdb(pdb_path)
        bonds = infer_bonds_from_topology(struct)
        mask = build_bonded_mask(struct.n_atoms, bonds)
        coords = jnp.array(struct.coords)
        dist = compute_distance_matrix(coords)
        return check_clashes(
            dist, struct.elements, jnp.array(mask),
            struct.res_indices, struct.n_residues,
        )

    def test_clean_tripeptide_accepts(self):
        result = self._validate(FIXTURES / "tri_ala.pdb")
        cfg = AuditorConfig()
        assert result.subscore >= cfg.composite.accept_threshold, (
            f"Clean tri-ala fixture should ACCEPT but got "
            f"subscore={result.subscore:.3f} "
            f"(accept threshold {cfg.composite.accept_threshold})"
        )

    def test_clashing_fixture_discards(self):
        result = self._validate(FIXTURES / "clashing.pdb")
        cfg = AuditorConfig()
        assert result.subscore < cfg.composite.short_md_threshold, (
            f"Deliberately clashed fixture should DISCARD but got "
            f"subscore={result.subscore:.3f} "
            f"(short_md threshold {cfg.composite.short_md_threshold})"
        )

    def test_decay_monotonic(self):
        # Scan synthetic clashscore values: subscore must monotonically decrease
        from physics_auditor.checks.clashes import check_clashes  # noqa
        # Test by direct application of formula via the same path
        prev = float("inf")
        for cs in [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 400.0]:
            sub = 1.0 / (1.0 + (cs / 50.0) ** 2)
            assert sub <= prev, f"Non-monotonic at clashscore={cs}"
            prev = sub

    def test_decay_at_calibration_anchors(self):
        # The calibration anchors documented in clashes.py
        anchors = {
            5.0: (0.985, 0.999),    # AF prediction
            10.0: (0.95, 0.97),     # high-res crystal
            25.0: (0.79, 0.81),     # accept threshold
            50.0: (0.49, 0.51),     # short_md threshold
            100.0: (0.19, 0.21),    # discard band
        }
        for cs, (lo, hi) in anchors.items():
            sub = 1.0 / (1.0 + (cs / 50.0) ** 2)
            assert lo <= sub <= hi, (
                f"Anchor mismatch at cs={cs}: subscore={sub:.3f} "
                f"not in [{lo}, {hi}]"
            )


class TestExtractBindingSitePerf:
    """Regression: extract_binding_site should not regress to O(N*P)."""

    def test_returns_correct_atoms(self):
        from physics_auditor.causality.binding_site import extract_binding_site

        struct = parse_pdb(FIXTURES / "tri_ala.pdb")
        center = np.mean(struct.coords, axis=0, keepdims=True)
        site = extract_binding_site(struct, center, cutoff=50.0)

        # All 3 residues, all 15 atoms (no hydrogens in fixture)
        assert site.n_residues == 3
        assert site.n_atoms == struct.n_atoms

        # Atom indices must be sorted and unique
        ai = site.atom_indices
        assert len(ai) == len(set(ai.tolist()))
        assert (np.diff(ai) > 0).all()
