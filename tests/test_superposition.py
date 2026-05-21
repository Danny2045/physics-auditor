"""Tests for the Kabsch superposition + cofactor-anchored correspondence engine.

Two layers:

1. Kabsch known-answer tests on synthetic point sets where we KNOW the
   correct rotation/translation because we applied it ourselves. These
   prove the SVD-based Kabsch is mathematically correct (and that the
   reflection correction fires) BEFORE the routine touches a real
   structure. If any of these fail, the convergent-validation result
   downstream is not interpretable as biology — it could be a Kabsch
   bug.

2. Anchor-pairing test: the sequence-independent atom-name pairing
   that seeds the structural superposition. Confirms the set-equality
   sanity check on cofactor atom names handles mismatched-name cases
   correctly (shared atoms paired, mismatched atoms reported).
"""

from __future__ import annotations

import numpy as np

from physics_auditor.causality.superposition import (
    build_anchor_pairing,
    kabsch_superpose,
)
from physics_auditor.core.parser import parse_pdb_string

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_from_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a proper rotation matrix from Z-Y-X intrinsic Euler angles (radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Kabsch known-answer tests
# ---------------------------------------------------------------------------


class TestKabschCore:
    def test_kabsch_recovers_known_rotation(self):
        """Apply a KNOWN rotation+translation; Kabsch must recover R, t, RMSD≈0."""
        rng = np.random.default_rng(42)
        P = rng.uniform(-5.0, 5.0, size=(20, 3))
        R_true = _rotation_from_euler(0.5, 0.3, 0.7)
        t_true = np.array([1.0, -2.0, 3.5])
        Q = P @ R_true.T + t_true

        R, t, rmsd = kabsch_superpose(P, Q)

        np.testing.assert_allclose(R, R_true, atol=1e-10)
        np.testing.assert_allclose(t, t_true, atol=1e-10)
        assert rmsd < 1e-9
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_kabsch_zero_rmsd_identical(self):
        """P == Q must give R ≈ I, t ≈ 0, RMSD ≈ 0."""
        rng = np.random.default_rng(123)
        P = rng.uniform(-3.0, 3.0, size=(15, 3))

        R, t, rmsd = kabsch_superpose(P, P)

        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-10)
        assert rmsd < 1e-10
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_kabsch_handles_reflection(self):
        """A reflected target must NOT be solved by an improper rotation.

        Without the det(R)<0 correction, the SVD optimum here would be a
        reflection (det = -1) with RMSD = 0 — biologically wrong (mirror
        image). With the correction, the result is a proper rotation
        (det = +1) and the RMSD is nonzero because no proper rotation
        maps a generic 3D point set onto its own mirror image.
        """
        rng = np.random.default_rng(7)
        P = rng.uniform(-2.0, 2.0, size=(10, 3))
        Q = P.copy()
        Q[:, 2] = -Q[:, 2]

        R, t, rmsd = kabsch_superpose(P, Q)

        det_R = float(np.linalg.det(R))
        assert np.isclose(det_R, 1.0, atol=1e-10), (
            f"reflection correction failed: det(R)={det_R}, expected +1"
        )
        assert rmsd > 0.1, (
            f"reflected target with proper rotation should give non-trivial "
            f"RMSD; got {rmsd:.6f}"
        )

    def test_kabsch_with_noise(self):
        """RMSD must scale with input noise — close to sigma*sqrt(3) for σ=0.1."""
        rng = np.random.default_rng(99)
        P = rng.uniform(-5.0, 5.0, size=(50, 3))
        R_true = _rotation_from_euler(0.3, 0.5, 0.2)
        t_true = np.array([2.0, -1.0, 0.5])
        sigma = 0.1
        noise = rng.normal(0.0, sigma, size=P.shape)
        Q = P @ R_true.T + t_true + noise

        R, t, rmsd = kabsch_superpose(P, Q)

        np.testing.assert_allclose(R, R_true, atol=0.05)
        # Expected post-Kabsch RMSD for 3D Gaussian noise N(0, σ²) ≈ √3·σ ≈ 0.173.
        # Allow a band that catches gross bugs while not flaking on randomness.
        assert 0.10 < rmsd < 0.25, (
            f"noisy-fit RMSD {rmsd:.6f} outside expected band [0.10, 0.25] "
            f"for σ=0.1; either noise scaling is wrong or Kabsch is broken"
        )
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Anchor pairing — atom-name intersection
# ---------------------------------------------------------------------------


_TARGET_COFACTOR_PDB = """\
HETATM    1  N1  COF X   1       0.000   0.000   0.000  1.00  0.00           N
HETATM    2  C1  COF X   1       1.000   0.000   0.000  1.00  0.00           C
HETATM    3  C2  COF X   1       2.000   0.000   0.000  1.00  0.00           C
HETATM    4  X1  COF X   1       3.000   0.000   0.000  1.00  0.00           C
END
"""

_ORTHOLOG_COFACTOR_PDB = """\
HETATM    1  N1  COF Y   1       0.500   0.100   0.000  1.00  0.00           N
HETATM    2  C1  COF Y   1       1.500   0.100   0.000  1.00  0.00           C
HETATM    3  C2  COF Y   1       2.500   0.100   0.000  1.00  0.00           C
HETATM    4  X2  COF Y   1       3.500   0.100   0.000  1.00  0.00           C
END
"""


def test_anchor_pairing_atom_name_intersection(capsys):
    """build_anchor_pairing pairs shared atom names and reports the rest."""
    target = parse_pdb_string(_TARGET_COFACTOR_PDB, name="target")
    ortholog = parse_pdb_string(_ORTHOLOG_COFACTOR_PDB, name="ortholog")

    P, Q, shared = build_anchor_pairing(target, ortholog, "COF")

    assert shared == ["C1", "C2", "N1"]
    assert P.shape == (3, 3)
    assert Q.shape == (3, 3)

    # Spot-check pairing: P[shared.index('N1')] is target's N1 atom (0,0,0);
    # Q at the same row is ortholog's N1 (0.5, 0.1, 0.0).
    n1_row = shared.index("N1")
    np.testing.assert_allclose(P[n1_row], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(Q[n1_row], [0.5, 0.1, 0.0])

    captured = capsys.readouterr().out
    assert "X1" in captured, "unmatched target atom 'X1' must be reported"
    assert "X2" in captured, "unmatched ortholog atom 'X2' must be reported"
    assert "COF" in captured
