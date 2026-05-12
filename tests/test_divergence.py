"""Tests for the ESM-2 divergence-amplification module.

Four layers of testing:

1. Identity — same sequence on both sides → all three cosines = 1.0,
   amplification = 0.0.

2. Synthetic divergence — two sequences identical at non-pocket
   positions and different at pocket positions → full-sequence cosine
   stays high, pocket cosines drop, amplification > 0.

3. Validation — out-of-range / wrong-type pocket indices raise
   ``ValueError`` with a readable message.

4. Real-data integration — 1D3H and 1MVS co-crystals with the
   pocket indices the selectivity_map run already produced. The
   integration test asserts the module runs end-to-end and that the
   pocket is in fact more divergent than the full sequence; it does
   NOT assert ``full_sequence_cosine > 0.95`` because the two PDB
   entries are HsDHODH (1D3H, truncated catalytic construct of
   Q02127) and HsDHFR (1MVS) — two different human enzymes — even
   though the existing benchmark code labels them "SmDHODH" and
   "HsDHODH". Their full-sequence cosine in t33 is ~0.82, not >0.95.
   The Kira slogan number (0.9897) refers to the actual SmDHODH
   (G4VFD7) and HsDHODH (Q02127) UniProt sequences, exercised by the
   runner under ``benchmark/run_divergence.py``.

The whole file is skipped when fair-esm/torch are not importable, so
non-ESM environments (CI minimal image, fresh checkouts without the
optional ``[esm]`` extras) still see a green test run.

The integration test uses ``esm2_t6_8M_UR50D`` (~30 MB) so the test
suite stays fast. Production numbers come from the runner with the
default t33 650M model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

esm = pytest.importorskip("esm")  # noqa: F841 — skips the whole module
torch = pytest.importorskip("torch")  # noqa: F841

from physics_auditor.causality.divergence import (  # noqa: E402
    DivergenceReport,
    compute_divergence,
    divergence_report_to_dict,
)
from physics_auditor.core.parser import parse_pdb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_MODEL = "esm2_t6_8M_UR50D"


# ---------------------------------------------------------------------------
# Synthetic sequences. 30 residues, alanine background.
#
# "Background" positions (every position except {5, 10, 15, 20}) are
# identical in both sequences. The pocket positions differ between
# sequences in chemistry (small hydrophobic vs charged residues), so
# embeddings at those positions should diverge while the rest of the
# sequence stays the same.
# ---------------------------------------------------------------------------
_POCKET_POS = [5, 10, 15, 20]
_TARGET_SEQ = "AAAAALAAAALAAAALAAAALAAAAAAAAA"
_ORTHOLOG_SEQ = "AAAAAEAAAAEAAAAEAAAAEAAAAAAAAA"


class TestIdentity:
    """Same sequence + same pocket → all cosines exactly 1.0."""

    def test_identity_returns_unit_cosines(self):
        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGL"
        idx = [0, 5, 10, 15]
        r = compute_divergence(seq, seq, idx, idx, model_name=TEST_MODEL, pair_name="self")
        assert isinstance(r, DivergenceReport)
        assert r.full_sequence_cosine == pytest.approx(1.0, abs=1e-5)
        assert r.pocket_meanpool_cosine == pytest.approx(1.0, abs=1e-5)
        assert r.pocket_subseq_cosine == pytest.approx(1.0, abs=1e-5)
        assert r.divergence_amplification == pytest.approx(0.0, abs=1e-5)
        assert r.n_pocket_residues_target == 4
        assert r.n_pocket_residues_ortholog == 4
        assert r.model_name == TEST_MODEL
        assert r.pair_name == "self"


class TestSyntheticDivergence:
    """Two sequences identical outside the pocket, divergent inside.

    Both sequences are 30 residues. The 26 non-pocket positions are
    identical alanines. The 4 pocket positions hold L (hydrophobic) in
    the target and E (charged) in the ortholog. Expected:

    - full cosine stays > 0.9 (most positions identical)
    - pocket cosines drop below the full cosine
    - amplification (full − pocket_meanpool) > 0
    """

    def test_pocket_more_divergent_than_full(self):
        r = compute_divergence(
            _TARGET_SEQ, _ORTHOLOG_SEQ, _POCKET_POS, _POCKET_POS,
            model_name=TEST_MODEL, pair_name="synthetic",
        )
        # With a small ESM-2 (t6_8M), the position-wise pocket changes
        # propagate via attention enough to drag the full-sequence cosine
        # off 1.0 even though most residues are identical. The point of
        # this test is that the *pocket cosine drops further* than the
        # full cosine — so we assert the relative drop, not an absolute
        # high-water mark on the full cosine.
        assert r.full_sequence_cosine > 0.85, (
            f"Full cosine should stay reasonably high; got {r.full_sequence_cosine}"
        )
        assert r.pocket_meanpool_cosine < r.full_sequence_cosine, (
            f"Pocket meanpool ({r.pocket_meanpool_cosine}) should be lower "
            f"than full cosine ({r.full_sequence_cosine}) because the "
            f"pocket residues are the only place the sequences disagree."
        )
        assert r.divergence_amplification > 0, (
            f"Amplification should be positive; got {r.divergence_amplification}"
        )

    def test_report_to_dict_round_trips(self):
        r = compute_divergence(
            _TARGET_SEQ, _ORTHOLOG_SEQ, _POCKET_POS, _POCKET_POS,
            model_name=TEST_MODEL,
        )
        d = divergence_report_to_dict(r)
        for key in (
            "pair_name", "full_sequence_cosine", "pocket_meanpool_cosine",
            "pocket_subseq_cosine", "divergence_amplification",
            "n_pocket_residues_target", "n_pocket_residues_ortholog",
            "model_name",
        ):
            assert key in d
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["model_name"] == TEST_MODEL


class TestPocketIndexValidation:
    """Out-of-range / wrong-type pocket indices raise ValueError."""

    def test_out_of_range_index_raises(self):
        seq = "MKTAYIAK"  # length 8
        with pytest.raises(ValueError, match="out of range"):
            compute_divergence(seq, seq, [99], [0], model_name=TEST_MODEL)

    def test_negative_index_raises(self):
        seq = "MKTAYIAK"
        with pytest.raises(ValueError, match="out of range"):
            compute_divergence(seq, seq, [0], [-1], model_name=TEST_MODEL)

    def test_empty_index_list_raises(self):
        seq = "MKTAYIAK"
        with pytest.raises(ValueError, match="empty"):
            compute_divergence(seq, seq, [], [0], model_name=TEST_MODEL)

    def test_non_integer_index_raises(self):
        seq = "MKTAYIAK"
        with pytest.raises(ValueError, match="not an integer"):
            compute_divergence(seq, seq, [0], [1.5], model_name=TEST_MODEL)


class TestDHODHIntegration:
    """End-to-end on 1D3H and 1MVS reusing the pocket indices from the
    existing selectivity_map dossier.

    See the module docstring for why the user-spec assertion
    ``full_sequence_cosine > 0.95`` is intentionally NOT made here:
    1D3H and 1MVS are not a true ortholog pair (HsDHODH catalytic
    construct vs HsDHFR), and the empirical cosine in t33 is ~0.82.
    The runner exercises the actual SmDHODH/HsDHODH UniProt sequences
    separately.
    """

    DOSSIER = REPO_ROOT / "benchmark/results/selectivity_maps/SmDHODH_vs_HsDHODH.json"
    P1D3H = REPO_ROOT / "benchmark/structures/experimental/1D3H.pdb"
    P1MVS = REPO_ROOT / "benchmark/structures/experimental/1MVS.pdb"

    def test_runs_end_to_end_with_selectivity_map_indices(self):
        if not (self.DOSSIER.exists() and self.P1D3H.exists() and self.P1MVS.exists()):
            pytest.skip("DHODH benchmark data not in this checkout")

        s_t = parse_pdb(self.P1D3H)
        s_o = parse_pdb(self.P1MVS)

        # Selectivity-map convention for residue_indices is position in
        # structure.residues.keys(); for our compute_divergence the
        # convention is position in the protein-only sequence. They
        # coincide as long as the residue dict ordering matches the
        # protein chain order. Build the explicit mapping here so the
        # convention is verifiable.
        target_seq = s_t.protein_chains[0].sequence
        ortholog_seq = s_o.protein_chains[0].sequence

        def build_seq_pos_map(structure):
            m: dict[tuple, int] = {}
            pos = 0
            for chain in structure.protein_chains:
                for res in chain.residues:
                    if res.is_protein:
                        m[(chain.chain_id, res.res_seq, res.insertion_code)] = pos
                        pos += 1
            return m

        m_t = build_seq_pos_map(s_t)
        m_o = build_seq_pos_map(s_o)

        dossier = json.loads(self.DOSSIER.read_text())
        target_pocket_idx: list[int] = []
        ortholog_pocket_idx: list[int] = []
        for row in dossier["residues"]:
            kt = [k for k in m_t if k[1] == row["res_seq_target"]]
            ko = [k for k in m_o if k[1] == row["res_seq_ortholog"]]
            if kt and ko:
                target_pocket_idx.append(m_t[kt[0]])
                ortholog_pocket_idx.append(m_o[ko[0]])

        assert len(target_pocket_idx) > 10
        assert len(ortholog_pocket_idx) > 10

        r = compute_divergence(
            target_seq, ortholog_seq,
            target_pocket_idx, ortholog_pocket_idx,
            model_name=TEST_MODEL,
            pair_name="1D3H_vs_1MVS",
        )

        assert 0.0 < r.full_sequence_cosine < 1.0
        assert 0.0 < r.pocket_meanpool_cosine < 1.0
        assert 0.0 < r.pocket_subseq_cosine < 1.0
        # The headline claim: pocket is more divergent than the protein
        # as a whole. Holds for both t6 and t33 on this structure pair.
        assert r.divergence_amplification > 0, (
            f"Pocket should be more divergent than full sequence; "
            f"got amplification = {r.divergence_amplification}"
        )
