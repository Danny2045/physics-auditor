"""Tests for the sequence-anchored correspondence engine.

Two layers:

1. Needleman-Wunsch known-answer tests. Tiny sequences whose optimal
   global alignment we can write down by hand, so we verify the
   aligner produces the expected positional mapping under several
   distinct alignment patterns (identity, leading insertion, point
   substitutions, internal deletion).

2. Structure-level tests for build_residue_index_map (the hard
   consistency gate) and build_pocket_alignment (the refusal gate
   and the high-homology happy path), using minimal hand-built
   Structure fixtures so the expected correspondence is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics_auditor.causality.binding_site import BindingSite
from physics_auditor.causality.correspondence import (
    PocketAlignment,
    build_pocket_alignment,
    build_residue_index_map,
    needleman_wunsch,
)
from physics_auditor.core.parser import Atom, Chain, Residue, Structure

# ---------------------------------------------------------------------------
# Needleman-Wunsch known-answer tests
# ---------------------------------------------------------------------------


class TestNeedlemanWunsch:
    def test_nw_identical_sequences(self):
        """Two identical sequences align as the identity mapping with 100% identity."""
        seq = "ACDEFGHIKL"
        aligned, _ = needleman_wunsch(seq, seq)

        assert aligned == [(i, i) for i in range(len(seq))]
        for p1, p2 in aligned:
            assert seq[p1] == seq[p2]

        n_id = sum(1 for p1, p2 in aligned if seq[p1] == seq[p2])
        assert n_id == len(seq)

    def test_nw_known_offset(self):
        """seq2 = 'MM' + seq1: NW must open one gap of length 2 at the N-terminus."""
        seq1 = "ACDEFGHIKL"
        seq2 = "MM" + seq1
        aligned, _ = needleman_wunsch(seq1, seq2)

        leading_gaps = [(p1, p2) for p1, p2 in aligned[:2]]
        assert leading_gaps == [(None, 0), (None, 1)]

        match_pairs = [(p1, p2) for p1, p2 in aligned if p1 is not None and p2 is not None]
        assert len(match_pairs) == 10
        for p1, p2 in match_pairs:
            assert seq1[p1] == seq2[p2]
            assert p2 == p1 + 2

    def test_nw_point_substitutions(self):
        """Two known substitutions: 1:1 alignment, identity = (L-2)/L."""
        seq1 = "ACDEFGHIKL"
        seq2 = "ACAEFGAIKL"
        aligned, _ = needleman_wunsch(seq1, seq2)

        assert len(aligned) == 10
        for i, (p1, p2) in enumerate(aligned):
            assert p1 == i
            assert p2 == i

        identical = sum(1 for p1, p2 in aligned if seq1[p1] == seq2[p2])
        assert identical == 8

    def test_nw_internal_gap(self):
        """seq2 = seq1 with 3 internal residues deleted: one gap of length 3."""
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = seq1[:9] + seq1[12:]
        assert seq2 == "ACDEFGHIKPQRSTVWY"

        aligned, _ = needleman_wunsch(seq1, seq2)

        seq1_positions = sorted(p1 for p1, _ in aligned if p1 is not None)
        assert seq1_positions == list(range(20))

        n_gaps_on_seq2 = sum(1 for _, p2 in aligned if p2 is None)
        assert n_gaps_on_seq2 == 3

        gap_seq1_positions = sorted(p1 for p1, p2 in aligned if p2 is None)
        assert gap_seq1_positions == [9, 10, 11]

        for p1, p2 in aligned:
            if p1 is not None and p2 is not None:
                assert seq1[p1] == seq2[p2]


# ---------------------------------------------------------------------------
# Hand-built Structure fixtures
# ---------------------------------------------------------------------------


def _make_residue(
    chain_id: str, res_seq: int, res_name: str, coord: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[Residue, Atom]:
    atom = Atom(
        serial=1,
        name="CA",
        alt_loc="",
        res_name=res_name,
        chain_id=chain_id,
        res_seq=res_seq,
        insertion_code="",
        x=coord[0],
        y=coord[1],
        z=coord[2],
        occupancy=1.0,
        b_factor=0.0,
        element="C",
        is_hetatm=False,
    )
    residue = Residue(
        chain_id=chain_id,
        res_seq=res_seq,
        insertion_code="",
        res_name=res_name,
        atoms={"CA": atom},
    )
    return residue, atom


def _make_synthetic_structure(
    spec: list[tuple[str, int, str]], name: str = "synth"
) -> Structure:
    """Build a Structure where structure.residues and chain.residues are both in spec order."""
    atoms_list: list[Atom] = []
    residues_dict: dict[tuple, Residue] = {}
    chains_dict: dict[str, Chain] = {}
    for chain_id, res_seq, res_name in spec:
        residue, atom = _make_residue(chain_id, res_seq, res_name)
        atoms_list.append(atom)
        residues_dict[(chain_id, res_seq, "")] = residue
        if chain_id not in chains_dict:
            chains_dict[chain_id] = Chain(chain_id=chain_id)
        chains_dict[chain_id].residues.append(residue)

    n = len(atoms_list)
    coords = np.zeros((n, 3), dtype=np.float32)
    res_indices = np.arange(n, dtype=np.int32)
    chain_ids_arr = np.array([a.chain_id for a in atoms_list], dtype="U2")
    elements = np.array(["C"] * n, dtype="U2")
    atom_names = np.array(["CA"] * n, dtype="U4")
    res_names = np.array([a.res_name for a in atoms_list], dtype="U4")

    return Structure(
        name=name,
        atoms=atoms_list,
        residues=residues_dict,
        chains=chains_dict,
        coords=coords,
        elements=elements,
        atom_names=atom_names,
        res_indices=res_indices,
        res_names=res_names,
        chain_ids_array=chain_ids_arr,
        is_protein_mask=np.ones(n, dtype=bool),
        is_backbone_mask=np.ones(n, dtype=bool),
        is_hydrogen_mask=np.zeros(n, dtype=bool),
    )


class TestIndexMapConsistencyGate:
    def test_index_map_consistency_gate_passes(self):
        """In-order single-chain structure: build_residue_index_map succeeds and round-trips."""
        struct = _make_synthetic_structure(
            [("A", 1, "GLY"), ("A", 2, "ALA"), ("A", 3, "VAL")]
        )
        assert struct.chains["A"].sequence == "GAV"

        m = build_residue_index_map(struct, "A")
        assert m == {0: 0, 1: 1, 2: 2}

        rid_list = list(struct.residues.keys())
        reconstructed = "".join(
            {"GLY": "G", "ALA": "A", "VAL": "V"}[struct.residues[rid_list[m[k]]].res_name]
            for k in sorted(m.keys())
        )
        assert reconstructed == struct.chains["A"].sequence

    def test_index_map_consistency_gate_fires(self):
        """If Chain.residues order disagrees with structure.residues iteration, gate raises."""
        struct = _make_synthetic_structure(
            [("A", 1, "GLY"), ("A", 2, "ALA"), ("A", 3, "VAL")]
        )
        struct.chains["A"].residues = list(reversed(struct.chains["A"].residues))
        assert struct.chains["A"].sequence == "VAG"

        with pytest.raises(ValueError) as excinfo:
            build_residue_index_map(struct, "A")

        msg = str(excinfo.value)
        assert "'A'" in msg
        assert ("consistency" in msg.lower()) or ("diverges" in msg.lower())


# ---------------------------------------------------------------------------
# build_pocket_alignment: refusal and happy-path
# ---------------------------------------------------------------------------


def _make_trivial_pocket(
    struct: Structure, res_seq_set: list[int], chain_id: str = "A"
) -> BindingSite:
    """Build a BindingSite covering the given res_seqs on the given chain."""
    rid_list = list(struct.residues.keys())
    residue_indices = []
    residues = []
    res_names = []
    for engine_idx, rid in enumerate(rid_list):
        if rid[0] == chain_id and rid[1] in res_seq_set:
            residue_indices.append(engine_idx)
            residues.append(struct.residues[rid])
            res_names.append(struct.residues[rid].res_name)
    return BindingSite(
        residues=residues,
        residue_indices=residue_indices,
        atom_indices=np.array(residue_indices, dtype=np.int32),
        coords=np.zeros((len(residue_indices), 3)),
        centroid=np.zeros(3),
        res_names=res_names,
    )


class TestBuildPocketAlignment:
    def test_build_pocket_alignment_refuses_low_homology(self):
        """Two unrelated sequences (poly-Ala vs poly-Trp): refusal with populated reason."""
        target = _make_synthetic_structure(
            [("A", i + 1, "ALA") for i in range(10)], name="target"
        )
        ortholog = _make_synthetic_structure(
            [("A", i + 1, "TRP") for i in range(10)], name="ortholog"
        )
        pocket_t = _make_trivial_pocket(target, [3, 4, 5])
        pocket_o = _make_trivial_pocket(ortholog, [3, 4, 5])

        result = build_pocket_alignment(
            target, ortholog, pocket_t, pocket_o, min_global_identity=0.25
        )

        assert isinstance(result, PocketAlignment)
        assert result.refused is True
        assert result.refusal_reason is not None
        assert "identity" in result.refusal_reason.lower()
        assert result.correspondence == []
        assert result.n_confidently_paired == 0
        assert result.n_unpaired_target == result.n_pocket_target
        assert result.global_sequence_identity < 0.25
        assert result.method == "global_sequence_nw"

    def test_build_pocket_alignment_high_homology(self):
        """Two identical-sequence single-chain structures: full pocket correspondence."""
        spec = [
            ("A", 1, "ALA"), ("A", 2, "ARG"), ("A", 3, "ASN"), ("A", 4, "ASP"),
            ("A", 5, "CYS"), ("A", 6, "GLN"), ("A", 7, "GLU"), ("A", 8, "GLY"),
            ("A", 9, "HIS"), ("A", 10, "ILE"),
        ]
        target = _make_synthetic_structure(spec, name="target")
        ortholog = _make_synthetic_structure(spec, name="ortholog")
        pocket_t = _make_trivial_pocket(target, [3, 4, 5])
        pocket_o = _make_trivial_pocket(ortholog, [3, 4, 5])

        result = build_pocket_alignment(
            target, ortholog, pocket_t, pocket_o, min_global_identity=0.25
        )

        assert result.refused is False
        assert result.refusal_reason is None
        assert result.global_sequence_identity == 1.0
        assert result.n_confidently_paired == 3
        assert result.n_unpaired_target == 0
        assert set(result.correspondence) == {(2, 2), (3, 3), (4, 4)}
        assert result.method == "global_sequence_nw"
