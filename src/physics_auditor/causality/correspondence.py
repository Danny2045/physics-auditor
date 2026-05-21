"""Sequence-anchored residue correspondence between ortholog structures.

Given two homologous protein structures (e.g., a parasite target and its
human ortholog), this module produces a residue-to-residue correspondence
that selectivity_map.py can consume as its `alignment` argument.

The previous default in compute_selectivity_map was a positional zip of
the two pocket-residue lists, which silently produces nonsense for any
pair whose pockets differ in size or whose residue numbering is offset.
This module replaces that with a global pairwise sequence alignment,
then maps each target-pocket residue to its true homologous counterpart
on the ortholog side.

NON-CLAIMS
----------
- Sequence-anchored only. No structural superposition. If two true
  homologs have undergone enough conformational/topological change that
  the sequence alignment is misleading at the pocket, the answer will
  be misleading. Future work: layer in a structural alignment fallback
  (CE / TM-align / DALI) when sequence identity is borderline.
- Assumes the two inputs are actually orthologs of the same enzyme.
  The refusal gate (min_global_identity) catches obvious mismatches —
  unrelated proteins, swapped target/ortholog — but two paralogs from
  the same superfamily can pass the gate while still being the wrong
  comparison. The caller is responsible for choosing biologically
  meaningful pairs.
- Refusal, not silent guessing. Below the identity threshold the
  function returns a PocketAlignment with refused=True and an empty
  correspondence rather than producing untrustworthy pairs. The caller
  must check `.refused` and decide what to do.
- Hard gate on engine-index consistency. If the per-chain protein
  residues are not contiguous in structure.residues' parse order
  (interleaved chains, out-of-order residues), build_residue_index_map
  raises a ValueError rather than silently producing a wrong index map.

The output's `correspondence` field is the list of
(target_engine_index, ortholog_engine_index) tuples that
compute_selectivity_map consumes via its `alignment` argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_auditor.causality.binding_site import BindingSite
from physics_auditor.core.parser import Structure

_AA_ORDER = "ARNDCQEGHILKMFPSTWYVX"

_BLOSUM62_RAW: list[list[int]] = [
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   X
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0,  0],  # A
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1],  # R
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, -1],  # N
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, -1],  # D
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -2],  # C
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, -1],  # Q
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, -1],  # E
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1],  # G
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, -1],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -1],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -1],  # L
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, -1],  # K
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -1],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -1],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2],  # P
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0],  # S
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0,  0],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -2],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -1],  # Y
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -1],  # V
    [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1],  # X
]

BLOSUM62: dict[tuple[str, str], int] = {
    (a, b): _BLOSUM62_RAW[i][j]
    for i, a in enumerate(_AA_ORDER)
    for j, b in enumerate(_AA_ORDER)
}

_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def _score(a: str, b: str) -> int:
    """BLOSUM62 score for two one-letter residues, unknowns scored as X."""
    if (a, b) in BLOSUM62:
        return BLOSUM62[(a, b)]
    a_eff = a if (a, "A") in BLOSUM62 else "X"
    b_eff = b if ("A", b) in BLOSUM62 else "X"
    return BLOSUM62[(a_eff, b_eff)]


def needleman_wunsch(
    seq1: str,
    seq2: str,
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Global pairwise sequence alignment with affine gap penalties (Gotoh).

    Pure numpy / Python — no biopython dependency. BLOSUM62 substitution
    matrix is hard-coded above. Cost of a gap of length k is
    ``gap_open + (k-1) * gap_extend`` (so a length-1 gap costs gap_open).

    Parameters
    ----------
    seq1, seq2 : str
        One-letter amino acid sequences.
    gap_open : float
        Penalty for opening a gap (applied to the first gap position).
    gap_extend : float
        Penalty for each additional gap position past the first.

    Returns
    -------
    aligned_positions : list[tuple[int | None, int | None]]
        One entry per column of the alignment, in left-to-right order.
        Each entry is ``(pos1, pos2)`` where ``pos1`` is the 0-based
        index into ``seq1`` (or None for a gap), similarly ``pos2``.
    score : float
        Final alignment score.
    """
    n, m = len(seq1), len(seq2)
    NEG_INF = -1e18

    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Ix = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Iy = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)

    M[0, 0] = 0.0
    for i in range(1, n + 1):
        Ix[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Iy[0, j] = gap_open + (j - 1) * gap_extend

    tb_M = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_Ix = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_Iy = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        a = seq1[i - 1]
        for j in range(1, m + 1):
            s = _score(a, seq2[j - 1])

            mm = M[i - 1, j - 1] + s
            mx = Ix[i - 1, j - 1] + s
            my = Iy[i - 1, j - 1] + s
            if mm >= mx and mm >= my:
                M[i, j] = mm
                tb_M[i, j] = 0
            elif mx >= my:
                M[i, j] = mx
                tb_M[i, j] = 1
            else:
                M[i, j] = my
                tb_M[i, j] = 2

            x_from_m = M[i - 1, j] + gap_open
            x_from_x = Ix[i - 1, j] + gap_extend
            if x_from_m >= x_from_x:
                Ix[i, j] = x_from_m
                tb_Ix[i, j] = 0
            else:
                Ix[i, j] = x_from_x
                tb_Ix[i, j] = 1

            y_from_m = M[i, j - 1] + gap_open
            y_from_y = Iy[i, j - 1] + gap_extend
            if y_from_m >= y_from_y:
                Iy[i, j] = y_from_m
                tb_Iy[i, j] = 0
            else:
                Iy[i, j] = y_from_y
                tb_Iy[i, j] = 2

    finals = [(float(M[n, m]), 0), (float(Ix[n, m]), 1), (float(Iy[n, m]), 2)]
    finals.sort(key=lambda v: -v[0])
    score, current = finals[0]

    aligned: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if current == 0:
            prev = int(tb_M[i, j])
            aligned.append((i - 1, j - 1))
            i -= 1
            j -= 1
            current = prev
        elif current == 1:
            if j == 0:
                prev = 1 if i > 1 else 0
            else:
                prev = int(tb_Ix[i, j])
            aligned.append((i - 1, None))
            i -= 1
            current = prev
        else:
            if i == 0:
                prev = 2 if j > 1 else 0
            else:
                prev = int(tb_Iy[i, j])
            aligned.append((None, j - 1))
            j -= 1
            current = prev

    aligned.reverse()
    return aligned, float(score)


@dataclass
class PocketAlignment:
    """Output of build_pocket_alignment.

    Attributes
    ----------
    correspondence : list[tuple[int, int]]
        (target_engine_index, ortholog_engine_index) pairs ready to pass
        as the ``alignment`` argument of compute_selectivity_map. Empty
        when refused=True.
    global_sequence_identity : float
        Fraction of identical residues over aligned (non-gap) positions
        in the global NW alignment.
    method : str
        Identifier of the alignment method (here, "global_sequence_nw").
    n_pocket_target, n_pocket_ortholog : int
        Pocket sizes on each side.
    n_confidently_paired : int
        Number of target-pocket residues paired to a real (non-gap)
        ortholog residue.
    n_unpaired_target : int
        Target-pocket residues whose homologous position on the ortholog
        is a gap (or which fall outside the chosen chain).
    refused : bool
        True if global identity fell below min_identity_threshold; the
        caller must NOT use the empty correspondence as a substitute.
    refusal_reason : str | None
        Human-readable explanation when refused, else None.
    min_identity_threshold : float
        The threshold the gate was checked against.
    """

    correspondence: list[tuple[int, int]]
    global_sequence_identity: float
    method: str
    n_pocket_target: int
    n_pocket_ortholog: int
    n_confidently_paired: int
    n_unpaired_target: int
    refused: bool
    refusal_reason: str | None
    min_identity_threshold: float


def build_residue_index_map(structure: Structure, chain_id: str) -> dict[int, int]:
    """Map protein-sequence position to engine residue index for one chain.

    Walks ``structure.residues`` in insertion order, keeps protein
    residues whose ``chain_id`` matches, and assigns 0-based positions.
    The output maps ``seq_pos -> engine_idx`` where ``engine_idx`` is
    the position of the residue in ``list(structure.residues.keys())``
    — the same integer compute_selectivity_map uses internally.

    HARD CONSISTENCY GATE
    ---------------------
    After building the map, the reconstructed one-letter sequence is
    compared against ``Chain.sequence`` for the requested chain. If
    they differ, a ValueError is raised that names the chain and the
    first mismatch position. This catches the case where protein
    residues for one chain are interleaved with another chain's
    residues in the parse order (which would break the positional
    indexing assumption). The downstream pocket alignment must not
    proceed on a structure where the gate fires.
    """
    if chain_id not in structure.chains:
        available = sorted(structure.chains.keys())
        raise ValueError(
            f"chain {chain_id!r} not found in structure {structure.name!r} "
            f"(available chains: {available})"
        )
    expected = structure.chains[chain_id].sequence

    seq_to_engine: dict[int, int] = {}
    reconstructed_chars: list[str] = []
    seq_pos = 0
    for engine_idx, rid in enumerate(structure.residues.keys()):
        residue = structure.residues[rid]
        if residue.chain_id != chain_id:
            continue
        if not residue.is_protein:
            continue
        seq_to_engine[seq_pos] = engine_idx
        reconstructed_chars.append(_THREE_TO_ONE.get(residue.res_name, "X"))
        seq_pos += 1

    reconstructed = "".join(reconstructed_chars)
    if reconstructed != expected:
        if len(reconstructed) != len(expected):
            raise ValueError(
                f"Index-map consistency gate failed for chain {chain_id!r} "
                f"in structure {structure.name!r}: reconstructed sequence "
                f"length {len(reconstructed)} != Chain.sequence length "
                f"{len(expected)}. This usually means residues for chain "
                f"{chain_id!r} are interleaved with other chains in the "
                f"parse order; the engine-index assumption is broken. "
                f"Use a structure-based correspondence instead."
            )
        for pos, (a, b) in enumerate(zip(reconstructed, expected)):
            if a != b:
                raise ValueError(
                    f"Index-map consistency gate failed for chain "
                    f"{chain_id!r} in structure {structure.name!r}: "
                    f"reconstructed sequence diverges from Chain.sequence "
                    f"at position {pos} (reconstructed={a!r}, "
                    f"expected={b!r}). This usually means residues for "
                    f"chain {chain_id!r} are interleaved with other "
                    f"chains in the parse order; the engine-index "
                    f"assumption is broken."
                )
    return seq_to_engine


def _pick_pocket_chain(structure: Structure, pocket: BindingSite) -> str:
    """Choose the chain holding the majority of pocket residues.

    Tie-breaker: the alphabetically first chain id among the tied set.
    If the pocket has no protein residues at all, fall back to the
    first protein chain in the structure.
    """
    counts: dict[str, int] = {}
    for r in pocket.residues:
        if not r.is_protein:
            continue
        counts[r.chain_id] = counts.get(r.chain_id, 0) + 1
    if counts:
        best = max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0][0])))
        return best[0]
    for cid, chain in structure.chains.items():
        if chain.is_protein:
            return cid
    raise ValueError(
        f"structure {structure.name!r} has no protein chain to anchor "
        f"sequence alignment against"
    )


def build_pocket_alignment(
    target_structure: Structure,
    ortholog_structure: Structure,
    target_pocket: BindingSite,
    ortholog_pocket: BindingSite,
    target_chain: str | None = None,
    ortholog_chain: str | None = None,
    min_global_identity: float = 0.25,
) -> PocketAlignment:
    """Build a residue correspondence between two ortholog pockets.

    Workflow
    --------
    1. Pick a chain on each side (majority of pocket residues, or the
       caller's override). Tie-broken alphabetically.
    2. Build seq_pos -> engine_idx maps via build_residue_index_map.
       This invokes the hard consistency gate (raises on interleaved
       chains).
    3. Run Needleman-Wunsch over the two chains' protein sequences.
    4. Compute global_sequence_identity = identical_aligned /
       total_aligned_non_gap_positions.
    5. If identity < min_global_identity, return a refused
       PocketAlignment (empty correspondence, refused=True). The caller
       must inspect ``.refused`` rather than treating empty as success.
    6. Otherwise, for each target-pocket engine index, look up its
       sequence position, follow the global alignment to the ortholog
       sequence position, and convert back to an ortholog engine index.
       Target residues whose homologous position is a gap are counted
       as ``n_unpaired_target`` and NOT force-paired.
    """
    t_chain = target_chain or _pick_pocket_chain(target_structure, target_pocket)
    o_chain = ortholog_chain or _pick_pocket_chain(ortholog_structure, ortholog_pocket)

    target_map = build_residue_index_map(target_structure, t_chain)
    ortholog_map = build_residue_index_map(ortholog_structure, o_chain)

    target_inv: dict[int, int] = {v: k for k, v in target_map.items()}

    target_seq = target_structure.chains[t_chain].sequence
    ortholog_seq = ortholog_structure.chains[o_chain].sequence

    aligned_positions, _ = needleman_wunsch(target_seq, ortholog_seq)

    n_aligned = 0
    n_identical = 0
    fwd: dict[int, int] = {}
    for p1, p2 in aligned_positions:
        if p1 is not None and p2 is not None:
            n_aligned += 1
            fwd[p1] = p2
            if target_seq[p1] == ortholog_seq[p2]:
                n_identical += 1
    identity = (n_identical / n_aligned) if n_aligned > 0 else 0.0

    n_pocket_t = len(target_pocket.residue_indices)
    n_pocket_o = len(ortholog_pocket.residue_indices)

    if identity < min_global_identity:
        return PocketAlignment(
            correspondence=[],
            global_sequence_identity=identity,
            method="global_sequence_nw",
            n_pocket_target=n_pocket_t,
            n_pocket_ortholog=n_pocket_o,
            n_confidently_paired=0,
            n_unpaired_target=n_pocket_t,
            refused=True,
            refusal_reason=(
                f"global identity {identity:.3f} < threshold "
                f"{min_global_identity}; sequence-anchored correspondence "
                f"not trustworthy for this pair — use structure-based "
                f"alignment or exclude"
            ),
            min_identity_threshold=min_global_identity,
        )

    correspondence: list[tuple[int, int]] = []
    n_unpaired = 0
    for t_ri in target_pocket.residue_indices:
        if t_ri not in target_inv:
            n_unpaired += 1
            continue
        t_pos = target_inv[t_ri]
        if t_pos not in fwd:
            n_unpaired += 1
            continue
        o_pos = fwd[t_pos]
        o_ri = ortholog_map.get(o_pos)
        if o_ri is None:
            n_unpaired += 1
            continue
        correspondence.append((t_ri, o_ri))

    return PocketAlignment(
        correspondence=correspondence,
        global_sequence_identity=identity,
        method="global_sequence_nw",
        n_pocket_target=n_pocket_t,
        n_pocket_ortholog=n_pocket_o,
        n_confidently_paired=len(correspondence),
        n_unpaired_target=n_unpaired,
        refused=False,
        refusal_reason=None,
        min_identity_threshold=min_global_identity,
    )
