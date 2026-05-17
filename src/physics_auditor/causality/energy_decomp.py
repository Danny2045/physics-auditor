"""Per-residue energy decomposition with binding-site focus.

Decomposes Lennard-Jones energy to per-residue contributions and
identifies which residues drive the binding-site energy difference
between two structures (e.g., an experimental holo crystal and an
AlphaFold apo prediction of the same protein).

This module is the foundation of the causality layer. It does not
predict; it attributes. Given the LJ energy matrix already computed
by `physics_auditor.core.energy`, it answers two operational questions:

1. Which residues carry the favorable / unfavorable interactions in
   this structure? (`per_residue_decomposition`)
2. For two homologous or matched structures, which residues are
   responsible for the difference in binding-site energy?
   (`per_residue_difference`)

NON-CLAIMS
----------
- Per-residue energy is the sum of pairwise LJ interactions involving
  each atom of that residue, halved at the pair level to avoid
  double-counting. It is not a free energy, not a binding affinity,
  and not a stability score.
- Per-residue *differences* between two structures are valid only
  when residues are properly aligned. The default sequential alignment
  works for closely homologous structures (parasite vs human ortholog
  with no insertions/deletions in the pocket region) and breaks down
  otherwise. Pass an explicit alignment for non-trivial cases.
- This module quantifies *what* differs at the residue level, not
  *why* (whether the cause is rotamer flip, backbone strain, or
  ligand-induced fit). That separation is intentional.
- LJ-only attribution. Electrostatics, hydrogen bonds, and solvation
  are not modeled. A residue that looks energetically "favorable"
  here may still be unfavorable in a full force field.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from physics_auditor.core.energy import run_lj_analysis
from physics_auditor.core.geometry import compute_distance_matrix
from physics_auditor.core.parser import Structure
from physics_auditor.core.topology import build_bonded_mask, infer_bonds_from_topology


@dataclass
class ResidueEnergy:
    """Energy attribution for a single residue."""

    residue_index: int
    res_name: str
    chain_id: str
    res_seq: int
    energy_kcal: float
    n_atoms: int
    is_in_pocket: bool
    n_hot_pairs: int


@dataclass
class StructureDecomposition:
    """Full per-residue energy decomposition for one structure."""

    structure_name: str
    total_energy_kcal: float
    residues: list[ResidueEnergy]
    pocket_residue_indices: list[int]

    @property
    def pocket_energy_kcal(self) -> float:
        """Sum of energies for pocket residues only."""
        return sum(r.energy_kcal for r in self.residues if r.is_in_pocket)

    @property
    def n_pocket_residues(self) -> int:
        return sum(1 for r in self.residues if r.is_in_pocket)


@dataclass
class ResidueDifference:
    """One residue's energy difference between two structures."""

    residue_index_a: int
    residue_index_b: int
    res_name_a: str
    res_name_b: str
    energy_a_kcal: float
    energy_b_kcal: float
    delta_kcal: float
    is_in_pocket_a: bool
    is_in_pocket_b: bool


@dataclass
class DecompositionDifference:
    """Side-by-side comparison of two decompositions."""

    name_a: str
    name_b: str
    total_delta_kcal: float
    pocket_delta_kcal: float
    residues: list[ResidueDifference]
    n_aligned: int
    n_pocket_aligned: int

    def top_n_by_abs_delta(self, n: int = 10) -> list[ResidueDifference]:
        """Return the n residues with the largest |delta|, sorted descending."""
        return sorted(self.residues, key=lambda r: abs(r.delta_kcal), reverse=True)[:n]

    def top_n_unfavorable_in_b(self, n: int = 10) -> list[ResidueDifference]:
        """Residues most unfavorable in B relative to A (largest positive delta)."""
        return sorted(self.residues, key=lambda r: r.delta_kcal, reverse=True)[:n]

    def top_n_favorable_in_b(self, n: int = 10) -> list[ResidueDifference]:
        """Residues most favorable in B relative to A (largest negative delta)."""
        return sorted(self.residues, key=lambda r: r.delta_kcal)[:n]


def per_residue_decomposition(
    structure: Structure,
    pocket_residue_indices: list[int] | None = None,
    energy_cap: float = 1000.0,
) -> StructureDecomposition:
    """Decompose a structure's LJ energy to per-residue contributions.

    Parameters
    ----------
    structure : Structure
        Parsed protein structure.
    pocket_residue_indices : list[int] or None
        Indices of pocket residues. If None, no pocket flagging is done
        (all residues marked is_in_pocket=False, pocket_energy_kcal=0).
    energy_cap : float
        Cap on per-pair LJ energy to prevent infinities at near-zero
        distances.

    Returns
    -------
    StructureDecomposition
        Per-residue energy attribution. Residues are returned in the
        order they appear in the structure.
    """
    coords = jnp.array(structure.coords)
    dist_matrix = compute_distance_matrix(coords)
    bonds = infer_bonds_from_topology(structure)
    mask = build_bonded_mask(structure.n_atoms, bonds)
    mask_jnp = jnp.array(mask)

    lj_result = run_lj_analysis(
        dist_matrix=dist_matrix,
        elements=structure.elements,
        mask=mask_jnp,
        res_indices=structure.res_indices,
        n_residues=structure.n_residues,
        energy_cap=energy_cap,
    )

    per_residue_e = lj_result["per_residue_energy"]
    energy_matrix = np.asarray(lj_result["energy_matrix"])

    # Hot-pair count per residue: each pair > 10 kcal/mol contributes to
    # both atoms' residues (no within-residue double-count).
    upper = np.triu(energy_matrix, k=1)
    hot_mask = upper > 10.0
    hot_pair_indices = np.argwhere(hot_mask)
    per_res_hot_pairs = np.zeros(structure.n_residues, dtype=np.int32)
    for i, j in hot_pair_indices:
        ri = int(structure.res_indices[i])
        rj = int(structure.res_indices[j])
        per_res_hot_pairs[ri] += 1
        if rj != ri:
            per_res_hot_pairs[rj] += 1

    atom_counts = np.bincount(
        structure.res_indices.astype(np.int64), minlength=structure.n_residues
    )

    pocket_set = set(int(i) for i in (pocket_residue_indices or []))

    rid_list = list(structure.residues.keys())
    residues_out: list[ResidueEnergy] = []
    for ridx, rid in enumerate(rid_list):
        residue = structure.residues[rid]
        residues_out.append(ResidueEnergy(
            residue_index=ridx,
            res_name=residue.res_name,
            chain_id=residue.chain_id,
            res_seq=residue.res_seq,
            energy_kcal=float(per_residue_e[ridx]),
            n_atoms=int(atom_counts[ridx]) if ridx < len(atom_counts) else 0,
            is_in_pocket=ridx in pocket_set,
            n_hot_pairs=int(per_res_hot_pairs[ridx]),
        ))

    return StructureDecomposition(
        structure_name=structure.name,
        total_energy_kcal=float(lj_result["total_energy"]),
        residues=residues_out,
        pocket_residue_indices=sorted(pocket_set),
    )


def per_residue_difference(
    decomp_a: StructureDecomposition,
    decomp_b: StructureDecomposition,
    alignment: list[tuple[int, int]] | None = None,
) -> DecompositionDifference:
    """Compare two per-residue decompositions at aligned positions.

    Parameters
    ----------
    decomp_a, decomp_b : StructureDecomposition
        Decompositions of two structures (e.g., experimental holo and
        AF apo of the same protein, or a parasite target and its human
        ortholog).
    alignment : list[tuple[int, int]] or None
        Pairs (residue_index_in_a, residue_index_in_b) defining
        positional correspondence. If None, uses sequential alignment
        from index 0 up to min(len_a, len_b). Sequential alignment is
        only correct when both structures have no insertions/deletions
        relative to each other in the aligned region.

    Returns
    -------
    DecompositionDifference
        Per-residue deltas and aggregates.
    """
    if alignment is None:
        n = min(len(decomp_a.residues), len(decomp_b.residues))
        alignment = [(i, i) for i in range(n)]

    res_a_by_idx = {r.residue_index: r for r in decomp_a.residues}
    res_b_by_idx = {r.residue_index: r for r in decomp_b.residues}

    diffs: list[ResidueDifference] = []
    n_pocket_aligned = 0
    pocket_delta_total = 0.0

    for idx_a, idx_b in alignment:
        if idx_a not in res_a_by_idx or idx_b not in res_b_by_idx:
            continue
        ra = res_a_by_idx[idx_a]
        rb = res_b_by_idx[idx_b]
        delta = rb.energy_kcal - ra.energy_kcal
        rd = ResidueDifference(
            residue_index_a=ra.residue_index,
            residue_index_b=rb.residue_index,
            res_name_a=ra.res_name,
            res_name_b=rb.res_name,
            energy_a_kcal=ra.energy_kcal,
            energy_b_kcal=rb.energy_kcal,
            delta_kcal=delta,
            is_in_pocket_a=ra.is_in_pocket,
            is_in_pocket_b=rb.is_in_pocket,
        )
        diffs.append(rd)
        if ra.is_in_pocket or rb.is_in_pocket:
            n_pocket_aligned += 1
            pocket_delta_total += delta

    return DecompositionDifference(
        name_a=decomp_a.structure_name,
        name_b=decomp_b.structure_name,
        total_delta_kcal=decomp_b.total_energy_kcal - decomp_a.total_energy_kcal,
        pocket_delta_kcal=pocket_delta_total,
        residues=diffs,
        n_aligned=len(diffs),
        n_pocket_aligned=n_pocket_aligned,
    )


def decomposition_to_dict(decomp: StructureDecomposition) -> dict:
    """Serialize a decomposition to a JSON-ready dict."""
    return {
        "structure_name": decomp.structure_name,
        "total_energy_kcal": round(decomp.total_energy_kcal, 3),
        "pocket_energy_kcal": round(decomp.pocket_energy_kcal, 3),
        "n_residues": len(decomp.residues),
        "n_pocket_residues": decomp.n_pocket_residues,
        "pocket_residue_indices": decomp.pocket_residue_indices,
        "residues": [
            {
                "residue_index": r.residue_index,
                "res_name": r.res_name,
                "chain_id": r.chain_id,
                "res_seq": r.res_seq,
                "energy_kcal": round(r.energy_kcal, 4),
                "n_atoms": r.n_atoms,
                "is_in_pocket": r.is_in_pocket,
                "n_hot_pairs": r.n_hot_pairs,
            }
            for r in decomp.residues
        ],
    }


def difference_to_dict(diff: DecompositionDifference, top_n: int = 10) -> dict:
    """Serialize a difference to a JSON-ready dict, with top-N tables.

    The full per-residue list is included; the top-N tables are
    convenience views for human inspection.
    """
    return {
        "name_a": diff.name_a,
        "name_b": diff.name_b,
        "total_delta_kcal": round(diff.total_delta_kcal, 3),
        "pocket_delta_kcal": round(diff.pocket_delta_kcal, 3),
        "n_aligned": diff.n_aligned,
        "n_pocket_aligned": diff.n_pocket_aligned,
        "top_n_by_abs_delta": [_diff_row(r) for r in diff.top_n_by_abs_delta(top_n)],
        "top_n_unfavorable_in_b": [_diff_row(r) for r in diff.top_n_unfavorable_in_b(top_n)],
        "top_n_favorable_in_b": [_diff_row(r) for r in diff.top_n_favorable_in_b(top_n)],
        "residues": [_diff_row(r) for r in diff.residues],
    }


def _diff_row(r: ResidueDifference) -> dict:
    return {
        "res_name_a": r.res_name_a,
        "res_name_b": r.res_name_b,
        "residue_index_a": r.residue_index_a,
        "residue_index_b": r.residue_index_b,
        "energy_a_kcal": round(r.energy_a_kcal, 4),
        "energy_b_kcal": round(r.energy_b_kcal, 4),
        "delta_kcal": round(r.delta_kcal, 4),
        "is_in_pocket_a": r.is_in_pocket_a,
        "is_in_pocket_b": r.is_in_pocket_b,
    }
