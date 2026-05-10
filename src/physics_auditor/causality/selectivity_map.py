"""Selectivity attribution maps.

For a target and its human ortholog, both with bound ligands, compute
per-residue selectivity attribution: which specific residue positions
drive the difference in protein-ligand LJ interaction energy between
the two structures.

This is the module that makes Physics Auditor a *mechanistic explainer*
rather than only a structural validator. Given two co-crystal (or
predicted-complex) structures from AF3, Boltz-2, RFdiffusion3, or PDB,
it answers the operational question every drug-design loop needs to
answer: "which residues of this target are responsible for compound
preference over the off-target ortholog?"

NON-CLAIMS
----------
- LJ-only attribution. Electrostatics, hydrogen bonds, solvation, and
  entropy are not modeled. A residue identified here as "favorable" is
  favorable in LJ terms; the full energetic picture may differ.
- The two compared structures must have ligands in comparable poses.
  This module does not align ligand poses; it assumes the experimenter
  or upstream prediction has placed them correctly. Two ligands in
  wildly different binding modes will produce a selectivity map that
  reflects the pose difference, not the residue chemistry.
- Pocket alignment defaults to sequential pairing of pocket residues.
  Sequence-divergent pockets (insertions, deletions) require an
  explicit alignment from the caller (e.g. via structural alignment).
- Different compounds on each side (parasite-bound vs human-bound, with
  no shared chemistry) yield a selectivity map that conflates "which
  residues prefer compound A" with "which residues prefer compound B
  poorly". A clean selectivity claim requires the same compound on
  both sides, or compounds known to occupy comparable subsites.
- Free energy of binding is NOT computed. Per-residue LJ interaction
  is a structural-physics quantity, not a thermodynamic one. Reporting
  it as "binding affinity contribution" would be incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from physics_auditor.causality.binding_site import extract_binding_site
from physics_auditor.core.energy import per_residue_ligand_interaction_energy
from physics_auditor.core.geometry import compute_distance_matrix
from physics_auditor.core.parser import Structure
from physics_auditor.core.topology import build_bonded_mask, infer_bonds_from_topology


@dataclass
class SelectivityResiduePair:
    """One aligned residue pair across two structures, with ligand-interaction
    energies on each side and the delta that drives selectivity."""

    res_name_target: str
    res_name_ortholog: str
    res_seq_target: int
    res_seq_ortholog: int
    residue_index_target: int
    residue_index_ortholog: int
    energy_target_kcal: float
    energy_ortholog_kcal: float
    delta_kcal: float  # ortholog - target (positive = target preferred)


@dataclass
class SelectivityMap:
    """End-to-end selectivity attribution between a target and its ortholog.

    The convention: a positive delta means the residue's ligand-interaction
    is more favorable in the target than in the ortholog — i.e., the
    residue *contributes to* target-over-ortholog selectivity. A negative
    delta means the residue is more favorable in the ortholog.
    """

    target_name: str
    ortholog_name: str
    target_ligand_name: str
    ortholog_ligand_name: str
    n_aligned_pocket_residues: int
    total_target_interaction_kcal: float
    total_ortholog_interaction_kcal: float
    pocket_target_interaction_kcal: float
    pocket_ortholog_interaction_kcal: float
    pocket_delta_kcal: float
    residues: list[SelectivityResiduePair]

    def top_n_target_selective(self, n: int = 10) -> list[SelectivityResiduePair]:
        """Residues that drive target-over-ortholog selectivity most strongly
        (most positive delta = ortholog - target; "target prefers ligand more
        than ortholog does")."""
        return sorted(self.residues, key=lambda r: -r.delta_kcal)[:n]

    def top_n_ortholog_selective(self, n: int = 10) -> list[SelectivityResiduePair]:
        """Residues where the ortholog prefers the ligand more than target."""
        return sorted(self.residues, key=lambda r: r.delta_kcal)[:n]


def find_ligand_atoms_by_resname(
    structure: Structure, resname: str
) -> tuple[np.ndarray, int]:
    """Return flat-array indices of all atoms in residues matching `resname`.

    Returns
    -------
    indices : np.ndarray
        Flat-array atom indices for the ligand.
    n_atoms : int
    """
    indices: list[int] = []
    rid_list = list(structure.residues.keys())
    matching_residue_idx: list[int] = []
    for ri, rid in enumerate(rid_list):
        if structure.residues[rid].res_name == resname:
            matching_residue_idx.append(ri)
    if not matching_residue_idx:
        return np.array([], dtype=np.int64), 0

    matching_set = set(matching_residue_idx)
    for i in range(structure.n_atoms):
        if int(structure.res_indices[i]) in matching_set:
            indices.append(i)
    return np.array(indices, dtype=np.int64), len(indices)


def compute_selectivity_map(
    target_structure: Structure,
    target_ligand_resname: str,
    ortholog_structure: Structure,
    ortholog_ligand_resname: str,
    pocket_cutoff: float = 5.0,
    alignment: list[tuple[int, int]] | None = None,
    energy_cap: float = 1000.0,
) -> SelectivityMap:
    """End-to-end selectivity attribution between two homologous targets.

    Workflow:
        1. Locate ligand atoms in each structure by residue name.
        2. Extract pocket residues around each ligand at `pocket_cutoff`.
        3. Build bonded masks (including disulfide and OXT fixes).
        4. Compute per-residue protein↔ligand LJ interaction energy on
           both sides.
        5. Align target-pocket and ortholog-pocket residues
           (sequentially by default; pass an explicit alignment for
           divergent pockets).
        6. Return a SelectivityMap whose residues are ranked by
           selectivity-driving delta.

    Parameters
    ----------
    target_structure, ortholog_structure : Structure
        Two parsed PDB structures, both with their respective ligands
        bound. "Target" is the protein the compound is meant to engage
        (typically parasite); "ortholog" is the off-target (typically
        human).
    target_ligand_resname, ortholog_ligand_resname : str
        Three-letter HETATM residue codes identifying the bound ligand
        in each structure. Must enumerate exactly the ligand of interest
        (not cofactors like FMN, not substrates like ORO, not crystal
        additives like SO4 or HOH).
    pocket_cutoff : float
        Angstrom cutoff around the ligand for pocket extraction.
        Default 5.0 — tight enough to focus on direct contacts.
    alignment : list[tuple[int, int]] or None
        Pairs (target_residue_index, ortholog_residue_index) for pocket
        positions. If None, sequential alignment of the two pocket
        residue lists. Sequence-divergent pockets need an explicit
        alignment.
    energy_cap : float
        Per-pair LJ energy cap (kcal/mol).
    """
    # Find ligand atoms
    target_ligand_idx, n_lt = find_ligand_atoms_by_resname(
        target_structure, target_ligand_resname
    )
    ortholog_ligand_idx, n_lo = find_ligand_atoms_by_resname(
        ortholog_structure, ortholog_ligand_resname
    )
    if n_lt == 0:
        raise ValueError(
            f"No atoms found in target with res_name='{target_ligand_resname}'. "
            f"Check the HETATM residue codes present in the structure."
        )
    if n_lo == 0:
        raise ValueError(
            f"No atoms found in ortholog with res_name='{ortholog_ligand_resname}'."
        )

    # Pocket extraction on each side, around the ligand atoms
    target_ligand_coords = target_structure.coords[target_ligand_idx]
    ortholog_ligand_coords = ortholog_structure.coords[ortholog_ligand_idx]
    target_pocket = extract_binding_site(
        target_structure, target_ligand_coords, cutoff=pocket_cutoff
    )
    ortholog_pocket = extract_binding_site(
        ortholog_structure, ortholog_ligand_coords, cutoff=pocket_cutoff
    )

    # Per-residue protein↔ligand interaction energy on each side
    target_interaction = _compute_protein_ligand_interaction(
        target_structure, target_ligand_idx, energy_cap=energy_cap
    )
    ortholog_interaction = _compute_protein_ligand_interaction(
        ortholog_structure, ortholog_ligand_idx, energy_cap=energy_cap
    )

    target_per_res = target_interaction["per_residue_ligand_interaction_kcal"]
    ortholog_per_res = ortholog_interaction["per_residue_ligand_interaction_kcal"]

    # Pocket residue indices on each side (ordered)
    target_pocket_ridx = list(target_pocket.residue_indices)
    ortholog_pocket_ridx = list(ortholog_pocket.residue_indices)

    # Build alignment
    if alignment is None:
        n_align = min(len(target_pocket_ridx), len(ortholog_pocket_ridx))
        alignment = list(zip(
            target_pocket_ridx[:n_align],
            ortholog_pocket_ridx[:n_align],
        ))

    # Build SelectivityResiduePair list
    target_rid_list = list(target_structure.residues.keys())
    ortholog_rid_list = list(ortholog_structure.residues.keys())
    pairs: list[SelectivityResiduePair] = []
    for t_ri, o_ri in alignment:
        if t_ri >= len(target_rid_list) or o_ri >= len(ortholog_rid_list):
            continue
        t_rid = target_rid_list[t_ri]
        o_rid = ortholog_rid_list[o_ri]
        t_res = target_structure.residues[t_rid]
        o_res = ortholog_structure.residues[o_rid]
        e_t = float(target_per_res[t_ri])
        e_o = float(ortholog_per_res[o_ri])
        # delta = ortholog - target. Positive means target is MORE
        # favorable for the ligand than ortholog is (smaller-or-more-
        # negative number on the target side, larger on the ortholog
        # side), i.e. this residue drives target-selectivity.
        delta = e_o - e_t
        pairs.append(SelectivityResiduePair(
            res_name_target=t_res.res_name,
            res_name_ortholog=o_res.res_name,
            res_seq_target=t_res.res_seq,
            res_seq_ortholog=o_res.res_seq,
            residue_index_target=t_ri,
            residue_index_ortholog=o_ri,
            energy_target_kcal=round(e_t, 4),
            energy_ortholog_kcal=round(e_o, 4),
            delta_kcal=round(delta, 4),
        ))

    # Aggregate pocket-level energies
    target_pocket_set = set(target_pocket_ridx)
    ortholog_pocket_set = set(ortholog_pocket_ridx)
    pocket_target_e = float(sum(
        target_per_res[ri] for ri in target_pocket_set if ri < len(target_per_res)
    ))
    pocket_ortholog_e = float(sum(
        ortholog_per_res[ri] for ri in ortholog_pocket_set if ri < len(ortholog_per_res)
    ))

    return SelectivityMap(
        target_name=target_structure.name,
        ortholog_name=ortholog_structure.name,
        target_ligand_name=target_ligand_resname,
        ortholog_ligand_name=ortholog_ligand_resname,
        n_aligned_pocket_residues=len(pairs),
        total_target_interaction_kcal=round(
            target_interaction["total_ligand_interaction_kcal"], 3
        ),
        total_ortholog_interaction_kcal=round(
            ortholog_interaction["total_ligand_interaction_kcal"], 3
        ),
        pocket_target_interaction_kcal=round(pocket_target_e, 3),
        pocket_ortholog_interaction_kcal=round(pocket_ortholog_e, 3),
        pocket_delta_kcal=round(pocket_ortholog_e - pocket_target_e, 3),
        residues=pairs,
    )


def _compute_protein_ligand_interaction(
    structure: Structure,
    ligand_atom_indices: np.ndarray,
    energy_cap: float = 1000.0,
) -> dict:
    """Run the per-residue ligand-interaction energy with full topology."""
    coords = jnp.array(structure.coords)
    dist_matrix = compute_distance_matrix(coords)
    bonds = infer_bonds_from_topology(structure)
    mask = build_bonded_mask(structure.n_atoms, bonds)
    mask_jnp = jnp.array(mask)

    return per_residue_ligand_interaction_energy(
        dist_matrix=dist_matrix,
        elements=structure.elements,
        mask=mask_jnp,
        res_indices=structure.res_indices,
        n_residues=structure.n_residues,
        ligand_atom_indices=ligand_atom_indices,
        energy_cap=energy_cap,
    )


def selectivity_map_to_dict(smap: SelectivityMap, top_n: int = 10) -> dict:
    """Serialize a SelectivityMap to a JSON-ready dict."""
    return {
        "target_name": smap.target_name,
        "ortholog_name": smap.ortholog_name,
        "target_ligand_name": smap.target_ligand_name,
        "ortholog_ligand_name": smap.ortholog_ligand_name,
        "n_aligned_pocket_residues": smap.n_aligned_pocket_residues,
        "total_target_interaction_kcal": smap.total_target_interaction_kcal,
        "total_ortholog_interaction_kcal": smap.total_ortholog_interaction_kcal,
        "pocket_target_interaction_kcal": smap.pocket_target_interaction_kcal,
        "pocket_ortholog_interaction_kcal": smap.pocket_ortholog_interaction_kcal,
        "pocket_delta_kcal": smap.pocket_delta_kcal,
        "top_n_target_selective": [_pair_row(r) for r in smap.top_n_target_selective(top_n)],
        "top_n_ortholog_selective": [_pair_row(r) for r in smap.top_n_ortholog_selective(top_n)],
        "residues": [_pair_row(r) for r in smap.residues],
    }


def _pair_row(r: SelectivityResiduePair) -> dict:
    return {
        "res_name_target": r.res_name_target,
        "res_name_ortholog": r.res_name_ortholog,
        "res_seq_target": r.res_seq_target,
        "res_seq_ortholog": r.res_seq_ortholog,
        "energy_target_kcal": r.energy_target_kcal,
        "energy_ortholog_kcal": r.energy_ortholog_kcal,
        "delta_kcal": r.delta_kcal,
    }
