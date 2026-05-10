"""Lennard-Jones energy computation.

Core JAX kernel for non-bonded energy evaluation.
Supports total energy, per-atom decomposition, and per-residue aggregation
(the last being essential for the causality module's selectivity attribution).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# AMBER ff14SB-like LJ parameters (sigma, epsilon) by element
# sigma in Angstroms, epsilon in kcal/mol
# These are simplified — a full implementation would use atom type assignments
ELEMENT_LJ_PARAMS: dict[str, tuple[float, float]] = {
    "C": (3.40, 0.086),
    "N": (3.25, 0.170),
    "O": (3.12, 0.210),
    "S": (3.56, 0.250),
    "H": (2.47, 0.016),
    "P": (3.74, 0.200),
    "F": (3.12, 0.061),
    "CL": (3.47, 0.265),
    "BR": (3.60, 0.320),
    "I": (3.80, 0.400),
    "SE": (3.60, 0.291),
    "FE": (2.91, 0.013),
    "ZN": (2.46, 0.013),
    "MG": (2.94, 0.013),
}

DEFAULT_LJ_PARAMS = (3.40, 0.086)  # Carbon-like defaults


def get_lj_params_arrays(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build sigma and epsilon arrays for all atoms.

    Parameters
    ----------
    elements : np.ndarray
        (N,) array of element symbols.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        sigma (N,) and epsilon (N,) arrays in float32.
    """
    n = len(elements)
    sigma = np.zeros(n, dtype=np.float32)
    epsilon = np.zeros(n, dtype=np.float32)

    for i, elem in enumerate(elements):
        s, e = ELEMENT_LJ_PARAMS.get(str(elem).strip().upper(), DEFAULT_LJ_PARAMS)
        sigma[i] = s
        epsilon[i] = e

    return sigma, epsilon


@jax.jit
def compute_lj_energy_matrix(
    dist_matrix: jnp.ndarray,
    sigma_i: jnp.ndarray,
    sigma_j: jnp.ndarray,
    epsilon_i: jnp.ndarray,
    epsilon_j: jnp.ndarray,
    mask: jnp.ndarray,
    energy_cap: float = 1000.0,
) -> jnp.ndarray:
    """Compute pairwise Lennard-Jones energy matrix.

    Uses Lorentz-Berthelot combining rules:
        sigma_ij = (sigma_i + sigma_j) / 2
        epsilon_ij = sqrt(epsilon_i * epsilon_j)

    Parameters
    ----------
    dist_matrix : jnp.ndarray
        (N, N) pairwise distance matrix.
    sigma_i, sigma_j : jnp.ndarray
        (N,) LJ sigma for each atom, broadcast to (N, 1) and (1, N).
    epsilon_i, epsilon_j : jnp.ndarray
        (N,) LJ epsilon for each atom.
    mask : jnp.ndarray
        (N, N) bool — True for non-bonded pairs to include.
    energy_cap : float
        Cap per-pair energy to prevent infinities.

    Returns
    -------
    jnp.ndarray
        (N, N) pairwise LJ energy matrix (only upper triangle meaningful,
        but symmetric). Masked pairs are 0.
    """
    # Combining rules
    sigma_ij = (sigma_i[:, None] + sigma_j[None, :]) / 2.0
    epsilon_ij = jnp.sqrt(epsilon_i[:, None] * epsilon_j[None, :])

    # LJ potential: 4 * eps * ((sigma/r)^12 - (sigma/r)^6)
    # Add small epsilon to prevent division by zero
    ratio = sigma_ij / (dist_matrix + 1e-10)
    ratio6 = ratio**6
    ratio12 = ratio6**2

    energy = 4.0 * epsilon_ij * (ratio12 - ratio6)

    # Cap extreme values
    energy = jnp.clip(energy, -energy_cap, energy_cap)

    # Apply mask (zero out bonded pairs and self)
    energy = energy * mask

    return energy


@jax.jit
def compute_total_lj_energy(energy_matrix: jnp.ndarray) -> jnp.ndarray:
    """Sum total LJ energy (counting each pair once).

    Parameters
    ----------
    energy_matrix : jnp.ndarray
        (N, N) pairwise energy matrix.

    Returns
    -------
    jnp.ndarray
        Scalar total energy.
    """
    # Sum upper triangle only to avoid double-counting
    return jnp.sum(jnp.triu(energy_matrix, k=1))


@jax.jit
def compute_per_atom_lj_energy(energy_matrix: jnp.ndarray) -> jnp.ndarray:
    """Compute per-atom LJ energy contribution.

    Each atom's energy is half the sum of its pairwise interactions
    (to split the pair energy evenly between the two atoms).

    Parameters
    ----------
    energy_matrix : jnp.ndarray
        (N, N) pairwise energy matrix.

    Returns
    -------
    jnp.ndarray
        (N,) per-atom energy.
    """
    return jnp.sum(energy_matrix, axis=1) / 2.0


def compute_per_residue_lj_energy(
    per_atom_energy: jnp.ndarray,
    res_indices: np.ndarray,
    n_residues: int,
) -> np.ndarray:
    """Aggregate per-atom LJ energy to per-residue.

    Parameters
    ----------
    per_atom_energy : jnp.ndarray
        (N,) per-atom energy from compute_per_atom_lj_energy.
    res_indices : np.ndarray
        (N,) int array mapping atoms to residue indices.
    n_residues : int
        Total number of residues.

    Returns
    -------
    np.ndarray
        (R,) per-residue energy. Positive = repulsive, negative = favorable.
    """
    per_atom_np = np.asarray(per_atom_energy)
    residue_energy = np.zeros(n_residues, dtype=np.float64)
    np.add.at(residue_energy, res_indices, per_atom_np)
    return residue_energy


def run_lj_analysis(
    dist_matrix: jnp.ndarray,
    elements: np.ndarray,
    mask: jnp.ndarray,
    res_indices: np.ndarray,
    n_residues: int,
    energy_cap: float = 1000.0,
) -> dict:
    """Run complete LJ analysis: total, per-atom, per-residue energies.

    This is the main entry point for the energy module.

    Parameters
    ----------
    dist_matrix : jnp.ndarray
        (N, N) pairwise distances.
    elements : np.ndarray
        (N,) element symbols.
    mask : jnp.ndarray
        (N, N) non-bonded mask.
    res_indices : np.ndarray
        (N,) residue indices.
    n_residues : int
        Total number of residues.
    energy_cap : float
        Cap per-pair energy.

    Returns
    -------
    dict
        Keys: 'total_energy', 'per_atom_energy', 'per_residue_energy',
              'energy_matrix', 'n_clashing_pairs' (pairs with E > 10 kcal/mol).
    """
    sigma, epsilon = get_lj_params_arrays(elements)
    sigma_jnp = jnp.array(sigma)
    epsilon_jnp = jnp.array(epsilon)

    energy_matrix = compute_lj_energy_matrix(
        dist_matrix, sigma_jnp, sigma_jnp,
        epsilon_jnp, epsilon_jnp, mask, energy_cap
    )

    total_energy = float(compute_total_lj_energy(energy_matrix))
    per_atom_energy = compute_per_atom_lj_energy(energy_matrix)
    per_residue_energy = compute_per_residue_lj_energy(
        per_atom_energy, res_indices, n_residues
    )

    # Count high-energy pairs (likely clashes)
    upper = jnp.triu(energy_matrix, k=1)
    n_hot_pairs = int(jnp.sum(upper > 10.0))

    return {
        "total_energy": total_energy,
        "per_atom_energy": np.asarray(per_atom_energy),
        "per_residue_energy": per_residue_energy,
        "energy_matrix": energy_matrix,
        "n_hot_pairs": n_hot_pairs,
    }


def per_residue_ligand_interaction_energy(
    dist_matrix: jnp.ndarray,
    elements: np.ndarray,
    mask: jnp.ndarray,
    res_indices: np.ndarray,
    n_residues: int,
    ligand_atom_indices: np.ndarray,
    energy_cap: float = 1000.0,
) -> dict:
    """Per-residue LJ energy of protein-residue ↔ ligand interactions only.

    Unlike :func:`run_lj_analysis`, which sums LJ energy over all atom
    pairs in the structure (mostly protein-protein), this function
    isolates the energy each protein residue contributes specifically
    to its interaction with the ligand. This is the central operation
    of selectivity-attribution: for a parasite target and its human
    ortholog, both with the same (or comparable) compound bound,
    compute this per-residue ligand-interaction energy on each side,
    align the pocket residues, and rank the residues that drive the
    selectivity difference.

    Parameters
    ----------
    dist_matrix, elements, mask, res_indices, n_residues, energy_cap
        Same as :func:`run_lj_analysis`. The mask is the bonded mask;
        atoms covalently bonded to each other are excluded so the
        function only sums non-bonded LJ interactions.
    ligand_atom_indices : np.ndarray
        Flat-array indices of atoms that belong to the ligand. Any
        atom not in this set is treated as protein side. The function
        sums LJ energy for pairs where one atom is in the ligand set
        and the other is not.

    Returns
    -------
    dict with keys:
        'total_ligand_interaction_kcal' : float
            Sum of all ligand-protein pairwise LJ energies.
        'per_residue_ligand_interaction_kcal' : np.ndarray, shape (n_residues,)
            For each residue, the sum of LJ energies between that
            residue's atoms and any ligand atom. Residues with no
            atoms near the ligand contribute zero.
        'n_ligand_atoms' : int
        'n_hot_protein_residues' : int
            Residues where |ligand interaction| > 1 kcal/mol — a
            simple count of "residues that meaningfully see the
            ligand".

    NON-CLAIMS:
        - LJ-only. Electrostatics, H-bonds, solvation not modeled.
        - Per-residue *attribution* means LJ sum, not free-energy
          contribution. A favorable LJ contact may sit alongside an
          unfavorable electrostatic contact not modeled here.
        - The ligand_atom_indices must correctly enumerate all and
          only the ligand atoms; cofactors (FMN, NADP), substrates
          (orotate), and crystallographic additives (sulfate, water)
          must be classified by the caller. This function does not
          guess.
    """
    if len(ligand_atom_indices) == 0:
        return {
            "total_ligand_interaction_kcal": 0.0,
            "per_residue_ligand_interaction_kcal": np.zeros(n_residues),
            "n_ligand_atoms": 0,
            "n_hot_protein_residues": 0,
        }

    sigma, epsilon = get_lj_params_arrays(elements)
    sigma_jnp = jnp.array(sigma)
    epsilon_jnp = jnp.array(epsilon)

    energy_matrix = compute_lj_energy_matrix(
        dist_matrix, sigma_jnp, sigma_jnp,
        epsilon_jnp, epsilon_jnp, mask, energy_cap
    )
    energy_np = np.asarray(energy_matrix)

    n_atoms = energy_np.shape[0]
    # Boolean mask: which atoms are ligand atoms
    is_ligand = np.zeros(n_atoms, dtype=bool)
    is_ligand[np.asarray(ligand_atom_indices, dtype=np.int64)] = True

    # For each protein atom, sum LJ energy across all ligand atoms.
    # That gives per-atom protein-side interaction energy.
    # Then aggregate to per-residue via np.add.at.
    # We don't divide by 2 here because each cross pair is counted once
    # in this asymmetric protein↔ligand setup (unlike the symmetric
    # protein↔protein per-atom function which halves to avoid double-count).
    protein_atom_ligand_interaction = energy_np[~is_ligand][:, is_ligand].sum(axis=1)
    protein_atom_indices = np.where(~is_ligand)[0]

    per_residue = np.zeros(n_residues, dtype=np.float64)
    for k, pa_idx in enumerate(protein_atom_indices):
        ri = int(res_indices[pa_idx])
        if 0 <= ri < n_residues:
            per_residue[ri] += protein_atom_ligand_interaction[k]

    total = float(per_residue.sum())
    n_hot = int(np.sum(np.abs(per_residue) > 1.0))

    return {
        "total_ligand_interaction_kcal": total,
        "per_residue_ligand_interaction_kcal": per_residue,
        "n_ligand_atoms": int(is_ligand.sum()),
        "n_hot_protein_residues": n_hot,
    }
