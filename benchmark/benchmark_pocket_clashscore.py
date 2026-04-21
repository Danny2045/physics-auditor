"""Binding-site physics benchmark.

Compares whole-protein vs binding-site clashscores between
experimental crystal structures and AlphaFold predictions.

The hypothesis: AlphaFold's 4-11x clashscore advantage over
experimental structures may disappear at the binding site,
where drug-design accuracy actually matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from physics_auditor.causality.binding_site import extract_binding_site
from physics_auditor.checks.clashes import (
    check_clashes,
    get_vdw_radii_array_from_elements,
)
from physics_auditor.core.geometry import compute_distance_matrix
from physics_auditor.core.parser import Structure, parse_pdb
from physics_auditor.core.topology import build_bonded_mask, infer_bonds_from_topology


@dataclass
class PocketClashResult:
    """Clash analysis restricted to binding-site atoms."""

    pocket_n_atoms: int
    pocket_n_residues: int
    pocket_n_clashes: int
    pocket_n_severe: int
    pocket_clashscore: float  # clashes per 1000 pocket atoms
    whole_n_atoms: int
    whole_n_clashes: int
    whole_clashscore: float
    pocket_sequence: str
    pocket_residue_indices: list[int]


def compute_pocket_clashscore(
    structure: Structure,
    pocket_atom_indices: np.ndarray,
    pocket_residue_indices: list[int],
    pocket_sequence: str = "",
) -> PocketClashResult:
    """Compute clashscore restricted to binding-site atoms.

    A clash is counted if at least one of the two atoms is in
    the binding pocket. This captures both intra-pocket clashes
    and pocket-to-rest clashes, which are both relevant for
    drug-design quality.

    Parameters
    ----------
    structure : Structure
        Full protein structure.
    pocket_atom_indices : np.ndarray
        Indices of atoms in the binding pocket.
    pocket_residue_indices : list[int]
        Residue indices in the pocket.
    pocket_sequence : str
        One-letter sequence of pocket residues.

    Returns
    -------
    PocketClashResult
        Pocket-restricted and whole-protein clash analysis.
    """
    # Compute full structure distance matrix and topology
    coords_jnp = jnp.array(structure.coords)
    dist_matrix = compute_distance_matrix(coords_jnp)
    bonds = infer_bonds_from_topology(structure)
    nonbonded_mask = build_bonded_mask(structure.n_atoms, bonds)

    # Whole-protein clashes
    whole_result = check_clashes(
        dist_matrix=dist_matrix,
        elements=structure.elements,
        nonbonded_mask=nonbonded_mask,
        res_indices=structure.res_indices,
        n_residues=structure.n_residues,
    )

    # Pocket-restricted clashes
    n_atoms = structure.n_atoms
    pocket_set = set(int(i) for i in pocket_atom_indices)

    # Build pocket mask: True if at least one atom in pair is in pocket
    pocket_mask = np.zeros((n_atoms, n_atoms), dtype=bool)
    for i in pocket_set:
        pocket_mask[i, :] = True
        pocket_mask[:, i] = True
    pocket_mask_jnp = jnp.array(pocket_mask)

    # Get vdW radii
    radii = get_vdw_radii_array_from_elements(structure.elements)
    radii_jnp = jnp.array(radii)
    vdw_sum = radii_jnp[:, None] + radii_jnp[None, :]
    threshold = vdw_sum - 0.4  # default tolerance
    overlap = threshold - dist_matrix

    # Only non-bonded, upper triangle, and involving pocket
    upper_mask = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
    check_mask = nonbonded_mask & upper_mask & pocket_mask_jnp

    is_clash = (overlap > 0) & check_mask
    is_severe = (overlap > 0.4) & check_mask  # default severe threshold

    pocket_n_clashes = int(jnp.sum(is_clash))
    pocket_n_severe = int(jnp.sum(is_severe))
    pocket_n_atoms = len(pocket_atom_indices)
    pocket_clashscore = (pocket_n_clashes / max(pocket_n_atoms, 1)) * 1000.0

    return PocketClashResult(
        pocket_n_atoms=pocket_n_atoms,
        pocket_n_residues=len(pocket_residue_indices),
        pocket_n_clashes=pocket_n_clashes,
        pocket_n_severe=pocket_n_severe,
        pocket_clashscore=pocket_clashscore,
        whole_n_atoms=structure.n_atoms,
        whole_n_clashes=whole_result.n_clashes,
        whole_clashscore=whole_result.clashscore,
        pocket_sequence=pocket_sequence,
        pocket_residue_indices=pocket_residue_indices,
    )


def find_ligand_coords(structure: Structure) -> np.ndarray | None:
    """Find non-water HETATM coordinates as ligand proxy.

    Returns coordinates of all non-water, non-protein heavy atoms.
    Returns None if no ligand atoms found.
    """
    ligand_indices = []
    for i, atom in enumerate(structure.atoms):
        if atom.is_hetatm and atom.res_name != "HOH" and not atom.is_hydrogen:
            ligand_indices.append(i)

    if not ligand_indices:
        return None

    return structure.coords[np.array(ligand_indices)]


def extract_pocket_by_residue_numbers(
    structure: Structure,
    target_res_seqs: set[int],
    res_seq_offset: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Extract atom indices for residues matching given sequence numbers.

    Used to apply the experimental structure's pocket definition
    to the AlphaFold prediction (which has no ligand).

    Parameters
    ----------
    structure : Structure
        Protein structure.
    target_res_seqs : set[int]
        Residue sequence numbers from the source structure.
    res_seq_offset : int
        Subtract this value from each target residue number before
        matching.  Use when the source (experimental) and target
        (predicted) structures use different numbering schemes, e.g.
        4HJO starts at residue 679 while AF-Q9BY41 starts at 1 —
        pass res_seq_offset=678 so 679→1, 680→2, etc.

    Returns
    -------
    atom_indices : np.ndarray
        Atom indices in the pocket.
    residue_indices : list[int]
        Unique residue indices in the pocket.
    """
    adjusted = {r - res_seq_offset for r in target_res_seqs}

    atom_indices = []
    residue_indices = set()

    for i, atom in enumerate(structure.atoms):
        if atom.res_seq in adjusted and atom.is_protein:
            atom_indices.append(i)
            residue_indices.add(int(structure.res_indices[i]))

    return (
        np.array(atom_indices, dtype=np.int32) if atom_indices else np.array([], dtype=np.int32),
        sorted(residue_indices),
    )


# Matched pairs: (experimental PDB path, AlphaFold path, name)
BENCHMARK_PAIRS = [
    {
        "name": "HsDHODH",
        "experimental": "benchmark/structures/experimental/1MVS.pdb",
        "predicted": "benchmark/structures/predicted/AF-Q02127-F1-model_v6.pdb",
        "description": "Human DHODH — Kira ortholog, ESM-2 blind spot pair",
    },
    {
        "name": "SmDHODH",
        "experimental": "benchmark/structures/experimental/1D3H.pdb",
        "predicted": "benchmark/structures/predicted/AF-G4VFD7-F1-model_v6.pdb",
        "description": "Parasite DHODH — 30.8x selectivity target",
    },
    {
        "name": "HsHDAC8",
        "experimental": "benchmark/structures/experimental/4HJO.pdb",
        "predicted": "benchmark/structures/predicted/AF-Q9BY41-F1-model_v6.pdb",
        "description": "Human HDAC8 — Kira ortholog for SmHDAC8",
        # 4HJO uses PDB numbering 679-960; AF-Q9BY41 uses UniProt numbering 1-377.
        # Subtract 678 so experimental residue 679 maps to AF residue 1, etc.
        "res_seq_offset": 678,
    },
    {
        "name": "Ubiquitin",
        "experimental": "benchmark/structures/experimental/1UBQ.pdb",
        "predicted": "benchmark/structures/predicted/AF-P0CG48-F1-model_v6.pdb",
        "description": "Human ubiquitin — small, no ligand (centroid fallback)",
    },
    {
        "name": "LmPTR1",
        "experimental": "benchmark/structures/experimental/2BPR.pdb",
        "predicted": "benchmark/structures/predicted/AF-Q01782-F1-model_v6.pdb",
        "description": "Leishmania pteridine reductase 1 — antiparasitic target",
        # 2BPR uses PDB numbering 381-553; AF-Q01782 uses UniProt numbering 1-288.
        # Subtract 380 so experimental residue 381 maps to AF residue 1, etc.
        "res_seq_offset": 380,
    },
    {
        "name": "HsABL1",
        "experimental": "benchmark/structures/experimental/2HYY.pdb",
        "predicted": "benchmark/structures/predicted/AF-P00519-F1-model_v6.pdb",
        "description": "Human ABL1 kinase with imatinib — cancer, tyrosine kinase",
    },
    {
        "name": "HsAromatase",
        "experimental": "benchmark/structures/experimental/3EQM.pdb",
        "predicted": "benchmark/structures/predicted/AF-P11511-F1-model_v6.pdb",
        "description": "Human aromatase/CYP19A1 with androstenedione — metalloenzyme",
    },
    {
        "name": "HsBRAF",
        "experimental": "benchmark/structures/experimental/3OG7.pdb",
        "predicted": "benchmark/structures/predicted/AF-P15056-F1-model_v6.pdb",
        "description": "Human BRAF kinase domain with inhibitor — cancer, Ser/Thr kinase",
    },
    {
        "name": "CoV2Mpro",
        "experimental": "benchmark/structures/experimental/5R82.pdb",
        "predicted": "benchmark/structures/predicted/AF-P0DTD1-F1-model_v6.pdb",
        "description": "SARS-CoV-2 main protease with inhibitor — cysteine protease",
        # 5R82 PDB residue 1 = UniProt 3264; AF fragment starts at UniProt 3267 (AF residue 1).
        # Subtract 3 so experimental residue 4 maps to AF residue 1, etc.
        "res_seq_offset": 3,
    },
    {
        "name": "HsAChE",
        "experimental": "benchmark/structures/experimental/4EY7.pdb",
        "predicted": "benchmark/structures/predicted/AF-P22303-F1-model_v6.pdb",
        "description": "Human acetylcholinesterase with donepezil — Alzheimer's, deep aromatic gorge",
        # 4EY7 PDB residue 2 = UniProt 33; AF uses UniProt numbering 1-614.
        # Subtract -31 (i.e., add 31) so PDB residue 4 maps to AF residue 35, etc.
        "res_seq_offset": -31,
    },
    {
        "name": "MmCOX2",
        "experimental": "benchmark/structures/experimental/3LN1.pdb",
        "predicted": "benchmark/structures/predicted/AF-Q05769-F1-model_v6.pdb",
        "description": "Mouse COX-2 with celecoxib — inflammation, buried hydrophobic channel",
    },
    {
        "name": "HsEGFR",
        "experimental": "benchmark/structures/experimental/1M17.pdb",
        "predicted": "benchmark/structures/predicted/AF-P00533-F1-model_v6.pdb",
        "description": "Human EGFR kinase with erlotinib — cancer, ATP-binding site",
        # 1M17 PDB residue 671 = UniProt 695; AF uses UniProt numbering 1-1210.
        # Subtract -24 (i.e., add 24) so PDB residue 672 maps to AF residue 696, etc.
        "res_seq_offset": -24,
    },
    {
        "name": "HsHMGCR",
        "experimental": "benchmark/structures/experimental/1HWL.pdb",
        "predicted": "benchmark/structures/predicted/AF-P04035-F1-model_v6.pdb",
        "description": "Human HMG-CoA reductase with statin — cardiovascular, large open site",
    },
    {
        "name": "HsDPP4",
        "experimental": "benchmark/structures/experimental/1X70.pdb",
        "predicted": "benchmark/structures/predicted/AF-P27487-F1-model_v6.pdb",
        "description": "Human DPP-4 with sitagliptin-like inhibitor — diabetes, beta-propeller",
    },
]


def run_benchmark() -> list[dict]:
    """Run the binding-site clashscore benchmark."""

    results = []

    for pair in BENCHMARK_PAIRS:
        exp_path = Path(pair["experimental"])
        af_path = Path(pair["predicted"])

        if not exp_path.exists() or not af_path.exists():
            print(f"SKIP {pair['name']}: files not found")
            continue

        print(f"\n{'='*70}")
        print(f"  {pair['name']}: {pair['description']}")
        print(f"{'='*70}")

        # Parse structures
        exp_struct = parse_pdb(exp_path)
        af_struct = parse_pdb(af_path)

        # Find ligand in experimental structure
        ligand_coords = find_ligand_coords(exp_struct)

        if ligand_coords is None:
            print(f"  WARNING: No ligand found in {exp_path.name}")
            print("  Using centroid-based pocket extraction (center of protein)")
            # Fall back to centroid-based extraction
            centroid = np.mean(exp_struct.coords, axis=0, keepdims=True)
            ligand_coords = centroid

        # Extract binding site from experimental structure
        exp_pocket = extract_binding_site(exp_struct, ligand_coords, cutoff=8.0)

        print(f"\n  Binding site: {exp_pocket.n_residues} residues, "
              f"{exp_pocket.n_atoms} atoms")
        print(f"  Pocket sequence: {exp_pocket.sequence}")

        # Get pocket residue sequence numbers for transfer to AF structure
        pocket_res_seqs = set()
        rid_list = list(exp_struct.residues.keys())
        for ridx in exp_pocket.residue_indices:
            if ridx < len(rid_list):
                rid = rid_list[ridx]
                pocket_res_seqs.add(rid[1])  # res_seq from (chain, resseq, icode)

        # Compute experimental pocket clashscore
        print(f"\n  --- Experimental ({exp_path.name}) ---")
        exp_result = compute_pocket_clashscore(
            exp_struct,
            exp_pocket.atom_indices,
            exp_pocket.residue_indices,
            exp_pocket.sequence,
        )
        print(f"  Whole protein:  {exp_result.whole_n_clashes} clashes, "
              f"clashscore={exp_result.whole_clashscore:.2f} "
              f"({exp_result.whole_n_atoms} atoms)")
        print(f"  Binding site:   {exp_result.pocket_n_clashes} clashes, "
              f"clashscore={exp_result.pocket_clashscore:.2f} "
              f"({exp_result.pocket_n_atoms} atoms)")

        # Extract same pocket from AlphaFold structure using residue numbers.
        # Apply any numbering offset declared in the pair config (e.g. HDAC8).
        res_seq_offset = pair.get("res_seq_offset", 0)
        af_atom_idx, af_res_idx = extract_pocket_by_residue_numbers(
            af_struct, pocket_res_seqs, res_seq_offset=res_seq_offset,
        )

        print(f"\n  --- AlphaFold ({af_path.name}) ---")
        if len(af_atom_idx) == 0:
            print("  WARNING: No matching pocket residues found in AF structure")
            continue

        af_result = compute_pocket_clashscore(
            af_struct,
            af_atom_idx,
            af_res_idx,
        )
        print(f"  Whole protein:  {af_result.whole_n_clashes} clashes, "
              f"clashscore={af_result.whole_clashscore:.2f} "
              f"({af_result.whole_n_atoms} atoms)")
        print(f"  Binding site:   {af_result.pocket_n_clashes} clashes, "
              f"clashscore={af_result.pocket_clashscore:.2f} "
              f"({af_result.pocket_n_atoms} atoms)")

        # Compute ratios
        whole_ratio = (exp_result.whole_clashscore / max(af_result.whole_clashscore, 0.01))
        pocket_ratio = (exp_result.pocket_clashscore / max(af_result.pocket_clashscore, 0.01))

        print("\n  --- Comparison ---")
        print(f"  Whole-protein clashscore ratio (exp/AF): {whole_ratio:.1f}x")
        print(f"  Binding-site clashscore ratio (exp/AF):  {pocket_ratio:.1f}x")

        if pocket_ratio < whole_ratio * 0.5:
            print("  >>> FINDING: AlphaFold advantage SHRINKS at the binding site")
        elif pocket_ratio > whole_ratio * 1.5:
            print("  >>> FINDING: AlphaFold advantage GROWS at the binding site")
        else:
            print("  >>> FINDING: AlphaFold advantage is SIMILAR at binding site vs whole protein")

        results.append({
            "name": pair["name"],
            "description": pair["description"],
            "experimental_pdb": exp_path.name,
            "predicted_pdb": af_path.name,
            "pocket_n_residues": exp_result.pocket_n_residues,
            "pocket_n_atoms_exp": exp_result.pocket_n_atoms,
            "pocket_n_atoms_af": len(af_atom_idx),
            "exp_whole_clashscore": round(exp_result.whole_clashscore, 2),
            "exp_pocket_clashscore": round(exp_result.pocket_clashscore, 2),
            "af_whole_clashscore": round(af_result.whole_clashscore, 2),
            "af_pocket_clashscore": round(af_result.pocket_clashscore, 2),
            "whole_ratio": round(whole_ratio, 2),
            "pocket_ratio": round(pocket_ratio, 2),
        })

    # Summary table
    if results:
        print(f"\n\n{'='*70}")
        print("  SUMMARY: Whole-Protein vs Binding-Site Clashscore")
        print(f"{'='*70}")
        print(f"  {'Protein':<12} {'Exp Whole':>10} {'Exp Pocket':>11} "
              f"{'AF Whole':>9} {'AF Pocket':>10} {'Whole Ratio':>12} {'Pocket Ratio':>13}")
        print(f"  {'-'*12} {'-'*10} {'-'*11} {'-'*9} {'-'*10} {'-'*12} {'-'*13}")
        for r in results:
            print(f"  {r['name']:<12} {r['exp_whole_clashscore']:>10.2f} "
                  f"{r['exp_pocket_clashscore']:>11.2f} "
                  f"{r['af_whole_clashscore']:>9.2f} {r['af_pocket_clashscore']:>10.2f} "
                  f"{r['whole_ratio']:>11.1f}x {r['pocket_ratio']:>12.1f}x")

        # Save results
        out_path = Path("benchmark/results/pocket_clashscore_comparison.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
