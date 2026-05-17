"""Run per-residue energy decomposition on all 14 benchmark pairs.

For each (experimental, AlphaFold) pair:
  1. Extract the binding pocket from the experimental structure
     (using the existing ligand-based pocket extraction).
  2. Decompose the LJ energy of both the experimental and AF
     structures, flagging the pocket residues.
  3. Compute the per-residue difference at aligned positions.
  4. Save the per-pair decomposition + difference as JSON to
     benchmark/results/causality/.

Output: one JSON file per pair plus a top-level summary CSV.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from physics_auditor.causality.binding_site import extract_binding_site  # noqa: E402
from physics_auditor.causality.energy_decomp import (  # noqa: E402
    decomposition_to_dict,
    difference_to_dict,
    per_residue_decomposition,
    per_residue_difference,
)
from physics_auditor.core.parser import parse_pdb  # noqa: E402

# Reuse the benchmark pair table by importing it
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
from benchmark_pocket_clashscore import (  # noqa: E402
    BENCHMARK_PAIRS,
    extract_pocket_by_residue_numbers,
    find_ligand_coords,
)


def run_one_pair(pair: dict, max_atoms: int = 4500) -> dict | None:
    """Run causality decomposition on one benchmark pair.

    Pairs whose structure has more than `max_atoms` heavy atoms are
    skipped to avoid OOM on the dense LJ energy matrix (memory grows
    as O(n²)). The over-broad pocket cutoff (8.0Å around all ligand
    atoms) produces some pockets >100 residues for ligands like
    celecoxib in MmCOX2; tightening the cutoff is queued separately.

    Returns a summary row, or None on skip / failure.
    """
    name = pair["name"]
    exp_path = REPO_ROOT / pair["experimental"]
    af_path = REPO_ROOT / pair["predicted"]
    if not exp_path.exists() or not af_path.exists():
        print(f"  [skip] {name}: files not found")
        return None

    print(f"\n=== {name}: {pair['description']} ===")

    exp_struct = parse_pdb(exp_path)
    af_struct = parse_pdb(af_path)

    if exp_struct.n_atoms > max_atoms or af_struct.n_atoms > max_atoms:
        print(f"  [skip] {name}: exceeds atom budget "
              f"(exp={exp_struct.n_atoms}, af={af_struct.n_atoms}, max={max_atoms}). "
              f"Causality layer needs O(n^2) memory; tighten benchmark pocket cutoff "
              f"(see methodology caveats in preprint).")
        return None

    # Pocket from experimental ligand (or centroid for ligand-free)
    ligand_coords = find_ligand_coords(exp_struct)
    if ligand_coords is None:
        centroid = np.mean(exp_struct.coords, axis=0, keepdims=True)
        ligand_coords = centroid
        print("  [info] ligand-free; using centroid pocket fallback")

    exp_pocket = extract_binding_site(exp_struct, ligand_coords, cutoff=8.0)
    print(f"  pocket: {exp_pocket.n_residues} residues, {exp_pocket.n_atoms} atoms")

    # Map pocket residues from experimental to AF using residue-number transfer
    pocket_res_seqs = set()
    rid_list = list(exp_struct.residues.keys())
    for ridx in exp_pocket.residue_indices:
        if ridx < len(rid_list):
            pocket_res_seqs.add(rid_list[ridx][1])

    res_seq_offset = pair.get("res_seq_offset", 0)
    af_atom_idx, af_pocket_residue_indices = extract_pocket_by_residue_numbers(
        af_struct, pocket_res_seqs, res_seq_offset=res_seq_offset
    )

    # Per-residue decomposition for both
    decomp_exp = per_residue_decomposition(
        exp_struct, pocket_residue_indices=exp_pocket.residue_indices
    )
    decomp_af = per_residue_decomposition(
        af_struct, pocket_residue_indices=af_pocket_residue_indices
    )

    # Build sequential pocket-only alignment: aligned position k maps
    # the k-th pocket residue in exp to the k-th pocket residue in AF.
    # This is approximate — when residue numbering differs across structures
    # (e.g. HsHDAC8 with 678 offset) the residues at the same offset *should*
    # be the same residue in sequence, but the simple "k-th in pocket list"
    # alignment can drift if the pocket sets aren't symmetric.
    n_align = min(len(exp_pocket.residue_indices), len(af_pocket_residue_indices))
    alignment = list(zip(
        exp_pocket.residue_indices[:n_align],
        af_pocket_residue_indices[:n_align],
    ))

    diff = per_residue_difference(decomp_exp, decomp_af, alignment=alignment)
    print(f"  pocket_delta_kcal: {diff.pocket_delta_kcal:+.2f}  "
          f"(exp_pocket={decomp_exp.pocket_energy_kcal:.1f}, "
          f"af_pocket={decomp_af.pocket_energy_kcal:.1f})")

    # Top-3 unfavorable in AF (where the AF apo is paying a strain cost
    # that exp didn't, or — more often given AF apo relaxes — vice versa).
    top_unfav = diff.top_n_unfavorable_in_b(n=3)
    if top_unfav:
        print("  top unfavorable in AF (vs exp):")
        for r in top_unfav:
            print(f"    res_idx_a={r.residue_index_a:4d} ({r.res_name_a}) "
                  f"-> res_idx_b={r.residue_index_b:4d} ({r.res_name_b}): "
                  f"{r.energy_a_kcal:+.2f} -> {r.energy_b_kcal:+.2f} "
                  f"(Δ={r.delta_kcal:+.2f})")
    top_fav = diff.top_n_favorable_in_b(n=3)
    if top_fav:
        print("  top favorable in AF (vs exp):")
        for r in top_fav:
            print(f"    res_idx_a={r.residue_index_a:4d} ({r.res_name_a}) "
                  f"-> res_idx_b={r.residue_index_b:4d} ({r.res_name_b}): "
                  f"{r.energy_a_kcal:+.2f} -> {r.energy_b_kcal:+.2f} "
                  f"(Δ={r.delta_kcal:+.2f})")

    # Persist
    out_dir = REPO_ROOT / "benchmark" / "results" / "causality"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "pair": {
            "name": name,
            "description": pair["description"],
            "experimental_pdb": exp_path.name,
            "predicted_pdb": af_path.name,
            "res_seq_offset": res_seq_offset,
            "pocket_residues_exp": exp_pocket.n_residues,
            "pocket_residues_af": len(af_pocket_residue_indices),
        },
        "experimental_decomposition": decomposition_to_dict(decomp_exp),
        "alphafold_decomposition": decomposition_to_dict(decomp_af),
        "difference": difference_to_dict(diff, top_n=10),
    }
    out_path = out_dir / f"{name}_causality.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  written: {out_path.relative_to(REPO_ROOT)}")

    return {
        "name": name,
        "description": pair["description"],
        "exp_total_kcal": round(decomp_exp.total_energy_kcal, 2),
        "af_total_kcal": round(decomp_af.total_energy_kcal, 2),
        "exp_pocket_kcal": round(decomp_exp.pocket_energy_kcal, 2),
        "af_pocket_kcal": round(decomp_af.pocket_energy_kcal, 2),
        "pocket_delta_kcal": round(diff.pocket_delta_kcal, 2),
        "n_pocket_aligned": diff.n_pocket_aligned,
        "top_unfav_res_in_af": (
            f"{top_unfav[0].res_name_b}{top_unfav[0].residue_index_b}:{top_unfav[0].delta_kcal:+.1f}"
            if top_unfav else ""
        ),
        "top_fav_res_in_af": (
            f"{top_fav[0].res_name_b}{top_fav[0].residue_index_b}:{top_fav[0].delta_kcal:+.1f}"
            if top_fav else ""
        ),
    }


def main():
    summary_rows = []
    for pair in BENCHMARK_PAIRS:
        row = run_one_pair(pair)
        if row is not None:
            summary_rows.append(row)

    # Write summary CSV
    out_dir = REPO_ROOT / "benchmark" / "results" / "causality"
    summary_csv = out_dir / "causality_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f"\n{'='*70}")
        print(f"  Causality summary: {summary_csv.relative_to(REPO_ROOT)}")
        print(f"{'='*70}")
        # Pretty print
        print(f"{'Name':<14} {'Pocket Δ (AF-exp)':>20} {'Top unfav in AF':>22} {'Top fav in AF':>22}")
        print("-" * 78)
        for r in summary_rows:
            print(f"{r['name']:<14} {r['pocket_delta_kcal']:>+20.2f} "
                  f"{r['top_unfav_res_in_af']:>22} {r['top_fav_res_in_af']:>22}")


if __name__ == "__main__":
    main()
