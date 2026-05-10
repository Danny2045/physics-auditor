"""Run the selectivity map on the SmDHODH (1D3H) / HsDHODH (1MVS) co-crystals.

Output: per-residue selectivity-attribution JSON dossier plus a
formatted markdown report. Both are committed under
benchmark/results/selectivity_maps/.

This is the first end-to-end run of `compute_selectivity_map` on real
co-crystal data and is the result Physics Auditor was designed to
produce: which specific residues of a parasite target drive compound
preference over the human ortholog.
"""

from __future__ import annotations

import json
from pathlib import Path

from physics_auditor.causality.selectivity_map import (
    compute_selectivity_map,
    selectivity_map_to_dict,
)
from physics_auditor.core.parser import parse_pdb

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    sm_dhodh = parse_pdb(REPO_ROOT / "benchmark" / "structures" / "experimental" / "1D3H.pdb")
    hs_dhodh = parse_pdb(REPO_ROOT / "benchmark" / "structures" / "experimental" / "1MVS.pdb")

    print(f"SmDHODH (1D3H): {sm_dhodh.n_atoms} atoms, {sm_dhodh.n_residues} residues")
    print(f"HsDHODH (1MVS): {hs_dhodh.n_atoms} atoms, {hs_dhodh.n_residues} residues")

    smap = compute_selectivity_map(
        target_structure=sm_dhodh,
        target_ligand_resname="A26",  # SmDHODH-bound triazole inhibitor
        ortholog_structure=hs_dhodh,
        ortholog_ligand_resname="DTM",  # HsDHODH-bound DTM inhibitor
        pocket_cutoff=5.0,
    )

    print()
    print(f"  Aligned pocket residues:     {smap.n_aligned_pocket_residues}")
    print(f"  Total target interaction:    {smap.total_target_interaction_kcal:+.2f} kcal/mol")
    print(f"  Total ortholog interaction:  {smap.total_ortholog_interaction_kcal:+.2f} kcal/mol")
    print(f"  Pocket target interaction:   {smap.pocket_target_interaction_kcal:+.2f} kcal/mol")
    print(f"  Pocket ortholog interaction: {smap.pocket_ortholog_interaction_kcal:+.2f} kcal/mol")
    print(f"  Pocket delta (ortho-target): {smap.pocket_delta_kcal:+.2f} kcal/mol")
    print()
    print("  Top 5 SmDHODH-selective residues (parasite preferred):")
    for r in smap.top_n_target_selective(n=5):
        print(f"    {r.res_name_target}{r.res_seq_target:<4} → "
              f"{r.res_name_ortholog}{r.res_seq_ortholog:<4}  "
              f"Δ = {r.delta_kcal:+.3f} kcal/mol  "
              f"(Sm={r.energy_target_kcal:+.2f}, Hs={r.energy_ortholog_kcal:+.2f})")
    print()
    print("  Top 3 HsDHODH-selective residues (human preferred):")
    for r in smap.top_n_ortholog_selective(n=3):
        print(f"    {r.res_name_target}{r.res_seq_target:<4} → "
              f"{r.res_name_ortholog}{r.res_seq_ortholog:<4}  "
              f"Δ = {r.delta_kcal:+.3f} kcal/mol  "
              f"(Sm={r.energy_target_kcal:+.2f}, Hs={r.energy_ortholog_kcal:+.2f})")

    out_dir = REPO_ROOT / "benchmark" / "results" / "selectivity_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SmDHODH_vs_HsDHODH.json"
    out_path.write_text(json.dumps(selectivity_map_to_dict(smap, top_n=15), indent=2))
    print(f"\n  Dossier: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
