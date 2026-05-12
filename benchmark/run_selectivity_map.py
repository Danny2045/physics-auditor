"""Run the selectivity-map machinery on the 1D3H / 1MVS pair.

Output: per-residue protein–ligand LJ-attribution JSON dossier plus a
formatted markdown report under
``benchmark/results/selectivity_maps/``.

PENDING — true parasite-vs-human selectivity attribution
--------------------------------------------------------
The current 1D3H / 1MVS pair is a *functional paralog* comparison
between HsDHODH (1D3H, catalytic construct of Q02127) and HsDHFR
(1MVS). Both PDB SOURCE records are HOMO SAPIENS. A real
parasite-vs-human selectivity attribution requires a true parasite-
DHODH co-crystal paired with an HsDHODH co-crystal carrying a
comparable inhibitor chemotype. Candidates:

* **6Q86** if confirmed as SmDHODH (verify ORGANISM_SCIENTIFIC before use).
* **PfDHODH co-crystals 4ORM / 4RX0** as a published-orthology
  alternative — well-characterized in the antimalarial DHODH literature
  and pairable with HsDHODH co-crystals such as 1D3H or a chemotype-
  matched alternative.

Until one of those candidate parasite co-crystals is pulled into
``benchmark/structures/experimental/`` and re-run here, the per-residue
*selectivity* claim is **on hold**. What this script currently
demonstrates is that the per-residue *attribution* machinery picks up
large local chemistry differences (the HIS56 ↔ GLU30 polar-to-polar
substitution between the two human enzymes) cleanly — a sanity check
on the tool, not a selectivity-design hypothesis.

See ``benchmark/results/selectivity_maps/SELECTIVITY_FINDINGS.md`` for
the full framing correction.
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
    # 1D3H is HsDHODH (Homo sapiens dihydroorotate dehydrogenase, UniProt
    # Q02127, residues 30+) bound to A26 / teriflunomide.
    hs_dhodh_1d3h = parse_pdb(REPO_ROOT / "benchmark" / "structures" / "experimental" / "1D3H.pdb")
    # 1MVS is HsDHFR (Homo sapiens dihydrofolate reductase) bound to a
    # DTM pyrido[2,3-d]pyrimidine antifolate.
    hs_dhfr_1mvs = parse_pdb(REPO_ROOT / "benchmark" / "structures" / "experimental" / "1MVS.pdb")

    print(f"1D3H HsDHODH: {hs_dhodh_1d3h.n_atoms} atoms, {hs_dhodh_1d3h.n_residues} residues")
    print(f"1MVS HsDHFR:  {hs_dhfr_1mvs.n_atoms} atoms, {hs_dhfr_1mvs.n_residues} residues")

    smap = compute_selectivity_map(
        target_structure=hs_dhodh_1d3h,
        target_ligand_resname="A26",  # teriflunomide bound in HsDHODH (1D3H)
        ortholog_structure=hs_dhfr_1mvs,
        ortholog_ligand_resname="DTM",  # antifolate bound in HsDHFR (1MVS)
        pocket_cutoff=5.0,
    )

    print()
    print(f"  Aligned pocket residues:      {smap.n_aligned_pocket_residues}")
    print(f"  Total 1D3H interaction:       {smap.total_target_interaction_kcal:+.2f} kcal/mol")
    print(f"  Total 1MVS interaction:       {smap.total_ortholog_interaction_kcal:+.2f} kcal/mol")
    print(f"  Pocket 1D3H interaction:      {smap.pocket_target_interaction_kcal:+.2f} kcal/mol")
    print(f"  Pocket 1MVS interaction:      {smap.pocket_ortholog_interaction_kcal:+.2f} kcal/mol")
    print(f"  Pocket delta (1MVS − 1D3H):   {smap.pocket_delta_kcal:+.2f} kcal/mol")
    print()
    print("  Top 5 residues where 1D3H (HsDHODH+A26) shows more favorable LJ than 1MVS (HsDHFR+DTM):")
    for r in smap.top_n_target_selective(n=5):
        print(f"    {r.res_name_target}{r.res_seq_target:<4} → "
              f"{r.res_name_ortholog}{r.res_seq_ortholog:<4}  "
              f"Δ = {r.delta_kcal:+.3f} kcal/mol  "
              f"(1D3H={r.energy_target_kcal:+.2f}, 1MVS={r.energy_ortholog_kcal:+.2f})")
    print()
    print("  Top 3 residues where 1MVS (HsDHFR+DTM) shows more favorable LJ than 1D3H (HsDHODH+A26):")
    for r in smap.top_n_ortholog_selective(n=3):
        print(f"    {r.res_name_target}{r.res_seq_target:<4} → "
              f"{r.res_name_ortholog}{r.res_seq_ortholog:<4}  "
              f"Δ = {r.delta_kcal:+.3f} kcal/mol  "
              f"(1D3H={r.energy_target_kcal:+.2f}, 1MVS={r.energy_ortholog_kcal:+.2f})")

    # Inline claims block so consumers reading the JSON see the framing
    # without having to chase down a separate markdown file. Kept in sync
    # with SELECTIVITY_FINDINGS.md.
    claims = {
        "what_this_is": (
            "Per-residue LJ-interaction attribution between 1D3H "
            "(HsDHODH catalytic construct, UniProt Q02127 residues 30+, "
            "bound to A26/teriflunomide) and 1MVS (HsDHFR, bound to a "
            "DTM pyrido[2,3-d]pyrimidine antifolate). Both PDB SOURCE "
            "records are HOMO SAPIENS."
        ),
        "what_this_is_not": (
            "This is NOT a parasite-vs-human selectivity comparison. "
            "The two structures are two different human enzymes (DHODH "
            "and DHFR) binding two different inhibitor chemotypes."
        ),
        "supported_finding": (
            "The tool surfaces a polar-to-polar residue substitution "
            "(HIS56 ↔ GLU30 at one pair of aligned active-site positions) "
            "as the dominant per-residue LJ contributor between "
            "HsDHODH+A26 and HsDHFR+DTM. This is a sanity check on the "
            "per-residue attribution machinery, NOT a selectivity-design "
            "hypothesis."
        ),
        "pending_followup": (
            "Rerun on a true parasite-DHODH co-crystal (candidates: 6Q86 "
            "if confirmed SmDHODH; PfDHODH co-crystals 4ORM/4RX0 as a "
            "published-orthology alternative) paired with an HsDHODH "
            "co-crystal carrying a comparable inhibitor chemotype. Until "
            "then, the per-residue selectivity claim is on hold."
        ),
    }

    out_dir = REPO_ROOT / "benchmark" / "results" / "selectivity_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "HsDHODH_vs_HsDHFR_pocket_attribution.json"
    dossier = {"claims": claims, **selectivity_map_to_dict(smap, top_n=15)}
    out_path.write_text(json.dumps(dossier, indent=2))
    print(f"\n  Dossier: {out_path.relative_to(REPO_ROOT)}")
    print("  NOTE: this is a functional-paralog comparison, not a")
    print("        parasite-vs-human selectivity comparison. See")
    print("        SELECTIVITY_FINDINGS.md for the full framing.")


if __name__ == "__main__":
    main()
