"""W-3 confirmatory selectivity run on the WT-PfLDH-vs-HsLDH-A pair.

Computes the PfLDH (1LDG, wild-type, loop ordered) vs HsLDH-A (1I10) selectivity
map, chemotype-matched on OXM (oxamate) + NAI (NADH form), and scores the
result against the W-2 pre-registered outcome map. The outcome is determined
by whether the engine flags the active-site tryptophan (located in W-3
STRUCTURALLY — not by residue number).

Outputs
-------
- ``benchmark/results/selectivity_maps/W107F_MECHANISTIC_RESULT.json``
  dossier mirroring the B-2 blind prediction's JSON shape, plus the
  pocket-completeness gate verdicts and the structural-TRP location.
- console summary with both correspondence methods, the convergence check,
  the full ranked selectivity map, and the tryptophan's row + rank.

Run:
    python benchmark/run_w107f_selectivity.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from physics_auditor.causality.binding_site import extract_binding_site
from physics_auditor.causality.correspondence import build_pocket_alignment
from physics_auditor.causality.pocket_completeness import pocket_completeness_gate
from physics_auditor.causality.selectivity_map import (
    compute_selectivity_map,
    selectivity_map_to_dict,
)
from physics_auditor.causality.superposition import (
    build_anchor_pairing,
    kabsch_superpose,
    superpose_and_correspond,
)
from physics_auditor.core.parser import parse_pdb_string
from physics_auditor.utils.pdb_provenance import verify_pdb_source

REPO_ROOT = Path(__file__).resolve().parent.parent
STRUCT_DIR = REPO_ROOT / "benchmark" / "structures" / "experimental"
OUT_DIR = REPO_ROOT / "benchmark" / "results" / "selectivity_maps"
OUT_JSON = OUT_DIR / "W107F_MECHANISTIC_RESULT.json"


def load_chain_restricted(pdb_id: str, chain_id: str):
    """Load a PDB, then re-parse keeping only ATOM/HETATM/SEQRES on `chain_id`.

    Mirrors the LDH_BLIND_TEST_SETUP option-2 pre-processing route: avoid
    the duplicate-atom-name guard in ``extract_cofactor_atoms`` triggered by
    multi-chain biological assemblies (1I10 is an 8-chain assembly with
    NAI in every chain).
    """
    raw = (STRUCT_DIR / f"{pdb_id}.pdb").read_text()
    kept = []
    for line in raw.splitlines():
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM"):
            if len(line) > 21 and line[21] == chain_id:
                kept.append(line)
        elif rec == "SEQRES":
            if len(line) > 11 and line[11] == chain_id:
                kept.append(line)
        elif rec == "ENDMDL":
            break
        # discard everything else; we already verified provenance upstream
    return parse_pdb_string("\n".join(kept), name=f"{pdb_id}_chain{chain_id}")


def structural_trp_in_pocket(structure, ligand_resname, chain_id, cutoff=5.0):
    """Find the TRP residue(s) within `cutoff` of the substrate ligand on `chain_id`.

    Identity criterion is 3D proximity, NOT residue number.
    Returns list of dicts sorted ascending by min heavy-atom distance to ligand.
    """
    lig_coords = np.array(
        [(a.x, a.y, a.z) for a in structure.atoms
         if a.res_name == ligand_resname and a.chain_id == chain_id
         and not a.is_hydrogen and a.is_hetatm],
        dtype=np.float32,
    )
    if lig_coords.size == 0:
        return []
    out = []
    for r in structure.chains[chain_id].residues:
        if r.res_name != "TRP":
            continue
        heavy = np.array(
            [(a.x, a.y, a.z) for a in r.atoms.values() if not a.is_hydrogen],
            dtype=np.float32,
        )
        if heavy.size == 0:
            continue
        d = float(np.min(np.linalg.norm(lig_coords[None, :, :] - heavy[:, None, :], axis=-1)))
        if d <= cutoff:
            out.append({"res_seq": r.res_seq, "res_name": r.res_name, "min_dist_to_ligand": d})
    out.sort(key=lambda x: x["min_dist_to_ligand"])
    return out


def main():
    print("=" * 76)
    print("W-3 selectivity run: WT PfLDH (1LDG, chain A) vs HsLDH-A (1I10, chain A)")
    print("    chemotype-matched on OXM (substrate) + NAI (cofactor)")
    print("=" * 76)
    print()

    # ---- Provenance gate (live, before parse) ----
    print("[STEP W-3a-1] Provenance gate")
    pf_path = STRUCT_DIR / "1LDG.pdb"
    hs_path = STRUCT_DIR / "1I10.pdb"
    verify_pdb_source(pf_path, "PLASMODIUM FALCIPARUM")
    print(f"  PASS: {pf_path.name} -> PLASMODIUM FALCIPARUM")
    verify_pdb_source(hs_path, "HOMO SAPIENS")
    print(f"  PASS: {hs_path.name} -> HOMO SAPIENS")
    print()

    # ---- Load structures, chain-restricted ----
    print("[STEP W-3a-2] Load chain-A structures")
    pf = load_chain_restricted("1LDG", "A")
    hs = load_chain_restricted("1I10", "A")
    print(f"  1LDG chain A: atoms={pf.n_atoms}, residues={pf.n_residues}, "
          f"SEQRES len={len(pf.seqres.get('A', []))}")
    print(f"  1I10 chain A: atoms={hs.n_atoms}, residues={hs.n_residues}, "
          f"SEQRES len={len(hs.seqres.get('A', []))}")
    print()

    # ---- Pocket-completeness gate on BOTH (the new gate's first
    # ----  load-bearing use guarding a real selectivity run)
    print("[STEP W-3a-3] Pocket-completeness gate on BOTH structures")
    pf_gate = pocket_completeness_gate(pf, ligand_resname="OXM", cutoff=5.0)
    hs_gate = pocket_completeness_gate(hs, ligand_resname="OXM", cutoff=5.0)
    print(f"  1LDG (PfLDH): {pf_gate.summary()}")
    print(f"  1I10 (HsLDH-A): {hs_gate.summary()}")
    if pf_gate.verdict != "COMPLETE" or hs_gate.verdict != "COMPLETE":
        print()
        print("FATAL: pocket-completeness gate did NOT pass on both structures.")
        print("Do not run selectivity computation on a silently incomplete pocket.")
        sys.exit(1)
    print()

    # ---- Pocket extraction for the comparability gate ----
    print("[STEP W-3a-4] Comparability gate (pocket sizes, ligand counts, "
          "NAI-anchored Kabsch RMSD, shared subsite)")
    pf_lig = np.array([(a.x, a.y, a.z) for a in pf.atoms
                       if a.res_name == "OXM" and not a.is_hydrogen and a.is_hetatm],
                      dtype=np.float32)
    hs_lig = np.array([(a.x, a.y, a.z) for a in hs.atoms
                       if a.res_name == "OXM" and not a.is_hydrogen and a.is_hetatm],
                      dtype=np.float32)
    pf_pocket = extract_binding_site(pf, pf_lig, cutoff=5.0, protein_only=True)
    hs_pocket = extract_binding_site(hs, hs_lig, cutoff=5.0, protein_only=True)
    print(f"  1LDG OXM atoms: {len(pf_lig)}    pocket residues (≤5 Å): "
          f"{pf_pocket.n_residues}")
    print(f"  1I10 OXM atoms: {len(hs_lig)}    pocket residues (≤5 Å): "
          f"{hs_pocket.n_residues}")
    pf_pocket_ids = [(r.chain_id, r.res_name, r.res_seq) for r in pf_pocket.residues]
    hs_pocket_ids = [(r.chain_id, r.res_name, r.res_seq) for r in hs_pocket.residues]
    print(f"  1LDG pocket: {[f'{n}{s}' for _,n,s in pf_pocket_ids]}")
    print(f"  1I10 pocket: {[f'{n}{s}' for _,n,s in hs_pocket_ids]}")

    # NAI Kabsch anchor
    P, Q, shared_names = build_anchor_pairing(pf, hs, "NAI")
    R, t, anchor_rmsd = kabsch_superpose(P, Q)
    print(f"  NAI shared atom-name pairs: {len(shared_names)} / 44")
    print(f"  NAI Kabsch anchor RMSD: {anchor_rmsd:.4f} Å")
    comparability_pass = (
        anchor_rmsd <= 1.0
        and abs(pf_pocket.n_residues - hs_pocket.n_residues) <= 4
        and len(shared_names) >= 3
    )
    print(f"  comparability gate: {'PASS' if comparability_pass else 'FAIL'}")
    if not comparability_pass:
        print()
        print("FATAL: comparability gate did NOT pass on this pair.")
        sys.exit(1)
    print()

    # ---- Sequence correspondence ----
    print("[STEP W-3b-1] Sequence correspondence (Needleman-Wunsch, identity gate)")
    seq_align = build_pocket_alignment(
        pf, hs, pf_pocket, hs_pocket,
        target_chain="A", ortholog_chain="A", min_global_identity=0.25,
    )
    print(f"  global sequence identity: {seq_align.global_sequence_identity:.4f}")
    print(f"  method: {seq_align.method}")
    print(f"  refused? {seq_align.refused}")
    print(f"  n_pocket_target/ortholog: {seq_align.n_pocket_target} / "
          f"{seq_align.n_pocket_ortholog}")
    print(f"  n_confidently_paired: {seq_align.n_confidently_paired}")
    print(f"  n_unpaired_target: {seq_align.n_unpaired_target}")
    if seq_align.refused:
        print(f"  refusal reason: {seq_align.refusal_reason}")
        print()
        print("FATAL: sequence-correspondence identity gate refused this pair.")
        sys.exit(1)
    print()

    # ---- Structural correspondence ----
    print("[STEP W-3b-2] Structural correspondence (NAI-anchored Kabsch)")
    sup = superpose_and_correspond(
        pf, hs, pf_pocket, hs_pocket,
        anchor_resname="NAI", match_cutoff=4.0,
        target_chain="A", ortholog_chain="A",
    )
    print(f"  method: {sup.method}, anchor: {sup.anchor_resname}, "
          f"n_anchor_atoms: {sup.n_anchor_atoms}")
    print(f"  anchor_rmsd: {sup.anchor_rmsd:.4f} Å")
    print(f"  n_spatially_matched: {sup.n_spatially_matched} / "
          f"target pocket size {sup.n_pocket_target}")
    print(f"  mean_match_distance: {sup.mean_match_distance:.4f} Å")
    print(f"  n_unmatched_target: {sup.n_unmatched_target}")
    print()

    # ---- Correspondence convergence ----
    print("[STEP W-3b-3] Convergence: do sequence and structural methods agree?")
    seq_pairs = set(seq_align.correspondence)
    sup_pairs = set(sup.correspondence)
    same = seq_pairs == sup_pairs
    print(f"  sequence pairs: {sorted(seq_pairs)}")
    print(f"  structural pairs: {sorted(sup_pairs)}")
    print(f"  sets equal? {'YES' if same else 'no'}")
    if not same:
        only_seq = seq_pairs - sup_pairs
        only_sup = sup_pairs - seq_pairs
        print(f"  only in sequence: {sorted(only_seq)}")
        print(f"  only in structural: {sorted(only_sup)}")
    print()

    # ---- Selectivity map ----
    print("[STEP W-3c-1] Selectivity map (compute_selectivity_map at 5.0 Å)")
    # Use structural correspondence (matches B-2 dossier's primary method)
    smap = compute_selectivity_map(
        target_structure=pf,
        target_ligand_resname="OXM",
        ortholog_structure=hs,
        ortholog_ligand_resname="OXM",
        pocket_cutoff=5.0,
        alignment=sup.correspondence,
    )
    print(f"  n_aligned_pocket_residues: {smap.n_aligned_pocket_residues}")
    print(f"  pocket_target_interaction_kcal:   {smap.pocket_target_interaction_kcal:.3f}")
    print(f"  pocket_ortholog_interaction_kcal: {smap.pocket_ortholog_interaction_kcal:.3f}")
    print(f"  pocket_delta_kcal:                {smap.pocket_delta_kcal:.3f}")
    # Also compute with sequence correspondence for cross-check
    smap_seq = compute_selectivity_map(
        target_structure=pf,
        target_ligand_resname="OXM",
        ortholog_structure=hs,
        ortholog_ligand_resname="OXM",
        pocket_cutoff=5.0,
        alignment=seq_align.correspondence,
    )
    print()
    print("  Full ranked selectivity map (structural correspondence)")
    print("  rank | target           | ortholog         | Δ kcal | abs Δ")
    print("  -----+------------------+------------------+--------+------")
    # rank by signed delta (positive = target more favorable, i.e. driving Pf-selectivity)
    ranked_target = sorted(smap.residues, key=lambda r: -r.delta_kcal)
    for i, r in enumerate(ranked_target, start=1):
        print(f"  {i:>4} | {r.res_name_target}{r.res_seq_target:<14} | "
              f"{r.res_name_ortholog}{r.res_seq_ortholog:<14} | "
              f"{r.delta_kcal:>+6.3f} | {abs(r.delta_kcal):.3f}")
    print()
    # also by absolute magnitude
    ranked_abs = sorted(smap.residues, key=lambda r: -abs(r.delta_kcal))
    print("  By |Δ| magnitude:")
    for i, r in enumerate(ranked_abs, start=1):
        print(f"  {i:>4} | {r.res_name_target}{r.res_seq_target:<14} | "
              f"{r.res_name_ortholog}{r.res_seq_ortholog:<14} | "
              f"{r.delta_kcal:>+6.3f} | {abs(r.delta_kcal):.3f}")
    print()

    # ---- Locate the tryptophan STRUCTURALLY ----
    print("[STEP W-3c-2] Locate the tryptophan structurally (TRP within 5 Å of OXM)")
    trps = structural_trp_in_pocket(pf, "OXM", "A", cutoff=5.0)
    if not trps:
        print("  FATAL: no TRP within 5 Å of OXM on 1LDG chain A.")
        sys.exit(1)
    for t in trps:
        print(f"  TRP{t['res_seq']} on chain A: min heavy-atom dist to OXM = "
              f"{t['min_dist_to_ligand']:.3f} Å")
    target_trp_res_seq = trps[0]["res_seq"]
    print(f"  -> structural TRP target: res_seq {target_trp_res_seq} "
          f"(identification by 3D proximity to ligand, not by residue number)")

    # Find that residue's selectivity entry — by res_seq on target side
    trp_entry = next(
        (r for r in smap.residues if r.res_seq_target == target_trp_res_seq
         and r.res_name_target == "TRP"),
        None,
    )
    if trp_entry is None:
        print("  WARNING: TRP found in pocket but NOT in selectivity map pairings.")
        print("  This means the structural-correspondence pipeline did not pair "
              "the TRP to any ortholog residue (the HsLDH-A pocket has no spatial "
              "partner for it within match_cutoff). Check the per-residue match "
              "records below.")
        # Inspect superposition match details for the TRP
        # build res_idx for TRP in target by engine index
        rid_list = list(pf.residues.keys())
        target_trp_engine_idx = None
        for i, rid in enumerate(rid_list):
            r = pf.residues[rid]
            if r.chain_id == "A" and r.res_seq == target_trp_res_seq and r.res_name == "TRP":
                target_trp_engine_idx = i
                break
        if target_trp_engine_idx is not None:
            # which match_details entry corresponds to this engine idx?
            for md in sup.match_details:
                if md.target_engine_idx == target_trp_engine_idx:
                    print(f"  TRP{target_trp_res_seq} engine_idx={target_trp_engine_idx}: "
                          f"nearest CA dist={md.nearest_distance:.3f} Å (cutoff={sup.max_match_distance} Å); "
                          f"runner-up dist={md.runner_up_distance:.3f} Å; "
                          f"accepted={md.accepted}; "
                          f"matched_ortholog_engine_idx={md.ortholog_engine_idx}")
                    break
        rank_signed = None
        rank_abs = None
    else:
        rank_signed = next((i for i, r in enumerate(ranked_target, start=1)
                            if r.res_seq_target == target_trp_res_seq), None)
        rank_abs = next((i for i, r in enumerate(ranked_abs, start=1)
                         if r.res_seq_target == target_trp_res_seq), None)
        print(f"  TRP{target_trp_res_seq} ↔ {trp_entry.res_name_ortholog}"
              f"{trp_entry.res_seq_ortholog}")
        print(f"  energy_target_kcal:   {trp_entry.energy_target_kcal:+.4f}")
        print(f"  energy_ortholog_kcal: {trp_entry.energy_ortholog_kcal:+.4f}")
        print(f"  delta_kcal:           {trp_entry.delta_kcal:+.4f}")
        print(f"  rank (by signed Δ, target-favorable first): "
              f"{rank_signed} of {len(ranked_target)}")
        print(f"  rank (by |Δ| magnitude): {rank_abs} of {len(ranked_abs)}")
    print()

    # ---- Dossier JSON ----
    dossier = {
        "what_this_is": (
            "W-3 confirmatory selectivity run on wild-type PfLDH (1LDG, chain A) "
            "vs HsLDH-A (1I10, chain A), chemotype-matched on OXM + NAI. Tests "
            "the B-3 pre-registered hypothesis that the engine's miss of W107f "
            "on 4PLZ was input-shaped (the substrate-specificity loop was "
            "disordered in 4PLZ; with the loop ORDERED in 1LDG and the same LJ "
            "kernel, does the engine flag the active-site tryptophan?). "
            "Confirmatory and explicitly NOT blind."
        ),
        "what_this_is_not": (
            "Not a free energy of binding (LJ-only). Not blind — the literature "
            "answer is already known. The W107f question is the only load-bearing "
            "claim; other ranking shifts vs the B-2 dossier are expected because "
            "the input structure changed."
        ),
        "pair": {
            "target": {"pdb": "1LDG", "organism": "PLASMODIUM FALCIPARUM",
                       "ligand_resname": "OXM", "cofactor_resname": "NAI",
                       "chain": "A"},
            "ortholog": {"pdb": "1I10", "organism": "HOMO SAPIENS",
                         "ligand_resname": "OXM", "cofactor_resname": "NAI",
                         "chain": "A"},
            "chemotype_matched": True,
        },
        "gates": {
            "provenance_pf": "PASS",
            "provenance_hs": "PASS",
            "pocket_completeness_pf": {
                "verdict": pf_gate.verdict,
                "missing_in_pocket": [
                    f"{m.chain_id}/{m.res_name}{m.res_seq}"
                    for m in pf_gate.missing_in_pocket
                ],
                "n_resolved_pocket_residues": pf_gate.n_resolved_pocket_residues,
                "n_unresolved_termini": len(pf_gate.unresolved_termini),
            },
            "pocket_completeness_hs": {
                "verdict": hs_gate.verdict,
                "missing_in_pocket": [
                    f"{m.chain_id}/{m.res_name}{m.res_seq}"
                    for m in hs_gate.missing_in_pocket
                ],
                "n_resolved_pocket_residues": hs_gate.n_resolved_pocket_residues,
                "n_unresolved_termini": len(hs_gate.unresolved_termini),
            },
            "comparability": {
                "n_pocket_target": pf_pocket.n_residues,
                "n_pocket_ortholog": hs_pocket.n_residues,
                "n_oxm_atoms_target": int(len(pf_lig)),
                "n_oxm_atoms_ortholog": int(len(hs_lig)),
                "nai_shared_atom_names": len(shared_names),
                "nai_kabsch_anchor_rmsd": round(float(anchor_rmsd), 4),
                "passed": comparability_pass,
            },
        },
        "alignment_provenance": {
            "sequence": {
                "method": seq_align.method,
                "global_sequence_identity": round(seq_align.global_sequence_identity, 4),
                "n_pocket_target": seq_align.n_pocket_target,
                "n_pocket_ortholog": seq_align.n_pocket_ortholog,
                "n_confidently_paired": seq_align.n_confidently_paired,
                "n_unpaired_target": seq_align.n_unpaired_target,
                "refused": seq_align.refused,
                "min_identity_threshold": seq_align.min_identity_threshold,
            },
            "structural": {
                "method": sup.method,
                "anchor_resname": sup.anchor_resname,
                "n_anchor_atoms": sup.n_anchor_atoms,
                "anchor_rmsd": round(sup.anchor_rmsd, 4),
                "n_pocket_target": sup.n_pocket_target,
                "n_pocket_ortholog": sup.n_pocket_ortholog,
                "n_spatially_matched": sup.n_spatially_matched,
                "max_match_distance": sup.max_match_distance,
                "mean_match_distance": round(float(sup.mean_match_distance), 4),
                "n_unmatched_target": sup.n_unmatched_target,
            },
            "convergence": {
                "sequence_pairs_equal_structural_pairs": same,
                "sequence_pairs": sorted([list(p) for p in seq_pairs]),
                "structural_pairs": sorted([list(p) for p in sup_pairs]),
            },
        },
        "selectivity_map_structural": selectivity_map_to_dict(smap, top_n=len(smap.residues)),
        "selectivity_map_sequence": selectivity_map_to_dict(smap_seq, top_n=len(smap_seq.residues)),
        "trp_location": {
            "method": "structural_proximity_to_ligand_no_residue_number_assumed",
            "target_chain": "A",
            "target_pdb": "1LDG",
            "target_ligand_resname": "OXM",
            "target_pocket_cutoff_angstroms": 5.0,
            "trp_residues_within_cutoff": trps,
            "selected_trp_res_seq": target_trp_res_seq,
        },
        "trp_in_selectivity_map": (
            None if trp_entry is None else {
                "res_name_target": trp_entry.res_name_target,
                "res_seq_target": trp_entry.res_seq_target,
                "res_name_ortholog": trp_entry.res_name_ortholog,
                "res_seq_ortholog": trp_entry.res_seq_ortholog,
                "energy_target_kcal": trp_entry.energy_target_kcal,
                "energy_ortholog_kcal": trp_entry.energy_ortholog_kcal,
                "delta_kcal": trp_entry.delta_kcal,
                "rank_signed_target_favorable_first": rank_signed,
                "rank_abs_magnitude": rank_abs,
                "n_pocket_aligned": smap.n_aligned_pocket_residues,
            }
        ),
        "outcome_classification": None,  # filled below
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "engine": "physics_auditor",
            "pocket_cutoff_angstroms": 5.0,
        },
    }

    # ---- Score against pre-registered outcome map ----
    print("[STEP W-3d] Score against pre-registered outcome map")
    print("    OUTCOME 1 (VINDICATED): TRP flagged divergent OR carries large favorable LJ Δ")
    print("    OUTCOME 2 (FALSIFIED):  TRP present + in-pocket but engine does NOT flag it")
    print()
    if trp_entry is None:
        outcome_label = "OUTCOME 2"
        outcome_reason = (
            "The TRP is present and within 5 Å of OXM, but the structural-"
            "correspondence pipeline did NOT pair it to any HsLDH-A pocket "
            "residue (no spatial partner within match_cutoff). The engine's "
            "selectivity map therefore carries no per-residue verdict on the "
            "tryptophan. Under the pre-registered map this is OUTCOME 2 "
            "(falsified): the loop tryptophan is present in the WT input and "
            "in-pocket, but the engine does not flag it — the miss is "
            "kernel/correspondence-shaped, not input-shaped."
        )
    else:
        # Use the distribution of |Δ| to define "large" honestly, not a hard threshold.
        abs_deltas = sorted([abs(r.delta_kcal) for r in smap.residues], reverse=True)
        trp_abs = abs(trp_entry.delta_kcal)
        # define "top" = top quartile by |Δ|
        n = len(abs_deltas)
        if n == 0:
            cutoff_top_quartile = float("inf")
        else:
            cutoff_top_quartile = abs_deltas[max(0, n // 4 - 1)]
        # rank_abs is the 1-indexed magnitude rank
        if rank_abs is not None and rank_abs <= max(1, n // 4):
            outcome_label = "OUTCOME 1"
            outcome_reason = (
                f"TRP{target_trp_res_seq} carries |Δ|={trp_abs:.3f} kcal "
                f"(Δ={trp_entry.delta_kcal:+.3f}), ranking {rank_abs} of {n} "
                f"by |Δ| magnitude — in the top quartile of the pocket's "
                f"selectivity-contributing residues. Under the pre-registered "
                f"map this is OUTCOME 1 (vindicated): with the loop ORDERED "
                f"in the WT input, the engine flags the tryptophan with a "
                f"large LJ delta. The B-3 miss was input-shaped."
            )
        elif trp_entry.res_name_target != trp_entry.res_name_ortholog:
            outcome_label = "OUTCOME 1"
            outcome_reason = (
                f"TRP{target_trp_res_seq} pairs to "
                f"{trp_entry.res_name_ortholog}{trp_entry.res_seq_ortholog} — "
                f"a non-identical residue, so the engine flags it as "
                f"divergent. Under the pre-registered map this satisfies the "
                f"OUTCOME 1 'flagged as divergent OR large favorable LJ delta' "
                f"trigger."
            )
        else:
            outcome_label = "OUTCOME 2"
            outcome_reason = (
                f"TRP{target_trp_res_seq} pairs to "
                f"{trp_entry.res_name_ortholog}{trp_entry.res_seq_ortholog} "
                f"(identity-conserved, so not flagged divergent) and carries "
                f"|Δ|={trp_abs:.3f} kcal at rank {rank_abs} of {n} by |Δ| "
                f"(not in the top quartile, which has |Δ| ≥ "
                f"{cutoff_top_quartile:.3f}). Under the pre-registered map "
                f"this is OUTCOME 2 (falsified): the tryptophan is present "
                f"and in-pocket but the engine does NOT flag it. The miss is "
                f"kernel-shaped, not input-shaped."
            )

    print(f"=> {outcome_label}")
    print(f"   {outcome_reason}")
    print()

    dossier["outcome_classification"] = {
        "outcome": outcome_label,
        "reason": outcome_reason,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dossier, indent=2, default=str))
    print(f"Dossier written: {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
