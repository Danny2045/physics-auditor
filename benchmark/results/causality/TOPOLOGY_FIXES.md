## Audit note (PR fix/benchmark-mislabel-pair1-and-causality-artifacts — PR-A1)

The pair labels used in the tables below ("HsDHODH", "SmDHODH", "HsHDAC8", "LmPTR1") are the original BENCHMARK_PAIRS labels at the time this document was written. PR-A1's identity audit found four of them mismatched the underlying experimental input:

- "HsDHODH" pair used 1MVS, which is **HsDHFR** (P00374). Dossier renamed `HsDHFR_causality.json`.
- "SmDHODH" pair used 1D3H, which is **HsDHODH** (Q02127). Dossier renamed `HsDHODH_causality.json`.
- "HsHDAC8" pair used 4HJO, which is **EGFR kinase domain** (P00533). Dossier renamed `EGFR_4HJO_causality.json`.
- "LmPTR1" pair used 2BPR, which is **DnaK** from *E. coli* (P0A6Y8). Dossier renamed `DnaK_2BPR_causality.json`.

The AF (predicted-side) models in each pair were genuine: pair 1's AF model is real HsDHODH, pair 2's is real SmDHODH, pair 3's is real HsHDAC8, pair 5's is real LmPTR1. The mismatch is on the experimental side. Tables below now use the corrected pair names. The kcal numbers themselves are unchanged. See STATE_OF_PHYSICS_AUDITOR.md Findings 1, 8, 9.

---

# Topology fixes from the causality run — May 2026

The first run of the causality layer surfaced two distinct topology
bugs that were invisible to aggregate metrics. Both are template
gaps, and both are fixed in this branch.

## Bug 1 — disulfide bonds not inferred

The original `infer_bonds_from_topology` builds bonds from intra-
residue templates plus the next-residue peptide bond, with a
distance fallback for HETATM/ligand atoms. **Two CYS sidechains
forming a disulfide bond between non-adjacent residues are not
covered by any of those rules.**

In a structure that genuinely has a disulfide bond (HsAChE has at
least one), the two SG sulfur atoms sit ~2.05 Å apart. The LJ
kernel reads that pair as a non-bonded clash and assigns several
hundred kcal/mol of repulsion to a single residue. The hit is
hidden in aggregate clashscore (it averages over thousands of
pairs) but appears as a residue-level spike in
`per_residue_decomposition`.

### Fix

A disulfide-detection pass in `infer_bonds_from_topology` after
the template phase: any pair of CYS-SG atoms at 1.9–2.4 Å is
added to the bond list. The window matches the AMBER ff14SB
disulfide criterion and is wide enough to handle B-factor smear
without producing false positives on solvent-exposed thiols
(which sit at ≥ 4 Å).

Synthetic test (`test_four_cys_structure_bonds_only_close_pair`):
a 4-CYS toy structure with one SG-SG pair at 2.05 Å and others at
≥ 5 Å. Only the 2.05 Å pair gets bonded.

Boundary test (`test_distance_window_boundaries`): SG pairs at
1.85, 2.50, and 3.00 Å — none bond, since they fall outside the
window.

## Bug 2 — C-terminal carboxylate (OXT) not bonded

This was the actual cause of CoV2Mpro's +1067 kcal residue, not
the disulfide story we initially expected.

The protein C-terminus has a carboxylate group: the backbone C is
double-bonded to O *and* single-bonded to OXT (the terminal
hydroxyl oxygen). The original `BACKBONE_BONDS` template was
`[("N","CA"), ("CA","C"), ("C","O")]` — no OXT. So **every C-
terminus of every chain in every protein** had an unbonded OXT
atom sitting ~1.25 Å from the C, and the LJ kernel returned
1000 kcal/mol of repulsion (capped) for that single pair.

### Fix

Append `("C", "OXT")` to `BACKBONE_BONDS`. Single-line change.

### Surface area

This bug was present in all 6 of the previously-runnable benchmark
pairs. The visible signal was concentrated on CoV2Mpro because
its C-terminus happens to be CYS296 (which then dominated the top-
unfavorable list). The other pairs absorbed the +1000 kcal hit in
their whole-protein total but didn't have the C-terminal residue
flagged as pocket, so the pocket-delta numbers were unchanged.

After the fix, whole-protein LJ totals dropped substantially across
the board:

| Pair | Before (kcal) | After (kcal) | Δ |
|---|---:|---:|---:|
| HsDHFR AF (was "HsDHODH"; AF model AF-Q02127 is genuine HsDHODH) | −1046 | −2111 | −1065 |
| HsDHODH AF (was "SmDHODH"; AF model AF-G4VFD7 is genuine SmDHODH) | −954 | −2021 | −1067 |
| EGFR_4HJO AF (was "HsHDAC8"; AF model AF-Q9BY41 is genuine HsHDAC8) | −1075 | −2138 | −1063 |
| DnaK_2BPR AF (was "LmPTR1"; AF model AF-Q01782 is genuine LmPTR1) | −346 | −1407 | −1061 |
| HsAromatase AF | −1717 | −2781 | −1064 |
| CoV2Mpro AF | −606 | −1671 | −1065 |

Each pair shifts by roughly −1064 kcal — the LJ-cap value, exactly
once, which is the signature of a single bad pair on each chain
becoming bonded.

## What this means for the causality layer

The pocket-delta numbers (the headline causality result) are
*unchanged* by these fixes:

| Pair | Pocket Δ before | Pocket Δ after |
|---|---:|---:|
| HsDHFR (was "HsDHODH") | −54.24 | −54.24 |
| HsDHODH (was "SmDHODH") | +52.83 | +52.83 |
| EGFR_4HJO (was "HsHDAC8") | −26.83 | −26.83 |
| DnaK_2BPR (was "LmPTR1") | −9.69 | −9.69 |
| HsAromatase | −20.20 | −20.20 |
| **CoV2Mpro** | **+1019.30** | **−45.67** |

CoV2Mpro now reports a sane −45.67 kcal pocket delta with SER109
as the top unfavorable residue (+6.8 kcal). That's a normal
apo-vs-holo pocket signal. The +1019 was an artifact.

The top-residue findings for the other five pairs (EGFR_4HJO
TYR94 [was "HsHDAC8 TYR94"; residue in 4HJO = EGFR kinase domain,
not HDAC8], HsDHFR GLY112 [was "HsDHODH GLY112"; residue in 1MVS
= HsDHFR], HsDHODH GLU132 [was "SmDHODH GLU132"; residue in 1D3H
= HsDHODH], etc.) are unchanged. Those residues are correctly
identified at the atomic level by the layer; the protein-family
attributions for the relabeled pairs need to be re-derived (see
CAUSALITY_FINDINGS.md audit note).

## Non-claims

- This branch only addresses two missing-bond cases. There are
  more topology gaps the layer could still surface — e.g.
  N-terminal H atoms (H1/H2/H3), modified residues with non-
  standard sidechain atoms, glycan linkages, metal-ligand
  coordination. None of those produced visible artifacts in the
  current 6-pair benchmark, but they will appear when the
  benchmark expands.
- The disulfide window [1.9, 2.4] Å is empirical, derived from
  the AMBER ff14SB convention. Disulfide bonds in poorly-
  resolved structures (R > 3.0 Å) can drift outside this window
  and would be missed. Acceptable for now; revisit if a benchmark
  structure has a known disulfide that fails to bond.
- These fixes WERE thought not to change *binding-site clashscore*
  (the original reasoning: the bonded mask operates on a much
  smaller atom set, where the C-terminal OXT is rarely in the
  pocket). That reasoning was incorrect — the bonded mask is
  shared infrastructure between the LJ and clash kernels, so the
  benchmark clashscore numbers WERE affected by this topology
  fix. The stale JSON was regenerated and locked under a
  regression test in PR-DET (see STATE_OF_PHYSICS_AUDITOR.md
  Finding 11). The original preprint that carried the stale
  numbers has since been withdrawn from master.
