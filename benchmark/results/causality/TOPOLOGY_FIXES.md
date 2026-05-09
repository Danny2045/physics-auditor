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
| HsDHODH AF | −1046 | −2111 | −1065 |
| SmDHODH AF | −954 | −2021 | −1067 |
| HsHDAC8 AF | −1075 | −2138 | −1063 |
| LmPTR1 AF | −346 | −1407 | −1061 |
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
| HsDHODH | −54.24 | −54.24 |
| SmDHODH | +52.83 | +52.83 |
| HsHDAC8 | −26.83 | −26.83 |
| LmPTR1 | −9.69 | −9.69 |
| HsAromatase | −20.20 | −20.20 |
| **CoV2Mpro** | **+1019.30** | **−45.67** |

CoV2Mpro now reports a sane −45.67 kcal pocket delta with SER109
as the top unfavorable residue (+6.8 kcal). That's a normal
apo-vs-holo pocket signal. The +1019 was an artifact.

The top-residue findings for the other five pairs (HsHDAC8 TYR94,
HsDHODH GLY112, SmDHODH GLU132, etc.) are unchanged. Those
residues are correctly identified by the layer, with or without
this fix.

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
- These fixes do not change *binding-site clashscore* (which uses
  the same bonded mask but on a much smaller atom set, where the
  C-terminal OXT is rarely in the pocket). The benchmark
  clashscore numbers in the preprint are unchanged.
