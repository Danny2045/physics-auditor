# WT-PfLDH W107f Mechanistic Check — Pre-Registered Outcome Map

**Status:** PRE-REGISTERED (committed BEFORE any WT structure is acquired)
**Branch:** feat/ldh-w107f-mechanistic-check
**Base commit:** 3ee581e (B-3 scoring commit)
**Date pre-registered:** 2026-05-23

## Integrity note

This is a CONFIRMATORY, POST-LITERATURE test — explicitly NOT a blind prediction.
The literature answer (W107f in the substrate-specificity loop is the dominant
selectivity determinant between PfLDH and HsLDH-A) is already known from the B-3
literature check (commit da9a05c). The blindness markers from the B-2 dossier do
NOT apply here. The integrity rule for this test is different: the outcome map
below is committed BEFORE any WT structure is acquired, so the interpretation
of the engine's behavior cannot be retrofitted after seeing the result.

## PURPOSE

The B-3 partial-pass verdict (commit 3ee581e) explained the engine's miss of
W107f as INPUT-SHAPED — the substrate-specificity loop (including W107f) was
disordered and absent from the 4PLZ deposit (a W107fA mutant), so the engine
could not flag a residue that was not present in the coordinates. This is a
falsifiable hypothesis.

This test acquires a WILD-TYPE PfLDH structure in which the loop is ordered and
W107f is present, deploys a new pocket-completeness gate to confirm the loop is
in fact resolved in the new deposit, and runs the same selectivity engine on
the WT structure (vs. the same HsLDH-A 1I10 reference) to test whether the
engine flags W107f when it IS present in the input.

## PRE-REGISTERED OUTCOME MAP

The four outcomes below exhaust the possibility space. The verdict that will be
assigned at the end of the test depends ONLY on which of these the observed
result matches; no fifth interpretation will be invented after the fact.

### OUTCOME 1 — Explanation VINDICATED (input-shaped miss)

W107f is present and ordered in the WT structure, AND lies within the
pocket-defining distance (5 Å) of the bound ligand,
AND the selectivity engine flags W107f as divergent OR carries a large LJ
delta on it.

**Interpretation:** The B-3 miss was input-shaped. The engine works correctly
when the relevant residue is present; the 4PLZ disordered-loop gap was the
sole cause of the miss. The B-3 hypothesis stands.

### OUTCOME 2 — Explanation FALSIFIED (kernel-shaped miss)

W107f is present and ordered in the WT structure, AND lies within pocket
distance, AND the engine does NOT flag it (no divergent verdict, no large LJ
delta).

**Interpretation:** LJ-only physics has a real blind spot for W107f that was
previously misattributed to the input. The B-3 hypothesis is falsified; the
engine's kernel needs work (likely π-stacking or other non-LJ terms).

### OUTCOME 3 — Conformation-dependent (neither confirmed nor falsified)

W107f is present in the coordinates of the WT structure, BUT the substrate-
specificity loop is conformationally retracted such that W107f lies OUTSIDE
the 5 Å pocket-defining cutoff in this particular deposit.

**Interpretation:** The engine's verdict on W107f depends on the
conformational state of the loop captured in the input. The static-structure
method is itself the limitation — a finding about the method, not the
engine's kernel or the input fidelity. Report transparently as conformation-
dependent.

### PRE-CONDITION FAILURE — Test cannot run as designed

No available wild-type PfLDH deposit has W107f present and ordered, or no
suitable chemotype-matched WT+ligand structure can be found.

**Interpretation:** Report honestly that the W107f mechanistic check cannot be
conducted as designed with currently available structural data; do NOT force
the test by using inadequate inputs.

## Commitment

This file is being committed alone, before any WT structure is acquired and
before any selectivity computation is run on a WT structure. The commit SHA of
this pre-registration is the timestamp of the outcome map: any deviation in
later steps from the outcomes above must be explicitly noted and explained.

---

# W-2 — WT-PfLDH structure acquisition and outcome classification

**Pre-registration commit:** 44e4cf7
**Gate v1 (W-1):** e3050c7
**Gate v2 (internal/terminal split, W-2a):** b2b7594

## W-2a — Gate refinement (terminal-vs-internal distinction)

The W-1 reach-bound gate falsely flagged 4PLZ's C-terminal His-tag residues
(HIS320, HIS321) as in-pocket-missing because a dangling terminus has no
second anchor to limit its reach. The W-2a fix classifies missing residues
into two strictly disjoint categories:

- **INTERNAL gap**: resolved residues on BOTH sequence sides in the same
  chain. Eligible for the reach-bound pocket check; only these residues can
  drive an INCOMPLETE verdict.
- **TERMINAL tail**: resolved residues on only ONE sequence side (the
  residue runs off the start or end of the resolved chain). Reported in
  `unresolved_termini` for transparency, never used to flip the verdict.

After the fix, the 4PLZ + OXM diagnostic now reports:

- `INCOMPLETE: 12 internal-gap pocket-region residue(s) missing:
  A/THR82, A/LYS83, A/ALA84, A/PRO85, A/GLY86, A/LYS87, A/SER88, A/ASP89,
  A/LYS90, A/GLU91, A/ALA92, A/ASN93`
- 8 unresolved terminal residues (A/MET0, A/ALA1, A/PRO2, A/HIS317–HIS321)
  are reported informationally — they no longer corrupt the in-pocket set.

The internal-loop signal (the documented B-3 finding) is preserved; the
His-tag artefact is suppressed.

## W-2b — Search criteria (metadata only, fixed before search)

Mechanically-applied RCSB filters, no abstracts read:

1. organism: Plasmodium falciparum (NCBI taxon 5833),
2. struct.title contains the phrase "lactate dehydrogenase",
3. carries a substrate-class active-site ligand: OXM / OXL / PYR / LAC,
4. carries an NAD-class cofactor: NAD / NAI / NAP / NDP / NAH,
5. resolution ≤ 2.5 Å,
6. NOT flagged as a W107-region mutant in the title or pdbx_mutation field
   (the structural TRP-in-loop check in W-2c is the real verification).

Initial broad query (filter 1+2+5) returned 18 PfLDH entries at ≤ 2.5 Å.
Applying ligand filters (3+4) narrowed to 4 entries.

## W-2b — Initial candidate table (before mutation filter)

| PDB | Res (Å) | Deposit date | Substrate | Cofactor | Mutation (metadata) | Status |
|-----|--------:|--------------|-----------|----------|---------------------|--------|
| 1T2D | 1.10 | 2004-04-21 | OXL | NAD | — | keep |
| 4PLZ | 1.05 | 2014-05-20 | OXM | NAI | "W107fA mutant" in title | **DISQUALIFIED** (already used in B-3, is the W107f loop mutant) |
| 1LDG | 1.74 | 1996-09-10 | OXM | NAI | — | keep |
| 1T2E | 1.85 | 2004-04-21 | OXM | NAI | S245A, A327P (NOT W107-region) | keep |

3 candidates pass to W-2c.

## W-2c — Acquisition + structural verification

For each candidate: download into `benchmark/structures/experimental/`,
run `verify_pdb_source` against PLASMODIUM FALCIPARUM, read
HEADER/TITLE/SOURCE/REMARK 2 live, select the lowest chain carrying both
the substrate and the cofactor (matching the blind-test selection rule),
run the pocket-completeness gate, and verify the active-site tryptophan
STRUCTURALLY (any TRP residue with heavy atoms within 5 Å of the substrate
ligand, regardless of residue number).

| PDB | Provenance | Chain | Gate verdict | missing_in_pocket | TRP near substrate | Min TRP–ligand dist | Outcome setup |
|-----|------------|-------|--------------|-------------------|--------------------|---------------------|---------------|
| 1T2D | PASS | A | COMPLETE | 0 | TRP92 (chain A) | 3.87 Å | OUTCOME 1/2 (engine will be tested) |
| 1LDG | PASS | A | COMPLETE | 0 | TRP107 (chain A) | 4.04 Å | OUTCOME 1/2 (engine will be tested) |
| 1T2E | PASS | A | COMPLETE | 0 | TRP107 (chain A) | 3.85 Å | OUTCOME 1/2 (engine will be tested) |

All three candidates satisfy the load-bearing pre-condition: a tryptophan is
present, ordered, AND within 5 Å of the bound substrate. The TRP appears at
res_seq 92 in the 1T2D numbering family and at res_seq 107 in the 1LDG /
1T2E numbering family — the structural identification (proximity to ligand)
does not depend on the deposit's choice of numbering.

The gate reports each candidate as COMPLETE: no internal-gap residue in the
pocket region. Some terminal residues are unresolved (informational), but
the catalytic-loop region is fully built in every WT candidate — i.e. the
disorder seen in 4PLZ is specific to the W107fA mutant deposit and does
not reflect intrinsic loop disorder in WT PfLDH.

## W-2c — Mechanical tiebreak and W-3 selection

Selection criteria, applied in this order:

1. Loop TRP present, ordered, within 5 Å of ligand — all three candidates
   tie.
2. Best resolution — 1T2D (1.10 Å) beats 1LDG (1.74 Å) and 1T2E (1.85 Å).
3. Earliest deposition — not reached.

**Selected for W-3: 1T2D**
(Plasmodium falciparum LDH complexed with NAD⁺ and oxalate; deposit
2004-04-21; resolution 1.10 Å; substrate OXL, cofactor NAD; chain A;
TRP92 — the W107f-equivalent in the 1T2D numbering family — sits 3.87 Å
from the bound oxalate.)

Note on ligand chemotype: 1T2D carries oxalate (OXL) + NAD, whereas the
B-3 blind test used the chemotype-matched OXM + NAI on both PfLDH (4PLZ)
and HsLDH-A (1I10). The mechanical tiebreak chose by resolution; the
chemotype difference will be reported transparently in W-3 and the W-3
HsLDH-A reference may need to be re-selected for chemotype matching
against 1T2D. That decision belongs to W-3, not W-2.

## W-2d — Pre-registered outcome assignment

The chosen W-3 structure (1T2D) has the active-site tryptophan PRESENT,
ORDERED, and within 5 Å of the substrate ligand. This sets up the test of:

- **OUTCOME 1** (explanation VINDICATED) — the engine, when run on 1T2D vs
  a chemotype-matched HsLDH-A reference in W-3, flags the loop tryptophan
  or carries a large LJ delta on it.
- **OUTCOME 2** (explanation FALSIFIED) — the engine, when run on 1T2D,
  does NOT flag the loop tryptophan despite its presence.

OUTCOME 3 (conformation-dependent) and the PRE-CONDITION FAILURE branch
do NOT apply: the tryptophan is present and ordered inside the 5 Å pocket
in 1T2D, so the test runs as designed.

The actual W-3 result will determine which of OUTCOME 1 vs OUTCOME 2 the
engine's behaviour matches. No selectivity computation has been run yet —
that is W-3.

