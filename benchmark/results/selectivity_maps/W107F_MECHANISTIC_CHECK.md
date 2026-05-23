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
