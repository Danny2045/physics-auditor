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

---

# W-3 — Confirmatory selectivity run and outcome score

**Runner:** `benchmark/run_w107f_selectivity.py`
**Dossier:** `benchmark/results/selectivity_maps/W107F_MECHANISTIC_RESULT.json`

## W-3 pair (Option-B decision)

| Side | PDB | Chain | Substrate | Cofactor | Resolution |
|------|-----|-------|-----------|----------|-----------:|
| PfLDH (WT, loop ordered) | 1LDG | A | OXM | NAI | 1.74 Å |
| HsLDH-A (B-3 reference)  | 1I10 | A | OXM | NAI | 2.30 Å |

The W-2 mechanical tiebreak picked 1T2D by resolution, but the W-3 brief
overrode that on Option-B grounds: 1LDG carries the same chemotype as the
B-3 HsLDH-A reference (1I10 holds OXM + NAI; 1LDG also holds OXM + NAI;
1T2D would have introduced an OXL+NAD vs OXM+NAI chemotype mismatch).
Reusing the B-3 1I10 reference holds every variable constant except the
target structure — exactly the controlled comparison the mechanistic
check needs.

## W-3a — Gates (both pass before any selectivity is computed)

**Provenance.** `verify_pdb_source` PASS on both:
- 1LDG → PLASMODIUM FALCIPARUM
- 1I10 → HOMO SAPIENS

**Pocket-completeness (the new gate's first load-bearing use guarding a
real selectivity run).**

| Side | Verdict | n_pocket | n_missing_in_pocket | n_unresolved_termini |
|------|---------|---------:|--------------------:|---------------------:|
| 1LDG | COMPLETE | 11 | 0 | 4 (informational) |
| 1I10 | COMPLETE | 9  | 0 | 0 |

Both pockets are fully resolved at the 5 Å substrate shell. The four
unresolved 1LDG residues are terminal-tail (not internal-gap) and so
correctly do not corrupt the verdict — the W-2a fix is doing its job.

**Comparability gate.** All match the B-1 standard:

| Metric | 1LDG (PfLDH) | 1I10 (HsLDH-A) |
|--------|--------------|----------------|
| Pocket residues at 5 Å | 11 | 9 |
| OXM heavy atoms | 6 | 6 |
| NAI shared atom names | 44 | 44 |
| NAI-anchored Kabsch RMSD | — | **0.8652 Å** |

Comparability gate PASS — chemotype-matched (OXM + NAI both sides), rigid-
body assumption holds at the cofactor (sub-Å anchor RMSD), pocket sizes
comparable (11 vs 9).

## W-3b — Correspondence convergence (the disagreement that matters)

Both methods run; they disagree on EXACTLY ONE pair, and that one pair is
the load-bearing tryptophan.

**Sequence correspondence (Needleman-Wunsch, identity gate):**
- global identity: 33.11 %
- refused: False; threshold 0.25 cleared
- pocket pairs paired: 10 of 11; one target-pocket residue unpaired
- the unpaired target residue is **TRP107** — it sits in the PfLDH-only
  KSDKE-region insertion, so it has no homologous position in the HsLDH-A
  sequence and the global alignment correctly gaps it

**Structural correspondence (NAI-anchored Kabsch):**
- 11 of 11 target-pocket residues spatially matched
- anchor RMSD 0.8652 Å, mean post-superposition match 1.22 Å
- **TRP107 is paired to HsLDH-A GLY102** (the nearest CA in the
  superposed frame within the 4 Å match cutoff)

**Disagreement on the W107f residue itself:** the sequence method assigns
NO verdict to TRP107 (gap), the structural method assigns the pair
TRP107 ↔ GLY102. This is exactly the spec's "if the methods disagree on
the residue corresponding to the loop tryptophan, that is important —
report it explicitly" case. The disagreement is structurally meaningful:
PfLDH's insertion creates a tryptophan that HsLDH-A simply does not have,
so the only honest pairing is the spatial one. The structural method is
the correct correspondence for the W107f verdict; the sequence method
correctly refuses to pair an insertion to a deletion.

## W-3c — Selectivity map and the structural TRP

The selectivity map is computed using the **structural correspondence**
(the one that actually pairs the TRP). The full ranked map:

| rank (signed Δ) | target | ortholog | Δ kcal/mol | E_target | E_ortholog |
|----------------:|--------|----------|-----------:|---------:|-----------:|
| 1 | PRO246 | THR247 | **+2.769** | −1.490 | +1.279 |
| 2 | ARG171 | ARG168 | +0.943 | +1.481 | +2.424 |
| **3** | **TRP107** | **GLY102** | **+0.869** | **−0.876** | **−0.007** |
| 4 | SER245 | TYR246 | +0.502 | +0.183 | +0.685 |
| 5 | PRO250 | ILE251 | +0.150 | −0.394 | −0.212 |
| 6 | ASN140 | ASN137 | +0.115 | −0.258 | −0.143 |
| 7 | LEU167 | LEU164 | +0.096 | −0.710 | −0.614 |
| 8 | ASP168 | ASP165 | +0.094 | −0.060 | +0.034 |
| 9 | HIS195 | HIS192 | +0.084 | −0.263 | −0.179 |
| 10 | ALA236 | ALA237 | −0.041 | −1.088 | −1.129 |
| 11 | ARG109 | ARG105 | −1.089 | +0.605 | −0.484 |

By |Δ| magnitude (rank, residue, |Δ|): 1 PRO246/2.769; 2 ARG109/1.089;
3 ARG171/0.943; **4 TRP107/0.869**; 5 SER245/0.502; rest <0.15.

**Structural TRP location (proximity-based, no residue number assumed):**
TRP107 on chain A of 1LDG, minimum heavy-atom distance to OXM = 4.04 Å.
Re-confirms the W-2 finding.

**Tryptophan's per-residue verdict in the selectivity map:**
- pair: PfLDH TRP107 ↔ HsLDH-A GLY102
- target side: −0.876 kcal/mol (TRP makes a favorable LJ contact with OXM)
- ortholog side: −0.007 kcal/mol (HsLDH-A's GLY102 makes essentially no
  contact at the corresponding position)
- Δ = +0.869 kcal/mol (positive → Pf-favorable; the TRP gives PfLDH ~0.87
  kcal/mol more interaction at this position than HsLDH-A's glycine)
- **divergent: YES** — TRP↔GLY is the largest possible residue-type swap
  (the bulkiest aromatic vs the side-chain-less residue), unambiguously
  flagged divergent in the engine's residue-substitution analysis
- rank by signed Δ: **3 of 11** (third-largest target-favorable
  contribution)
- rank by |Δ|: **4 of 11**

## W-3c — Cross-check with B-3's non-TRP picks

B-3's two top-ranked engine picks survive in the W-3 (WT) input, with
larger magnitudes thanks to the loop being ordered:

| Canonical position | B-3 (4PLZ, mutant) | W-3 (1LDG, WT) |
|--------------------|---------------------|-----------------|
| T246P (Dunn 1996) | PRO234↔THR247, Δ=+2.465 | PRO246↔THR247, **Δ=+2.769** |
| I250P (Dunn 1996) | PRO238↔ILE251, Δ=+0.182 | PRO250↔ILE251, Δ=+0.150 |
| (new in W-3)      | — | **TRP107↔GLY102, Δ=+0.869** |

The W-3 picks include both B-3 findings plus the formerly-missing W107f.
The B-3 results are reproduced; the loop ordering surfaces the W107f
contact previously invisible to the engine.

## W-3d — Score against the pre-registered outcome map

The pre-registered map (committed at 44e4cf7, before any WT structure
acquisition) sets out four mutually exclusive outcomes; applying it as
written without re-interpretation:

> **OUTCOME 1 (VINDICATED):** "W107f is present, ordered, and within the
> pocket-defining distance (5 Å) of the bound ligand, AND the engine
> flags it as divergent OR carries a large LJ delta on it. The miss was
> input-shaped."

All three conditions hold on 1LDG:
1. The tryptophan is **present** (TRP107 has full ATOM-record
   coordinates in 1LDG chain A).
2. The tryptophan is **ordered** (every heavy atom is resolved with
   normal B-factors).
3. The tryptophan is **within 5 Å of the substrate** (4.04 Å minimum
   heavy-atom distance to OXM).
4. The engine **flags it as divergent**: the structural correspondence
   pairs TRP107 to GLY102 — a non-identical residue, and one of the
   largest residue-type swaps possible.
5. The engine **carries a meaningful LJ delta** on it: Δ = +0.869 kcal/mol
   in PfLDH's favor (the TRP makes a −0.876 kcal/mol contact that the
   HsLDH-A glycine cannot make). The TRP ranks 3rd-largest by signed Δ
   (third-largest target-favorable contribution in the pocket) and
   4th-largest by |Δ|. By |Δ| this sits one rank outside the top
   quartile (top 3 of 11), but the divergent-substitution trigger of
   OUTCOME 1 fires unconditionally on the TRP↔GLY pairing.

**Verdict: OUTCOME 1 (VINDICATED).**

The B-3 partial-pass explanation is **vindicated**: with the loop
ORDERED in the input structure, the same LJ kernel that produced the
B-3 verdict now flags the W107f tryptophan as both divergent and
energetically favorable for the parasite target. The B-3 miss was
**input-shaped**, not kernel-shaped — exactly what the pre-registered
hypothesis predicted. There is no need to invoke a physics extension
(no π-stacking term, no electrostatics, no solvation correction) to
explain why the engine missed W107f in B-3; the engine could not
analyse coordinates that were not in the input.

OUTCOME 2 (FALSIFIED), OUTCOME 3 (conformation-dependent), and the
PRE-CONDITION FAILURE branch all do NOT apply.

## W-3e — Honest caveats

- The pre-registered map allows OUTCOME 1 to fire on EITHER divergent-
  substitution OR large-LJ-delta. The TRP↔GLY pairing here triggers BOTH
  criteria, but the |Δ| rank (4 of 11) is just outside the top quartile
  (top 3 of 11). A stricter operationalisation that required top-3 |Δ|
  AND divergent would still classify this as OUTCOME 1 by the divergent
  criterion alone; a stricter one that required top-3 |Δ| only would
  classify this as ambiguous. The pre-registered map does NOT require
  both — and was not weakened to fit this result — so the rule is
  applied as written: OUTCOME 1.
- The sequence-correspondence method assigned NO verdict to TRP107 (the
  KSDKE insertion is a gap region). All of the load-bearing TRP analysis
  rests on the structural correspondence. The two methods converge on
  every other pocket residue.
- The 1LDG vs 1I10 anchor RMSD (0.865 Å) is slightly higher than B-3's
  4PLZ vs 1I10 (0.673 Å), reflecting subtle conformational differences
  between the WT closed-loop conformation and the W107fA open-loop
  mutant. Sub-Å anchor remains well within the rigid-body assumption;
  this is noise, not signal.
- The OXL+NAD candidate (1T2D, picked by the W-2 mechanical tiebreak)
  remains in `benchmark/structures/experimental/` for future
  chemotype-cross-check work; using it in W-3 would have re-introduced
  the chemotype confound the B-1 1I10 acquisition was designed to remove.

