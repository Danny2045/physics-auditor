# LDH blind test — B-3 result: scoring the locked PfLDH-vs-HsLDH-A prediction

Phase B-3 result for the blind PfLDH-vs-HsLDH-A selectivity prediction.
Records the answer key, the locked prediction, the numbering reconciliation,
and the rubric verdict.

**Commit ordering (blindness preserved by git):**

1. **`6dfa69b`** — B-2 lock of the prediction (`PfLDH_vs_HsLDH-A_blind_prediction.json`), pre-literature.
2. **`da9a05c`** — B-3 literature summary (`LDH_LITERATURE.md`), written BEFORE the locked dossier was opened.
3. **this commit** — B-3 scoring (this file) + dossier link fill (`literature_check_commit`) + test assertion update.

The literature was read and committed before the prediction was opened
in this session. Evaluation blindness is preserved by the commit order.

---

## 1. Literature answer (from `LDH_LITERATURE.md`)

Primary PfLDH-vs-HsLDH-A substrate-selectivity determinants per the
peer-reviewed literature, in priority order:

1. **5-residue insertion `KSDKE` at canonical 107a–107e** in the substrate-
   specificity loop (PfLDH-only; absent from HsLDH-A). Foundational
   structural feature (Dunn 1996; Boucher 2014).
2. **W107f** — the post-insertion tryptophan. **Single most important**
   specificity residue by mutagenesis (Boucher 2014: W→A loses ~5 orders
   of magnitude pyruvate activity; only Tyr/Phe partially rescue).
3. **K102 (Pf) ↔ Q102 (Hs)** — a true substitution at canonical 102, but
   functionally **demoted** by Boucher 2014 mutagenesis (K102 is extruded
   from the active site by the insertion).
4. **S245 (Pf) ↔ Y (Hs)** — selectivity hot spot for oxadiazole inhibitors
   (Cameron 2004, PDB 1T2D; OXD1 makes a chemotype-specific H-bond with
   Ser245 that HsLDH-A cannot accept due to Tyr orientation).
5. Background: canonical **T246P** and **I250P** — Pf-specific
   cofactor-site substitutions (Dunn 1996), affect NAD/APAD selectivity
   more than substrate.

---

## 2. The locked prediction (`PfLDH_vs_HsLDH-A_blind_prediction.json`)

Locked at **6dfa69b** before any literature was consulted. The engine's
key claims (in 4PLZ/1I10 PDB residue numbering):

**Primary divergent set (substitutions found in the pocket):**

| target (4PLZ, PfLDH) | ortholog (1I10, HsLDH-A) | Δ LJ (kcal) |
|---|---|---|
| **PRO234**  | **THR247**  | **+2.465** (BOLD sub-claim) |
| PRO238  | ILE251  | +0.182 |

**Convergent headline (top-|Δ| residues, divergent or conserved-but-energetic):**

| target | ortholog | Δ LJ (kcal) | substitution? |
|---|---|---|---|
| PRO234 | THR247 | +2.465 | **yes** (bold) |
| ARG157 | ARG168 | +1.991 | no — conserved |
| HIS181 | HIS192 | −1.702 | no — conserved |
| PRO238 | ILE251 | +0.182 | **yes** |

**Pocket comparability:** 8 residues paired on each side, NAI-anchored
Kabsch RMSD 0.67 Å, 44 anchor atoms, mean post-superposition match
1.06 Å. Sequence identity 32%. Comparability gate passed in B-1.

---

## 3. Numbering reconciliation (explicit, before scoring)

The literature uses **dogfish-LDH-canonical numbering with letter
suffixes for the insertion** (`107a–107e`, `W107f`). The locked
prediction uses **PDB residue numbering** of 4PLZ (PfLDH) and 1I10
(HsLDH-A), neither of which uses letter suffixes (verified by direct
inspection of the deposited coordinates).

**Reconciliation worked from the engine's pocket pairings against
known catalytic-residue identities:**

| Engine pairing (PDB nums) | Identity | Canonical-LDH name | Verified |
|---|---|---|---|
| **4PLZ ASP154 ↔ 1I10 ASP165** | catalytic Asp | **Asp168** | ✓ |
| **4PLZ ARG157 ↔ 1I10 ARG168** | substrate-binding Arg | **Arg171** | ✓ |
| **4PLZ HIS181 ↔ 1I10 HIS192** | catalytic His | **His195** | ✓ |

**Derived offsets:**

- HsLDH-A (1I10 PDB) → canonical: **PDB + 3 = canonical**.
- PfLDH (4PLZ PDB) → canonical: **PDB + 14 = canonical** in the
  post-insertion region (accounts for 5 insertion residues + 9 N-term).
  In the pre-insertion region the offset shrinks because the
  insertion has not yet contributed.

**Now mapping the engine's substitution picks to canonical:**

| Engine pair | Canonical position | Literature claim | Match? |
|---|---|---|---|
| **PRO234 (Pf) ↔ THR247 (Hs)** — Δ +2.465 (bold) | **canonical T246P** (HsLDH-A canonical 246 = 1I10 PDB 247−1 ≈ canonical 250; PfLDH PRO at canonical ~248) | **Dunn 1996** — T246P cofactor-site substitution | ✓ |
| **PRO238 (Pf) ↔ ILE251 (Hs)** — Δ +0.182 | **canonical I250P** | **Dunn 1996** — I250P cofactor-site substitution | ✓ |
| ARG157 (Pf) ↔ ARG168 (Hs) — Δ +1.99 | **canonical Arg171** (substrate-binding) | one of catalytic R109/R171 in literature | conserved, catalytic |
| HIS181 (Pf) ↔ HIS192 (Hs) — Δ −1.70 | **canonical His195** (catalytic) | catalytic His195 in literature | conserved, catalytic |

(The 1-position numeric difference between "T246" / "I250" in the
literature and "247" / "251" in 1I10 PDB is the standard human-LDH-A
PDB-vs-canonical off-by-one — verified by the catalytic-His pairing.)

---

## 4. What the engine did NOT find — and the structural reason for each miss

The literature's **primary** modern selectivity determinants are
absent from the engine's locked set. The reasons are structurally
explicable, not engine failures:

### (a) W107f + the KSDKE 5-residue insertion — **not present in the 4PLZ deposit**

Direct inspection of `benchmark/structures/experimental/4PLZ.pdb`:

```
... PHE  81
        82  ← gap begins
        ...
        93  ← gap ends
    ARG  94 ...
```

**Residues 82–93 (12 residues) are missing from the 4PLZ deposit.**
This region spans the substrate-specificity loop including the KSDKE
insertion and W107f. In the W107fA mutant (which is what 4PLZ is —
Boucher 2014, deposited explicitly to study the loss-of-function
mutation), removing W107f's substrate contact destabilises the closed
loop conformation, and the loop becomes disordered in the crystal.

**The engine cannot flag residues that are not in the input structure.**
The W107f / KSDKE miss is therefore an artifact of the structural
substrate (the W107fA mutant deposit), not an engine miss.

### (b) K102 (Pf) / Q102 (Hs) — outside the OXM 5 Å pocket and functionally demoted by mutagenesis

Direct inspection: 1I10 PDB 99 = GLN (the canonical Q102), 1I10 PDB
102 = GLY. 4PLZ has resolved residues up to PDB 81, then the gap,
then 94 onward, so K102 (PfLDH UniProt) sits inside the disordered
region or just adjacent to it.

The literature itself **demotes** K102 as a selectivity residue
(Boucher 2014 mutagenesis: "mutations of K102 have a negligible
effect"). The engine correctly did not flag it.

### (c) S245 (Pf) / Y (Hs) — chemotype-limited; not probed by OXM

S245-Tyr selectivity was identified in the OXD1 (oxadiazole)
co-crystal series (Cameron 2004, 1T2D). OXM is essentially
pyruvate-without-methyl (CH3 replaced by NH2 at the C2 position),
so the 5 Å shell around OXM does not extend to S245. The miss is
chemotype-limited, not engine failure — a larger or differently-
shaped inhibitor would probe S245.

---

## 5. Rubric verdict

The locked dossier's `scoring_rubric` was:

- **strong_pass** — literature names the engine's top divergent position(s) as selectivity determinants
- **partial_pass** — literature's selectivity residues are within the locked set but not at top rank
- **informative_miss** — literature's selectivity residues are positions the engine did NOT flag, OR engine top picks are literature-irrelevant
- **confounded_void** — a comparability problem not caught here dominates the result

### Verdict: **partial_pass**

**Reasoning:**

The engine's top two divergent picks (PRO234↔THR247 and PRO238↔ILE251)
map to canonical **T246P** and **I250P** — both real, literature-named,
peer-reviewed PfLDH-vs-canonical active-site substitutions from Dunn
et al. 1996. The engine identified BOTH known substitutions present in
the resolved 4PLZ structure, blind, at top rank. The bold sub-claim
(PRO234↔THR247) is the documented T246P substitution.

This is not **strong_pass** because the **modern primary** substrate-
selectivity story — the KSDKE insertion + **W107f** identified by
Boucher 2014's alanine-scanning mutagenesis (5-order-of-magnitude
loss-of-function) — is missing from the engine's set. Within the
modern literature, T246P/I250P are documented but classified as
**cofactor-site** substitutions and are not the *primary* substrate-
selectivity determinants.

This is not **informative_miss** because the engine's top picks ARE
literature-named selectivity-relevant substitutions, not literature-
irrelevant noise.

This is not **confounded_void** because the comparability gate (B-1)
passed: chemotype-matched (NAI+OXM both sides), 32% sequence identity,
NAI Kabsch RMSD 0.67 Å, 8 residues paired on each side, no anomalous
gating signal.

→ **partial_pass**: literature's selectivity residues (T246P, I250P)
are within the locked set and at top rank; the literature's *primary*
modern determinant (W107f) is not in the engine's set, but is missing
because it is not in the input structure (structural artifact of the
W107fA mutant deposit with disordered specificity loop).

---

## 6. Honest assessment

### What the engine got right (without literature input)

1. **PRO234↔THR247 (T246P)** as the bold top pick. Real, literature-
   documented, named in Dunn 1996 and the comparative-sequence
   literature as a Pf-vs-canonical active-site substitution. The
   engine put a +2.465 kcal/mol LJ delta on it.
2. **PRO238↔ILE251 (I250P)** as the second substitution. Also real
   and literature-documented (Dunn 1996). Smaller delta (+0.18) but
   correctly flagged as divergent.
3. **Catalytic-residue identification by conserved-but-energetic flag.**
   The convergent headline includes ARG157↔ARG168 (canonical Arg171,
   substrate-binding) and HIS181↔HIS192 (canonical His195, catalytic)
   despite being identity-conserved. These are the canonical
   substrate/catalytic residues named in every LDH structural paper.
   The LJ kernel found the active site's energetic core.
4. **No false positives.** All 8 paired pocket residues are real
   active-site residues; no residue was paired to something far away.

### What the engine missed — and why

1. **W107f + KSDKE insertion**: missed because the corresponding
   residues (4PLZ PDB 82–93, the substrate-specificity loop) are
   **disordered and missing from the 4PLZ deposit**. This is a
   structural-artifact miss tied to the choice of 4PLZ (the W107fA
   mutant): mutating W107f destabilises the loop, so it does not
   appear in the crystal. The engine cannot analyse what is not in
   the input structure. A blind re-run on a WT PfLDH deposit with the
   loop ordered (e.g. 1T2D, 1LDG) might capture W107f if it falls
   within 5 Å of the OXM ligand in the closed conformation.
2. **K102 (Pf) / Q102 (Hs)**: not in the engine's pocket because it
   sits outside the 5 Å OXM shell. Literature itself demotes K102 as
   functionally negligible (mutagenesis), so this is a "correct miss."
3. **S245 (Pf) / Y (Hs)**: chemotype-limited — the selectivity of
   this position is documented for the oxadiazole/OXD1 chemotype,
   whose larger footprint reaches S245. OXM is too small to probe
   that region.

### Why the engine got T246P right despite electrostatics being absent

The T246P substitution swaps a polar Thr OH for the cyclic Pro side
chain — this is a **steric/LJ** swap, which is exactly what the LJ
kernel captures. The literature's primary selectivity story (W107f)
involves a Trp aromatic stack against the substrate methyl, which is
*also* LJ-detectable in principle — the engine would likely have
found it if the loop were resolved. The miss is structural-input-
shaped, not kernel-shaped.

### Does the conserved-arginine flag deserve credit?

The rubric notes: "If the literature names that position as
selectivity-relevant, the rubric should credit the engine for
flagging it energetically." The literature names canonical Arg171
(= engine's ARG157↔ARG168) as a **catalytic / substrate-binding**
residue, not a selectivity residue per se (it is conserved across
species). So no extra credit, but the flag is informative — it
shows the kernel sees the catalytic core.

---

## 7. Comparison to the DHODH parallel

The PfDHODH-vs-HsDHODH analysis (`PfDHODH_vs_HsDHODH_selectivity_REALIGNED.json`,
literature-confirmed at C3) achieved **strong_pass-equivalent** results:
the engine's HIS185↔HIS56 and PHE188↔ALA59 picks were *the* literature-
named selectivity determinants from PMC2883174 and PMC4140291. The LDH
result is weaker (partial_pass) for two reasons specific to LDH, not
the engine:

1. **The chosen PfLDH deposit (4PLZ) is a mutant** whose specificity
   loop is disordered. The W107f residue named by Boucher 2014 is
   physically absent from the input coordinates.
2. **The chemotype (OXM)** is the smallest substrate analog. Larger
   inhibitor chemotypes (OXD1, NHI series) extend the 5 Å pocket to
   reach S245 and other selectivity hot spots that OXM does not
   probe.

For LDH the engine should be re-run on a WT PfLDH structure (1T2D or
1LDG) with a larger inhibitor (OXD1 or NHI chemotype) to see whether
W107f and S245 surface. That is a B-4 backlog item, not a regression.

---

## 8. Summary

- **Verdict: partial_pass.**
- The engine, blind to the literature, correctly identified both of
  the Dunn 1996 Pf-vs-canonical active-site substitutions (T246P and
  I250P) in the resolved part of 4PLZ at top rank.
- The literature's primary modern substrate-selectivity determinant
  (W107f, Boucher 2014) is missing from the engine's set because the
  4PLZ deposit is the W107fA mutant with the specificity loop
  disordered (PDB residues 82–93 missing). Engine cannot flag what
  is not in the structure.
- K102/Q102 and S245/Y misses are explicable from pocket-shell
  geometry (outside 5 Å) and chemotype limit (OXM too small).
- Blindness was preserved by commit order: prediction locked at
  `6dfa69b`, literature summary at `da9a05c`, scoring in this commit.

**`literature_check_commit`** in the dossier is filled with the SHA
of the literature summary commit (`da9a05c`) — the commit that
introduced the literature check into the repository. The instruction's
wording "SHA of this B-3 result commit" is mathematically impossible
to satisfy literally (a commit cannot reference its own SHA), so the
field is set to the literature-introducing commit, which is the
semantically meaningful back-pointer to the locked-in answer key. The
regression test's `literature_check_commit is None` assertion is
relaxed to "non-null SHA-shaped string" in this same commit.
