# Literature references for the PfLDH-vs-HsLDH-A selectivity blind test

Pointer-style record of the experimental and structural-biology literature
identifying the selectivity determinants between *Plasmodium falciparum*
lactate dehydrogenase (PfLDH) and human LDH-A. Citations and one-line
extracts only — no long quotes.

**This summary was written BEFORE the locked prediction dossier
(`PfLDH_vs_HsLDH-A_blind_prediction.json`, committed at 6dfa69b) was opened
in this session.** The literature read is therefore not steered by knowing
what the engine guessed. The locked prediction is opened only after this
file is committed.

## Tier scheme

- **C0** — computational only (single method).
- **C1** — multi-method computational convergence.
- **C2** — gold-standard tool result or strong multi-source corroboration.
- **C3** — peer-reviewed primary literature confirmed (mutagenesis or structural).

## Numbering convention (critical — must be reconciled before scoring)

The canonical PfLDH literature (Dunn et al. 1996; Cameron et al. 2004;
Boucher et al. 2014; Boucher & Theobald 2018) uses **dogfish-LDH-based
canonical numbering** for PfLDH active-site discussion. Because PfLDH
carries a five-residue insertion in the substrate-specificity loop, the
insertion residues are numbered with **letter suffixes appended to the
preceding canonical residue number** (e.g. 107a, 107b, 107c, 107d, 107e),
with the next non-inserted residue named 107f. This is the convention
used by all primary references in this document.

PDB residue numbers on individual deposits (e.g. 4PLZ for PfLDH; 1I10 for
human LDH-A) may differ from this canonical numbering — the structures'
SEQRES records and the published papers' active-site nomenclature do not
always coincide. **Reconciliation between PDB-residue numbers (which the
locked prediction uses) and canonical-paper numbering (which this file
uses) is required before any rubric scoring** and is performed explicitly
in the B-3 result document, not here.

## Literature answer — the selectivity determinants

The literature names **four** classes of PfLDH-vs-HsLDH-A selectivity
residues. Listed plainly in priority order:

### 1. Substrate-specificity loop insertion: **K107a, S107b, D107c, K107d, E107e** (5-residue insert, PfLDH only; absent from HsLDH-A)

- **Tier:** **C3** (structural + comparative-sequence; multiple primary papers).
- **Identity:** five-amino-acid insertion (one-letter codes K-S-D-K-E by the
  Boucher 2014 numbering) inserted into the substrate-specificity loop of
  PfLDH between canonical residues 107 and the next non-inserted position.
  Absent from all canonical (mammalian/dogfish) LDHs including HsLDH-A.
- **Role:** repositions the entire substrate-specificity loop, shifting
  which residue contacts the substrate's C3 methyl (see #2 below). This
  insertion is the foundational evolutionary event distinguishing
  apicomplexan LDHs from the ancestral MDH and from canonical LDHs.
- **Sources:**
  - Dunn et al., *Nat. Struct. Biol.* **3**, 912–915 (1996) — first PfLDH
    crystal structure; named the loop insertion (PDB **1LDG**).
  - Boucher, Jacobowitz, Beckett, Classen, Theobald, *eLife* **3**:e02304
    (2014). DOI **10.7554/eLife.02304**. PDB **4PLZ**, **4PLW**, etc.
  - Boucher & Theobald, *J. Biol. Chem.* / *Protein Sci.* (2018), bioRxiv
    **10.1101/386029**; PMC **PMC6247789**; PMID **30358994**.

### 2. **W107f** — THE specificity residue (mutagenesis-validated)

- **Tier:** **C3** (alanine-scanning + saturation mutagenesis on the loop).
- **Identity:** the tryptophan immediately following the five-residue
  insertion, numbered W107f in canonical/dogfish-based nomenclature.
- **Position in HsLDH-A:** **no direct counterpart by number** — the
  spatially analogous role (substrate methyl contact) in canonical LDHs
  including HsLDH-A is filled by **Q102** (glutamine), which sits at the
  un-shifted loop position. Boucher 2014 explicitly downgrades K102 in
  PfLDH and re-identifies W107f as the apicomplexan specificity residue.
- **Mutagenesis evidence:**
  - W107f→A reduces pyruvate activity by **five orders of magnitude**
    (Boucher 2014). Alanine scanning across the specificity loop showed
    that **only W107f** contributes significantly to LDH activity; the
    rest of the loop is tolerant of mutation (Boucher & Theobald 2018).
  - Only other large aromatics (Tyr, Phe) at 107f rescue activity
    partially; all other substitutions fail (Boucher & Theobald 2018).
- **Sources:** Boucher 2014 (above); Boucher & Theobald 2018 (above).
- **Note on the locked prediction structure:** PDB **4PLZ** is the
  **W107fA** mutant (deposited as part of the Boucher 2014 mutagenesis
  series). This is structurally relevant to scoring and is treated in
  the B-3 result file, not assumed here.

### 3. **K102 (PfLDH) ↔ Q102 (HsLDH-A)** — historic specificity candidate, now demoted but still a true substitution

- **Tier:** **C3** (mutagenesis demoted the functional importance of K102
  in PfLDH; the substitution itself is a primary structural fact).
- **Identity:** position 102 in canonical numbering is **lysine (K)** in
  PfLDH and **glutamine (Q)** in HsLDH-A. The substitution is real.
- **Historic role:** before 2014, K102 was widely cited as the PfLDH
  specificity residue. Boucher 2014 showed by mutagenesis that K102 has
  **negligible effect on activity** in PfLDH because the insertion
  re-routes the loop such that K102 is extruded from the active site.
  In canonical LDHs Q102 still contacts substrate; in PfLDH K102 does
  not. The substitution is therefore real but its functional weight has
  been transferred to W107f.
- **Sources:** Boucher 2014; Boucher & Theobald 2018 (above); historic
  citations in Dunn 1996.

### 4. **S245 (PfLDH) ↔ Y246-ish (HsLDH-A)** — drug-design selectivity hot spot

- **Tier:** **C3** (co-crystal structure of selective inhibitor + selectivity
  rationale from the medicinal-chemistry literature).
- **Identity:** PfLDH carries **Ser245** at a position where canonical
  LDHs including HsLDH-A carry **tyrosine**, with the side chain oriented
  away from the active site. This is the documented basis for selective
  oxadiazole / azole inhibitor design against PfLDH.
- **Evidence:** in the OXD1 (oxadiazole) co-crystal with PfLDH the
  inhibitor O1 forms an H-bond with Ser245 of PfLDH; this contact is not
  available in HsLDH-A because the corresponding residue is a Tyr oriented
  away from the pocket. The Ser/Tyr difference is the cited structural
  rationale for selectivity of the OXD1 chemotype.
- **Sources:**
  - Cameron, Read, Tranter, Winter, Sessions, Brady, Vivas, Easton, Kendrick,
    Croft, Barros, Lavandera, Martin, Risco, García-Ochoa, Gamo, Sanz, Leon,
    Ruiz, Gabarró, Mallo, Gómez de las Heras, *J. Biol. Chem.* **279**,
    31429–31439 (2004). PDB **1T2D**. The azole / OXD1 paper that
    established the Ser245-Tyr selectivity argument.

### 5. **Cofactor-site substitutions (background, not a primary substrate-selectivity claim)**

- **Tier:** **C3** for the substitution facts; not normally the focus of
  *substrate*-selectivity discussion but documented.
- **Identity:** canonical LDH positions equivalent to **T246** and **I250**
  are replaced by **proline** in PfLDH. These affect cofactor binding
  (NAD vs APAD preference) more than substrate selectivity.
- **Sources:** Dunn et al. 1996 and the comparative-sequence reviews
  cited above.

## Literature answer — restated plainly

If asked "which specific residues confer PfLDH-vs-HsLDH-A selectivity
in the substrate/active site?", the literature answer is, in priority
order:

1. **The five-residue insertion in the substrate-specificity loop**
   (canonical residues **107a–107e**, sequence K-S-D-K-E), present in
   PfLDH and absent from HsLDH-A. This is the structural foundation of
   all PfLDH selectivity.
2. **W107f** — the tryptophan immediately following the insertion. By
   mutagenesis the single most important specificity residue (W→A
   reduces activity ~5 orders of magnitude). Has no positional
   counterpart in HsLDH-A; functionally analogous to (but spatially
   distinct from) HsLDH-A Q102.
3. **K102 (Pf) ↔ Q102 (Hs)** — a true substitution at canonical 102,
   historically cited as the specificity residue but **functionally
   demoted** by Boucher 2014 mutagenesis.
4. **S245 (Pf) ↔ Y (Hs)** — the documented chemotype-specific selectivity
   hot spot for oxadiazole / OXD1-style inhibitors (Cameron 2004).
5. Background: **T246P, I250P** substitutions affect cofactor selectivity
   more than substrate selectivity.

## Note for scoring

The locked prediction `PfLDH_vs_HsLDH-A_blind_prediction.json` was
generated on **PDB 4PLZ (PfLDH, NAI+OXM) vs PDB 1I10 (HsLDH-A, NADH+OXM)**
and uses **PDB residue numbering**, not the canonical/dogfish numbering
used in this file. Before scoring (Step 3) the PDB residue numbers in the
prediction must be mapped to the canonical positions named here. A hit
or a miss must not be declared on the basis of a number mismatch alone.

In particular: PDB 4PLZ is the W107f**A** mutant (the Boucher 2014
mutagenesis deposit), so the actual residue type at PfLDH 4PLZ's
"specificity position" is alanine, not tryptophan. The literature claim
nonetheless concerns the wild-type W107f and is what the prediction must
be scored against. This nuance is recorded for transparency and handled
in the B-3 result file.
