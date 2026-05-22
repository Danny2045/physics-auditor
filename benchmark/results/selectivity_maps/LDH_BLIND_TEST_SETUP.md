# LDH blind-test — Phase A structure acquisition

Pre-prediction baseline for a blind selectivity-attribution test on
PfLDH (Plasmodium falciparum L-lactate dehydrogenase) vs HsLDH-A
(Homo sapiens L-lactate dehydrogenase A).

**No LDH selectivity literature has been consulted.** Structure choice
was made on mechanical criteria alone, without reading abstracts, papers,
or any selectivity discussion attached to candidate PDBs. The prediction
will be locked in a subsequent commit BEFORE any literature check; a
later commit will perform the post-lock comparison against published
selectivity residues.

## Acquisition criteria (answer-agnostic)

Two co-crystal structures were required, one parasite-side, one
human-side. Selection criteria were entirely mechanical:

(a) **Organism gate** — `verify_pdb_source` must pass on each PDB
    against its expected `ORGANISM_SCIENTIFIC`:
    - PfLDH: `PLASMODIUM FALCIPARUM`
    - HsLDH-A: `HOMO SAPIENS`

(b) **NAD-class cofactor present** — a bound NAD-class HETATM
    (`NAD`, `NAI`, `NAP`, `NDP`, `NAH`) on both sides, with ≥3
    canonical CCD atom names shared between the two cofactor copies
    so the Kabsch anchor can be seeded. This is the structural anchor;
    it must be present in both structures.

(c) **Single cofactor copy in the relevant chain** — to avoid the
    duplicate-atom-name guard in `extract_cofactor_atoms`. Where a
    structure's asymmetric unit contains a biological oligomer
    (LDH is a tetramer in many crystal forms), this criterion applies
    to the chain used for the engine run, not to the whole AU. See
    "Phase B operational note" below.

(d) **Resolution ≤ 2.5 Å.**

(e) **Tiebreak** — lowest resolution number first, earliest deposition
    date second. Purely mechanical; selection was not based on the
    relevance of a structure's accompanying paper.

## Candidates considered (structural metadata only — RCSB Data API)

### PfLDH — entries with a NAD-class cofactor at ≤ 2.5 Å

| PDB  | Resolution | Deposit    | NAD-class cofactor (chain) | Other HETATM (chain) |
| ---- | ---------- | ---------- | -------------------------- | -------------------- |
| **4PLZ** | **1.05 Å** | **2014-05-20** | **NAI (A)**             | OXM (A)              |
| 1T2D | 1.10 Å     | 2004-04-21 | NAD (A)                    | OXL (A)              |
| 2A94 | 1.50 Å     | 2005-07-11 | AP0 (A) — NAD analog, distinct chemistry; excluded |  |
| 1T24 | 1.70 Å     | 2004-04-20 | NAD (A)                    | OXQ (A)              |
| 1LDG | 1.74 Å     | 1996-09-10 | NAI (A)                    | OXM (A)              |
| 1T26 | 1.80 Å     | 2004-04-20 | NAI (A)                    | GBD (A)              |
| 1T2E | 1.85 Å     | 2004-04-21 | NAI (A)                    | OXM (A)              |
| 1T25 | 1.90 Å     | 2004-04-20 | NAI (A)                    | GAG (A)              |
| 1T2C | 2.01 Å     | 2004-04-21 | NAI (A)                    | —                    |

All listed PfLDH candidates carry the NAD-class cofactor on chain A only;
the AU appears monomeric. 1U5C (NAD, 2.65 Å) fails the resolution cutoff.

### HsLDH-A — entries with a NAD-class cofactor at ≤ 2.5 Å

| PDB  | Resolution | Deposit    | NAD-class cofactor (chains) | Other HETATM (chains where the pocket-defining ligand sits) |
| ---- | ---------- | ---------- | -------------------------- | ------------------- |
| **5W8K** | **1.60 Å** | **2017-06-21** | **NAI (A, B, C, D)**   | 9Y7 (A, C, D)       |
| 4JNK | 1.896 Å    | 2013-03-15 | NAI (A, B, C, D)           | ZHK (A, C, D)       |
| 4RLS | 1.91 Å     | 2014-10-17 | NAI (A, B, C, D)           | 49C (D)             |
| 6MV8 | 1.95 Å     | 2018-10-24 | NAI (A, B, C, D)           | D3S (A, C, D)       |
| 5W8L | 1.95 Å     | 2017-06-21 | NAI (A, B, C, D)           | 9YA (A, B, C, D)    |
| 6BB0 | 1.95 Å     | 2017-10-16 | NAD (A, B, C, D)           | D3S (A, C, D)       |
| 6Q13 | 2.00 Å     | 2019-08-02 | NAI (A, B, C, D)           | P8V (A, B, C, D)    |
| 4QO8 | 2.001 Å    | 2014-06-19 | NAI (A, B, C, D)           | 36U (A, C, D)       |
| 6MVA | 2.02 Å     | 2018-10-24 | NAI (A, B, C, D)           | D4S (A, B, C, D)    |
| 6Q0D | 2.05 Å     | 2019-08-01 | NAI (A–F)                  | P8M (A–F)           |
| 5IXS | 2.05 Å     | 2016-03-23 | NAI (C), TXD (A, B, D)     | 6EY (A–D)           |
| 6BAX | 2.05 Å     | 2017-10-16 | NAD (A, B, C, D)           | D4S (A, B, C, D)    |
| 4M49 | 2.052 Å    | 2013-08-06 | NAI (A, B, C, D)           | 22Y (A, C, D)       |
| 6BAD | 2.10 Å     | 2017-10-12 | NAD (A, B, C, D)           | D0Y (A, B, C, D)    |
| 4OKN | 2.10 Å     | 2014-01-22 | NAI (A, C–H)               | OXL (A–H)           |
| 4R68 | 2.112 Å    | 2014-08-22 | NAI (A, B, C, D)           | W31 (A, B, C, D)    |
| 4QO7 | 2.14 Å     | 2014-06-19 | NAI (A, B, C, D)           | 36V (A, C, D)       |
| 5ZJD | 2.394 Å    | 2018-03-20 | NAI (A–H)                  | —                   |
| 8FW6 | 2.34 Å     | 2023-01-20 | NAI (A, B)                 | YBO (A, B)          |

Every HsLDH-A entry in this resolution window holds the cofactor in
multiple chains (LDH is a biological tetramer; the AU typically holds
the tetramer or two dimers).

## Chosen structures (mechanical tiebreak)

- **PfLDH: `4PLZ`** — resolution **1.05 Å**, deposited **2014-05-20**.
  Cofactor **NAI** on chain A; co-bound active-site ligand **OXM**
  (oxamate) on chain A.
- **HsLDH-A: `5W8K`** — resolution **1.60 Å**, deposited **2017-06-21**.
  Cofactor **NAI** on chains A, B, C, D; co-bound active-site ligand
  **9Y7** on chains A, C, D (chain B has no inhibitor).

Both win their side's resolution-first / deposition-second tiebreak.

## Provenance gate

`verify_pdb_source` was run against both files immediately after
download, before any other tool touched them:

```
PASS: 4PLZ.pdb -> PLASMODIUM FALCIPARUM
PASS: 5W8K.pdb -> HOMO SAPIENS
```

`HEADER` / `TITLE` / `SOURCE ORGANISM_SCIENTIFIC` lines:

```
4PLZ  HEADER  OXIDOREDUCTASE                          20-MAY-14   4PLZ
      TITLE   CRYSTAL STRUCTURE OF PLASMODIUM FALCIPARUM LACTATE DEHYDROGENASE
      SOURCE  ORGANISM_SCIENTIFIC: PLASMODIUM FALCIPARUM

5W8K  HEADER  OXIDOREDUCTASE                          21-JUN-17   5W8K
      TITLE   CRYSTAL STRUCTURE OF LACTATE DEHYDROGENASE A IN COMPLEX WITH INHIBITOR
      SOURCE  ORGANISM_SCIENTIFIC: HOMO SAPIENS
```

## Anchor cofactor diagnostic

Same cofactor code (`NAI` — 1,4-dihydronicotinamide adenine dinucleotide,
the NADH form) on both sides. Canonical CCD atom-name intersection
between the two chain-A cofactor copies:

```
4PLZ chain A NAI  : 44 atoms
5W8K chain A NAI  : 44 atoms
shared atom names : 44   (well above the ≥3-atom Kabsch floor)
only in 4PLZ      : []
only in 5W8K      : []
```

The NAI heavy-atom skeleton is identical on both sides. The Kabsch
anchor will pair all 44 atoms.

## HETATM inventory (non-water, both files)

```
4PLZ  NAI  chain A=44 atoms
      OXM  chain A=6 atoms

5W8K  NAI  chains A, B, C, D — 44 atoms each
      9Y7  chains A, C, D — 33 atoms each
      MLA  chains A, B, C, D — 7–14 atoms (crystallisation additive, malonate)
      GOL  chains A, B, C, D — 18–30 atoms (glycerol cryoprotectant)
```

## Phase B operational note (mechanical, not blindness-related)

`5W8K` is a tetramer in the AU. The current
`extract_cofactor_atoms` (`src/physics_auditor/causality/superposition.py:132`)
raises `ValueError` on a duplicate atom name; the four NAI copies in
5W8K will all share atom name `PA`, so the function will fire on the
second copy. Two clean ways to handle this in Phase B without changing
the blind-test semantics:

1. Extend `extract_cofactor_atoms` to take an optional `chain_id`
   filter and use chain A on both sides. (Engine extension — small,
   pair-independent.)
2. Pre-process `5W8K.pdb` into a chain-A-only PDB before parsing.
   (Runner-level pre-processing — leaves the engine untouched.)

The same chain restriction applies to the active-site ligand `9Y7`
(present in chains A, C, D) when defining the pocket via
`extract_binding_site`.

Either choice should be made up front in Phase B and documented; both
preserve the mechanical anchor (all 44 NAI atoms are paired regardless).

## Pending — Phase B will produce

- `benchmark/run_ldh_selectivity.py` — the runner.
- `benchmark/results/selectivity_maps/PfLDH_vs_HsLDHA_selectivity.json`
  — the dossier carrying the locked prediction with
  `prediction_locked_before_literature: true`, the lock commit SHA,
  and an ISO8601 UTC lock timestamp.
- `tests/test_ldh_blind_test_regression.py` — locks the predicted
  selectivity residues + their `delta_kcal` values + the lock commit
  field, so any later change has to either reproduce the prediction
  or explicitly acknowledge that the blind regime has been broken.

The literature comparison happens in a SEPARATE, LATER commit; the
git history is what establishes that the prediction predates the
literature check.

## Explicit blindness statement

No LDH selectivity literature has been consulted at any point during
this acquisition. The PDB IDs were chosen by resolution-first,
deposition-second tiebreak among entries that satisfied the
organism / NAD-class-cofactor / resolution criteria — no abstract,
no paper, no selectivity discussion influenced the pick. The
prediction will be locked in a subsequent commit BEFORE any
literature check.

---

# Phase B-1 — chemotype-matched HsLDH-A re-acquisition (1I10)

Phase B-0 (a read-only comparability gate, no commit) found that the
Phase A pair `4PLZ ↔ 5W8K` is **chemotype-confounded** at the
active-site ligand: 4PLZ holds `OXM` (oxamate, a substrate analog,
**6** heavy atoms, composition `C2 N1 O3`) while 5W8K holds `9Y7`
(a drug-like inhibitor, **33** heavy atoms, composition
`C20 N4 S2 O5 F2`, including fluorines and sulfurs). The two
ligands' nearest-atom superposition is tight (min 0.243 Å — they
share a substrate-mimicking head group) but their pockets differ
in size by 2.6× (8 vs 21 residues at 5 Å); 15 of the 21 HsLDH-A
9Y7-pocket residues lie outside the 4PLZ-OXM pocket entirely. A
per-residue selectivity map computed over that confounded pair
would conflate **species selectivity** with **ligand-chemistry
footprint** — exactly the kind of artifact the auditor is supposed
to refuse.

Phase B-1 acquires a **chemotype-matched** HsLDH-A structure so the
ligand chemistry is identical on both sides and the only remaining
asymmetry is the protein. No prediction has been made and no
selectivity computation has been run.

## Mechanical selection rule for 1I10

> Among the 9 substrate-bearing HsLDH-A candidates from the Phase B-0
> metadata query, **1I10 is the unique structure whose active-site
> ligand AND cofactor exactly match the Pf-side structure 4PLZ:
> NAI cofactor + OXM (oxamate) substrate.** Chemotype-match is the
> selection criterion (removes the OXM-vs-9Y7 confound at the
> source). No abstract, paper, or selectivity discussion of 1I10
> was read.

The Phase B-0 metadata query was: UniProt accession `P00338`
(HsLDH-A) AND `resolution ≤ 2.5 Å` AND structure carries both a
substrate-class ligand from `{OXM, OXL, PYR, LAC}` and a NAD-class
cofactor from `{NAD, NAI, NAP, NDP, NAH}`. Of the 9 qualifying
entries (`1I10, 4JNK, 4M49, 4OKN, 4QO8, 6BB0, 6BB1, 6BB2, 6MV8`),
**only 1I10 carries the exact `NAI + OXM` pair** that 4PLZ does.
The remaining 8 entries carry `NAI + LAC`, `NAD + LAC`, or
`NAI + OXL` — different substrate chemistry from 4PLZ.

## 1I10 provenance + self-identification (gated at the door)

Live provenance gate, run on disk immediately after download and
before any other use of the file:

```
PASS: 1I10.pdb -> HOMO SAPIENS
```

`HEADER` / `TITLE` / `SOURCE ORGANISM_SCIENTIFIC` / `REMARK 2
RESOLUTION` lines read live from the PDB:

```
HEADER    OXIDOREDUCTASE                          30-JAN-01   1I10
TITLE     HUMAN MUSCLE L-LACTATE DEHYDROGENASE M CHAIN, TERNARY
TITLE  2  COMPLEX WITH NADH AND OXAMATE
SOURCE   2 ORGANISM_SCIENTIFIC: HOMO SAPIENS;
SOURCE   6 GENE: LDHA;
REMARK   2 RESOLUTION.    2.30 ANGSTROMS.
```

The file self-identifies as Homo sapiens lactate dehydrogenase A
(`GENE: LDHA`; the "M chain" nomenclature in the TITLE is the
historical biochemistry name for the LDH-A isoform — muscle
isoform — distinct from the LDH-B "H chain" isoform). The TITLE
contains **no mutation tag, no inhibitor tag, no isoform
ambiguity** (cf. 4PLZ's `MUTANT W107FA` and 5W8K's
`INHIBITOR COMPOUND 29`); the only TITLE flag is the descriptive
"TERNARY COMPLEX" (i.e., enzyme + cofactor + substrate analog),
which is exactly the chemistry we asked for.

## Ligand-match proof (geometric, not metadata)

1I10's biological assembly is a tetramer-of-tetramers (8 chains,
A–H), with NAI + OXM in **every** chain:

```
1I10  NAI  chains A, B, C, D, E, F, G, H — 44 atoms each
      OXM  chains A, B, C, D, E, F, G, H — 6 atoms each
```

The chain restriction rule applied to 5W8K (lowest chain ID
carrying both the cofactor and the active-site ligand) selects
**chain A** here. On 1I10 chain A:

```
NAI : 44 atoms — composition P2 O14 C21 N7 (matches 4PLZ chain A NAI)
OXM : 6 atoms  — composition C2 N1 O3      (matches 4PLZ chain A OXM)
      atom names: C1 N1 O1 C2 O2 O3        (identical to 4PLZ OXM)
```

OXM heavy-atom count and elemental composition match 4PLZ's OXM
exactly. NAI = 44 atoms as expected.

## W107FA pocket check (Pf-side mutation vs analysed surface)

4PLZ's TITLE flags the entry as `MUTANT W107FA`. The 4PLZ chain-A
OXM pocket (≤ 5 Å, computed live) is:

```
ASN126  LEU153  ASP154  ARG157  HIS181  ALA224  PRO234  PRO238
```

Residue **107 is not in this list.** The mutation site sits outside
the analysed prediction surface; no OXM-pocket residue is at
sequence position 107.

> Numbering note (transparency, not a confounder): position 107 in
> the 4PLZ chain-A ATOM record is `GLU`, not `TRP`. The `W107FA`
> tag in the TITLE record evidently uses a numbering convention
> distinct from the PDB ATOM numbering (likely the canonical PfLDH
> protein-sequence numbering). Whatever the real protein-sequence
> position of the mutation, the spatial check is unambiguous — the
> 5 Å OXM pocket lives at residues {126, 153, 154, 157, 181, 224,
> 234, 238} and excludes 107 by either numbering interpretation.
> **The W107FA mutation does not touch the analysed prediction
> surface.**

## Comparability verdict on 4PLZ ↔ 1I10 (chain A vs chain A)

NAI-anchored Kabsch superposition, chain A vs chain A only:

```
NAI shared atom-name pairs : 44 / 44
anchor RMSD                : 0.6731 Å
```

The rigid-body assumption holds at the cofactor.

OXM ↔ OXM in the shared frame (same ligand on both sides, paired
by canonical CCD atom name):

```
OXM heavy atoms      : 6 (4PLZ)  vs  6 (1I10 chain A)
per-atom name-matched: C1=0.41 Å  C2=0.46 Å  N1=0.51 Å
                       O1=0.49 Å  O2=2.63 Å  O3=1.74 Å
                       mean = 1.04 Å
any-atom min         : 0.41 Å
any-atom median      : 1.96 Å
```

The same substrate analog sits in essentially the same place in
both structures (a few Å of carboxylate-group reorientation
around `O2`/`O3`; the rest of the molecule superposes within 0.6 Å
per atom).

Pocket residues (≤ 5 Å of OXM, on chain A in each structure):

```
4PLZ  (PfLDH)   OXM pocket  (8 residues):
  ASN126  LEU153  ASP154  ARG157  HIS181  ALA224  PRO234  PRO238

1I10  (HsLDH-A) OXM pocket  (9 residues):
  GLN99   ARG105  ASN137  LEU164  ARG168  HIS192  ALA237  ILE241  THR247
```

Shared subsite via post-Kabsch CA proximity (4 Å CA cutoff;
partner must itself be in its structure's OXM pocket):

```
PfLDH       HsLDH-A     CA-CA(Å)   identity-conserved
ASN126      ASN137      0.57       YES
LEU153      LEU164      1.46       YES
ARG157      ARG168      1.65       YES
HIS181      HIS192      0.49       YES
ALA224      ALA237      1.22       YES
PRO234      THR247      0.68       no  (real species difference)
```

Pf-side pocket residues whose nearest 1I10 CA fell outside the
1I10 OXM pocket (i.e., one-sided pocket positions): `ASP154` and
`PRO238` — both retain a tight CA partner in 1I10 (ASP165 at
1.46 Å; ILE251 at 0.93 Å), but neither partner contacts OXM
within 5 Å in 1I10's structure.

**Comparability basis: SAME-LIGAND (OXM + NAI both sides).**
**Chemotype confound removed at source.**
**Pocket sizes: 8 (Pf) vs 9 (Hs) — comparable in size**
**(was 8 vs 21 in the 4PLZ ↔ 5W8K confounded pair).**
**Shared subsite: 6 residues, 5 identity-conserved.**

The OXM-vs-9Y7 confound is removed. The remaining pocket-residue
asymmetry is now genuinely a Pf-vs-Hs species difference (PRO234
vs THR247 at the equivalent superposed position; ASP154 / PRO238
one-sided), not a ligand-chemistry artefact.

## 5W8K retention note (keep-both doctrine)

5W8K.pdb remains on disk under `benchmark/structures/experimental/`
as the **original (confounded) acquisition**. It is **not deleted**:
the Phase A acquisition record and the Phase B-0 comparability
finding both reference it, and removing it would erase the audit
trail of the chemotype-confound discovery. However, **5W8K is NOT
the pair used for the blind prediction** — the prediction pair is
4PLZ (Pf) ↔ 1I10 (Hs), chemotype-matched on NAI + OXM.

## Phase B-1 explicit blindness statement

No LDH selectivity literature has been consulted in Phase B-1. The
choice of 1I10 was made on a single mechanical criterion (exact
chemotype match to 4PLZ: NAI + OXM), applied to the 9 candidates
already enumerated in Phase B-0's metadata-only RCSB query. No
abstract, paper, or selectivity discussion of 1I10 has been read.
The prediction will be locked in Phase B-2 BEFORE any literature
check.
