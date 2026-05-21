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
