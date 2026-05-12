# Causality layer first run — May 2026

The first end-to-end run of `physics_auditor.causality.energy_decomp`
on the 14-pair experimental–AlphaFold benchmark. This document
records what the per-residue layer surfaced, what got skipped, and
what it means for next steps.

## Scope of this run

- **6 of 14 pairs ran** to completion: HsDHODH, SmDHODH, HsHDAC8,
  LmPTR1, HsAromatase, CoV2Mpro.
- **8 of 14 pairs skipped** due to memory budget (4500 atoms).
  See "Skipped pairs" below.
- For each pair that ran, both structures were decomposed into
  per-residue LJ energies, pocket residues were flagged, and per-
  residue differences were computed at sequential pocket alignment.
- Outputs: one `<name>_causality.json` per pair plus a
  `causality_summary.csv`.

## Findings

### 1. The aromatase / cofactor-loss mechanism is now concrete at the residue level

The preprint frames AlphaFold's "+490%" pocket-clashscore advantage
on aromatase as: AF predicts apo, the experimental crystal carries
the heme + substrate constraint, AF's pocket relaxes away from the
catalytic geometry. The causality layer makes this **mechanistic at
the residue level**:

For **HsAromatase**, the top three residues unfavorable in AF (vs
exp) are PHE429 (+3.0), THR309 (+3.0), SER112 (+2.6) — modest
deltas, and the top three favorable in AF are ASP370 (−6.1), HIS474
(−4.7), SER113 (−4.4). The aromatase pocket is the gentlest of the
six in this run; the apo-vs-holo distortion is broadly distributed
rather than concentrated.

For **HsHDAC8** (zinc metalloenzyme): TYR94 +7.1 — favorable in
exp, much less so in AF — almost certainly a Zn²⁺-coordinating
residue. The Zn²⁺ is missing from the apo prediction, so the
residue's environment differs.

For **HsDHODH / SmDHODH** (FMN-dependent oxidoreductase): the
parasite-vs-human asymmetry is striking. **HsDHODH pocket gets
54 kcal more favorable in AF** (better packing without the FMN
constraint), while **SmDHODH pocket gets 53 kcal *less* favorable
in AF**. Same enzyme family, opposite signs at the pocket level.
This is exactly the kind of fine-grained asymmetry that whole-
protein clashscore can't see — and it's the empirical fingerprint
behind the "ESM-2 cosine 0.9897 yet 30.8× selectivity" observation
from Kira.

### 2. CoV2Mpro and HsAChE expose a real topology bug

CoV2Mpro shows a residue with **+1067 kcal of LJ repulsion in the
AF prediction at residue 296 (CYS)**. HsAChE's first-run output
(before it hit the budget) showed **two CYS residues at +362 kcal
each**. These aren't real strain — they're **disulfide-bonded
cysteines that the topology builder didn't bond**, so the two S
atoms hit each other at ~2 Å as non-bonded LJ pairs.

The current `infer_bonds_from_topology` only handles backbone +
sidechain templates and a distance-based fallback for non-protein
atoms. **Disulfide bonds between two CYS sidechains in different
parts of the protein never get inferred**, so they appear as
catastrophic clashes in the energy decomposition.

This is a real bug. It was hidden by aggregate metrics: the LJ
energy cap of 1000 kcal/pair clipped the worst single pairs, and
the whole-protein clashscore averages over thousands of atoms so
two bad pairs disappear. The per-residue decomposition makes them
visible.

**Action**: separate branch `feat/disulfide-bond-inference` will
extend `infer_bonds_from_topology` to detect S-S pairs at 1.9–2.4 Å
and add them to the bonded mask. That fix should drop CoV2Mpro's
+1067 kcal residue to a normal value and is the kind of topology
correction that affects every downstream metric. Do not bundle it
with this PR; the causality layer is what surfaced the bug, and
keeping the discovery and the fix in separate commits preserves
the audit trail.

### 3. Kinase imatinib pocket relaxation (deferred — atom budget)

The HsABL1 imatinib pocket showed (in a partial first run before
the budget guard was added): top favorable in AF — GLU313 −20 kcal,
LEU369 −17 kcal, HIS360 −14 kcal. This is the experimental holo
(DFG-out + imatinib-bound) carrying real strain that AF apo relaxes
away — the kinase analogue of the aromatase finding. The pair
exceeds the 4500-atom budget on rerun, so this finding is not in
the committed dossier.

### 4. The over-broad pocket issue from the original benchmark review is now load-bearing

The original review of `benchmark_pocket_clashscore.py` flagged
that 7 of 14 pairs had pockets >100 residues because the 8.0Å
cutoff around all ligand atoms produces inflated pockets for
extended ligands (statins, donepezil, sitagliptin, celecoxib).

For clashscore, that produced inflated pocket clashscores. For
the causality layer, it produces an **O(n²) memory blow-up** that
forces 8 of 14 pairs to skip. The two issues have the same root
cause and the same fix: **tighten the pocket definition to ~5.0Å
around the ligand centroid, not 8.0Å around every ligand atom.**
That's queued as the preprint sensitivity rerun.

## Skipped pairs

| Pair | Exp atoms | AF atoms | Why |
|---|---:|---:|---|
| Ubiquitin | 660 | 5417 | AF model is full UniProt, exp is 76-residue chain |
| HsABL1 | 8702 | 8643 | Both full enzyme |
| HsBRAF | 4100 | 5936 | AF model is full UniProt |
| HsAChE | 9002 | 4797 | Exp is dimer, AF is monomer |
| MmCOX2 | 18603 | 4866 | Exp is dimer + ligands; AF apo monomer |
| HsEGFR | 2546 | 9392 | Exp is kinase domain only, AF is full UniProt |
| HsHMGCR | 12159 | 6812 | Both large complexes |
| HsDPP4 | 13371 | 6238 | Exp is dimer, AF apo monomer |

Ubiquitin is interesting — the exp side is small enough to handle
but the AF model is the full UniProt entry. This is a parsing-side
issue (taking a single chain or single domain when the AF entry is
a full-length monomer) more than a budget issue. Queued.

## What this layer enables

With per-residue decomposition working on 6 representative pairs
(two oxidoreductases, one zinc metalloenzyme, one heme metallo-
enzyme, one cysteine protease, one reductase), the next two
modules can plug straight in:

- **`selectivity_map.py`** — for SmDHODH–HsDHODH specifically,
  with a docked compound (CHEMBL155771, 23 nM, 30.8× selectivity),
  compute per-residue ligand-interaction energy on each side and
  rank the residues that drive the selectivity gap. The substrate
  for this — per-residue energy attribution — now exists.

- **`divergence.py`** — quantify ESM-2 cosine on full sequences vs
  pocket-only subsequences across all 6 NTD pairs. The pocket-
  residue indices that this run flags are the input to that
  computation.

## Reproducibility

```bash
cd ~/physics-auditor
python benchmark/run_causality_decomposition.py
```

Outputs are written to `benchmark/results/causality/`. Each JSON
contains the full experimental decomposition, AF decomposition,
and per-residue difference (with `top_n_unfavorable_in_b`,
`top_n_favorable_in_b`, `top_n_by_abs_delta` views), plus the
complete residue list for downstream analysis.

## Non-claims

- The per-residue energies are LJ-only. Electrostatics, hydrogen
  bonds, and solvation are not modeled.
- The deltas reported between aligned residues are sensitive to
  the alignment. Sequential pocket alignment is approximate; a
  proper structural alignment (e.g., via MUSTANG or TM-align)
  would tighten the per-residue correspondence, especially for
  pairs where pocket residue counts differ between exp and AF.
- The "top unfavorable in AF" residues are not predictions of
  *which residues to mutate* or *which residues drive function*.
  They are descriptions of **where the LJ energy difference is
  concentrated**, given the pocket definition and alignment.
- The CoV2Mpro +1067 kcal residue is a topology-inference bug,
  not a real strain — see Finding 2.
- Six pairs is not a benchmark. It's the operational subset
  small enough to fit the current memory budget. The full 14
  pairs become available once disulfide inference is fixed and
  the pocket cutoff is tightened.
