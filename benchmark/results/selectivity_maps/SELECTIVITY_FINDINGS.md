# Per-residue LJ attribution — 1D3H vs 1MVS (May 2026)

A first end-to-end run of the selectivity-attribution module
(`physics_auditor.causality.selectivity_map`) on real co-crystal data.
The run exercises the machinery — pocket extraction, per-residue
protein–ligand LJ energy, alignment, ranked delta attribution — on two
PDB co-crystals that have ligands in directly comparable subsites.

## Important framing correction

This run was originally written up as "SmDHODH vs HsDHODH selectivity."
That framing was a label error caught during the
`feat/divergence` review on 2026-05-11. The structures are not a
parasite-vs-human ortholog pair:

| | PDB | Protein (SOURCE record) | Bound inhibitor |
|---|---|---|---|
| Target | 1D3H | **HsDHODH catalytic construct** (Homo sapiens dihydroorotate dehydrogenase, UniProt Q02127 residues 30+) | A26 / teriflunomide |
| Ortholog | 1MVS | **HsDHFR** (Homo sapiens dihydrofolate reductase) | DTM pyrido[2,3-d]pyrimidine antifolate |

Both PDB SOURCE records read `ORGANISM_SCIENTIFIC: HOMO SAPIENS`. The
two are different human enzymes binding different inhibitor chemotypes
at functionally analogous nucleotide/flavin sites. This is a
**functional paralog comparison**, not an ortholog comparison.

The 30× selectivity / 99% ESM-2 cosine slogan from the Kira program
refers to *real* SmDHODH (UniProt G4VFD7) versus *real* HsDHODH
(UniProt Q02127); the divergence runner reproduces that slogan exactly
(`full_sequence_cosine = 0.989732`) on those true sequences. See
`benchmark/results/divergence/summary.json`.

## What the per-residue attribution shows

The numerical results are valid for what is actually compared (1D3H
vs 1MVS) — the energies are computed correctly; only the framing was
wrong. The headline observation:

| 1D3H residue | 1MVS residue | 1D3H energy | 1MVS energy | Δ (1MVS − 1D3H) |
|---|---|---:|---:|---:|
| HIS56 | GLU30 | −3.82 | +0.76 | **+4.58** |
| LEU359 | LEU67 | −3.06 | −0.35 | +2.71 |
| THR360 | VAL115 | −3.95 | −1.58 | +2.37 |
| MET43 | ILE7 | −0.93 | +1.10 | +2.03 |
| PRO52 | ASP21 | −1.95 | −0.35 | +1.60 |

The tool surfaces a **polar-to-polar residue substitution** (HIS56 ↔
GLU30 at the dominant aligned active-site position) as the largest
per-residue LJ contributor between HsDHODH+A26 and HsDHFR+DTM. The
machinery picks up the substitution cleanly without being told to look
there — that is the sanity claim this run supports.

What this is NOT:

- It is **not** a selectivity-design hypothesis. Medicinal chemistry
  does not design selectivity around different-enzyme-different-drug
  comparisons. A HIS↔GLU difference observed between HsDHODH bound to
  teriflunomide and HsDHFR bound to a pyrimidine antifolate tells you
  the two enzymes have different chemistries at their respective
  active sites — it does not predict what will happen if the same
  compound is exposed to both enzymes.
- It is **not** a parasite-vs-human selectivity number. The 30×
  selectivity Kira reports is between *Schistosoma mansoni* DHODH and
  *Homo sapiens* DHODH bound to the same compound class — neither of
  which is the comparison in this dossier.
- It is **not** a free energy of binding. The reported numbers are
  Lennard-Jones interaction energy contributions, a structural-physics
  quantity. Reporting them as binding affinity contributions would be
  incorrect.

## Aggregate energies (still valid as numbers, with reframed meaning)

| Quantity | Value |
|---|---:|
| Aligned pocket residues | 19 |
| Total 1D3H (HsDHODH) ligand LJ interaction | −29.57 kcal/mol |
| Total 1MVS (HsDHFR) ligand LJ interaction | −35.36 kcal/mol |
| Pocket 1D3H ligand LJ interaction | −25.30 kcal/mol |
| Pocket 1MVS ligand LJ interaction | −26.17 kcal/mol |
| Pocket delta (1MVS − 1D3H) | −0.87 kcal/mol |

Both pockets sit at favorable LJ totals (a real inhibitor is bound on
each side; the magnitudes are sensible). The pocket-level delta
averaging to near zero is the expected outcome of comparing two
different compounds in two different pockets — the per-residue
attribution carries the structural signal, not the totals.

## Pending follow-up

A true selectivity attribution requires a **parasite-DHODH co-crystal
paired with an HsDHODH co-crystal carrying a comparable inhibitor
chemotype**. The current checkout has neither side of that pair (no
SmDHODH or PfDHODH co-crystal in `benchmark/structures/`). Candidates
identified in the `run_selectivity_map.py` header:

- **6Q86** (if confirmed SmDHODH) — direct ortholog
- **PfDHODH co-crystals 4ORM / 4RX0** — published-orthology alternative
  with HsDHODH; well-characterized in the antimalarial DHODH literature

paired with a chemotype-matched HsDHODH co-crystal (e.g. 1D3H bound to
the same triazolopyrimidine class). Until that data lands, the
per-residue *selectivity* claim is **on hold**; the per-residue
*attribution* machinery is exercised and working.

## Reproducibility

```bash
cd ~/physics-auditor
python benchmark/run_selectivity_map.py
```

Reads `benchmark/structures/experimental/{1D3H,1MVS}.pdb`, both
committed in the repo. Writes
`benchmark/results/selectivity_maps/HsDHODH_vs_HsDHFR_pocket_attribution.json`.
The dossier carries a `claims` field at the top spelling out the same
framing as this document, so any consumer reading the JSON sees the
caveats inline.
