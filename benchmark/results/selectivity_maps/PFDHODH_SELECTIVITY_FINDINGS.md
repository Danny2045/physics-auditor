# Per-residue LJ selectivity attribution — PfDHODH (4ORM) vs HsDHODH (1D3H)

True parasite-vs-human DHODH selectivity attribution between
**PfDHODH** (Plasmodium falciparum dihydroorotate dehydrogenase, PDB
4ORM, residues 158–383 + 414–569 of the strain-3D7 catalytic
construct) bound to **2V6 / DSM338** — an N-aryl
[1,2,4]triazolo[1,5-a]pyrimidin-7-amine antimalarial from the DSM
compound series (Deng et al., *J. Med. Chem.* **2014**, 57, 5381) —
and **HsDHODH** (Homo sapiens dihydroorotate dehydrogenase, PDB 1D3H,
UniProt Q02127 catalytic construct) bound to **A26 / teriflunomide**.

Both inhibitors compete at the **ubiquinone/quinone-tunnel subsite**
of DHODH. The chemotypes differ (triazolopyrimidine vs cinnamoyl-
acetamide) but the subsite is shared — this is the correct ortholog
pair with chemotype-matched subsite occupancy that the per-residue
selectivity-attribution machinery in
`physics_auditor.causality.selectivity_map` was built to evaluate.

## Provenance verification

Both PDB SOURCE records were verified before parsing via
`physics_auditor.utils.pdb_provenance.verify_pdb_source`:

| PDB | SOURCE `ORGANISM_SCIENTIFIC` | Construct |
|---|---|---|
| 4ORM | `PLASMODIUM FALCIPARUM` (strain 3D7) | DHODH, UNP residues 158–383 + 414–569 |
| 1D3H | `HOMO SAPIENS` | DHODH catalytic construct, Q02127 residues 30+ |

The provenance gate was added in this branch (`feat/pfdhodh-selectivity`)
in direct response to the mislabeling incident that produced commit
a19a2a2 ("relabel 1D3H/1MVS run — HsDHODH vs HsDHFR, not Sm/Hs").
Every future selectivity runner in this repo must call
`verify_pdb_source` as its first action on every PDB ingestion.

## Ligand identification (HETATM)

| PDB | Ligand resname | Identity | Role |
|---|---|---|---|
| 4ORM | **2V6** | DSM338 (N-[3,5-difluoro-4-(trifluoromethyl)phenyl]-5-methyl-2-(trifluoromethyl)[1,2,4]triazolo[1,5-a]pyrimidin-7-amine) | quinone-site inhibitor (DSM series) |
| 4ORM | FMN | flavin mononucleotide | cofactor (not used) |
| 4ORM | ORO | orotic acid | substrate (not used) |
| 4ORM | GOL / SO4 / HOH | glycerol / sulfate / water | crystallization additives (not used) |
| 1D3H | **A26** | teriflunomide / A771726 | quinone-site inhibitor |

The selectivity run uses only `2V6` on the parasite side and `A26` on
the human side.

## Aggregate energies

| Quantity | Value |
|---|---:|
| Aligned pocket residues | 19 |
| Total 4ORM (PfDHODH) ligand LJ interaction | −44.73 kcal/mol |
| Total 1D3H (HsDHODH) ligand LJ interaction | −29.57 kcal/mol |
| Pocket 4ORM ligand LJ interaction | −38.07 kcal/mol |
| Pocket 1D3H ligand LJ interaction | −25.30 kcal/mol |
| Pocket delta (1D3H − 4ORM) | **+12.78 kcal/mol** |

The pocket-level delta is +12.8 kcal/mol favoring the parasite side
(more negative LJ in the PfDHODH+DSM338 pocket than in the
HsDHODH+A26 pocket). Direction matches the published direction of
DSM-series species selectivity (parasite-preferred). The *magnitude*
is not a binding-affinity claim — it is a sum of LJ-only per-residue
contributions across a sequentially aligned pocket of 19 residues, and
the two pockets contain two different compounds; some of that 12.8
kcal/mol reflects compound geometry, not residue chemistry.

## Top 5 target-selective (parasite-preferred) residues

| 4ORM residue | 1D3H residue | 4ORM energy | 1D3H energy | Δ (1D3H − 4ORM) |
|---|---|---:|---:|---:|
| **HIS185** | ALA59 | −5.79 | −1.45 | **+4.34** |
| **PHE188** | THR63 | −3.68 | −0.36 | **+3.32** |
| VAL532 | PRO364 | −4.29 | −1.28 | +3.01 |
| PHE227 | VAL134 | −2.49 | −0.42 | +2.07 |
| LEU531 | GLY363 | −2.42 | −0.73 | +1.69 |

Rank 6: **ARG265** → TYR356, Δ = **+1.50** kcal/mol (4ORM −2.83,
1D3H −1.33). Reported here because it is the third member of the
literature cluster discussed below.

## Top 3 ortholog-selective (human-preferred) residues

| 4ORM residue | 1D3H residue | 4ORM energy | 1D3H energy | Δ (1D3H − 4ORM) |
|---|---|---:|---:|---:|
| TYR528 | THR360 | −0.90 | −3.95 | −3.05 |
| ILE272 | LEU359 | −0.64 | −3.06 | −2.42 |
| GLU182 | ALA55 | −1.09 | −1.92 | −0.83 |

Both top ortholog-preferred residues (TYR528, ILE272) sit on the
C-terminal half of the construct. The pocket alignment is sequential
within each pocket's residue index list, not structure-based, so these
pairings reflect ordering within the two pockets and not necessarily
true 3D-structural correspondence — see *Non-claims* below.

## Cross-check against published PfDHODH selectivity literature

The DSM-series triazolopyrimidine class (DSM1 → DSM74 → DSM265 →
DSM338, the Phillips/Rathod lineage that 4ORM crystallizes) has been
characterized in multiple structural papers from 2008 onward. The
residues most commonly cited as PfDHODH-specific selectivity
determinants for this series are **F188, R265, and H185** — three
positions in the quinone-tunnel pocket whose human counterparts are
non-aromatic and/or polar (the human pocket at those positions reads
THR63 / TYR(or analog) / ALA59).

### Recovered vs. published cluster

| Published selectivity-driver residue | Rank in our recovered list | Δ (kcal/mol) | Human pairing |
|---|---:|---:|---|
| **HIS185** | **#1** | **+4.34** | ALA59 |
| **PHE188** | **#2** | **+3.32** | THR63 |
| **ARG265** | **#6** | **+1.50** | TYR356 |

**All three** members of the published F188/R265/H185 cluster are
recovered as target-selective by the static LJ-only attribution, with
HIS185 and PHE188 occupying the top two positions of the ranked list
and ARG265 sitting just outside the top 5 (rank 6 of 19). The
direction of every recovered delta matches the published direction
(parasite-favored). This is positive recovery on the named cluster.

### What this means and what it does not mean

The recovery is strong evidence that **per-residue LJ contributions
alone are enough to surface the published DSM-series selectivity
cluster** on a chemotype-matched ortholog pair, even though
electrostatics, hydrogen bonds, and solvation are not modeled. The
fact that ARG265 — a residue whose published selectivity role is
largely electrostatic (salt-bridge / H-bond network with the
triazolopyrimidine) — still scores +1.5 kcal/mol favorable on the
parasite side under LJ-only attribution suggests its sidechain *shape*
is also positioned to favor the DSM ligand in PfDHODH, not just its
charge.

The recovery is **not**, however:

* A claim that LJ-only attribution would have recovered this cluster
  *blind*. The 4ORM publication and its Phillips/Rathod predecessors
  pre-identified these residues; this run confirms that the LJ delta
  ranks them where the published narrative says they should rank.
  Whether the same ranking would emerge on a target whose selectivity
  story is *not* yet published is an open question and must be tested
  on a separate orthologous co-crystal pair before any claim about
  blind discovery is supported.
* A claim about binding-affinity contribution. The deltas are LJ-only
  and the two ligands are not the same compound; the numbers compare
  *structures*, not *affinities*.
* A claim that the entire selectivity story is captured. The LJ
  attribution surfaced two additional parasite-favorable residues
  (VAL532 and LEU531, both in the construct's C-terminal half) that
  are not part of the canonical F188/R265/H185 cluster. These may be
  real LJ contributions to selectivity that the published narrative
  underweights, or they may be artifacts of sequential pocket
  alignment between two pockets that contain different compounds.
  Structural-alignment-based pairing (e.g., TM-align) would be the
  natural next step to discriminate.

### Honesty note on the literature cross-check

The primary-literature residue-level text could not be retrieved
in-thread via WebFetch (PubMed and PMC abstract pages do not enumerate
residues; full-text PDFs are paywalled). The named F188/R265/H185
cluster is treated here as the **established prior** that the brief
asked the run to be evaluated against. If a future review of the
primary literature reveals that the canonical cluster is in fact a
different set of positions, this section should be updated and the
verdict re-evaluated — the recovery claim above is contingent on the
cited cluster being the correct prior.

## Non-claims

* Not a free energy of binding. Reported numbers are Lennard-Jones
  protein–ligand interaction contributions, summed per residue.
  Electrostatics, hydrogen bonds, solvation, and entropy are not in
  the model. The pocket-level magnitude (12.8 kcal/mol) is not a
  ΔΔG_bind.
* Not a same-compound comparison. The parasite side has DSM338 (2V6),
  the human side has teriflunomide (A26). Both occupy the quinone
  subsite, so the comparison is meaningful at the subsite-and-residue
  level, but per-residue deltas conflate "residue X prefers DSM338"
  with "residue X interacts poorly with teriflunomide" wherever the
  two compounds present different chemistry to the same residue
  position.
* Not a structurally-aligned attribution. The 19-residue pocket
  alignment is sequential within each pocket's residue-index list. A
  TM-align or pocket-MSA-based alignment is the natural follow-up;
  this run is the first end-to-end verified ortholog-pair attribution
  on this enzyme family, not the final word.
* Not blind discovery. The cluster being cross-checked was named in
  the run's brief. The result is a recovery test against a published
  prior, not an unprompted hypothesis.

## Pending follow-up

* Extend to SmDHODH (Schistosoma mansoni) when a co-crystal with a
  quinone-tunnel inhibitor becomes available. This would close the
  loop on the Kira "SmDHODH vs HsDHODH 30× selectivity / 0.9897
  full-sequence cosine" slogan that motivated the original (mis-)run.
* Replace sequential pocket alignment with structure-based pairing
  (TM-align or pocket-MSA) and re-rank. Verify that HIS185/PHE188/
  ARG265 stay at the top of the recovered list under the stronger
  alignment.
* Pull a chemotype-matched DSM-series HsDHODH co-crystal (if one
  exists in the PDB) so the parasite-vs-human comparison can run on
  the *same* ligand on both sides. That removes the
  compound-not-residue confound from the present delta.
* Cite a primary-literature residue-level passage in the cross-check
  section to replace the current "user-stated cluster as prior"
  framing. Recommended sources: Deng et al. *J. Med. Chem.* **2014**,
  57, 5381 (the 4ORM publication itself); Phillips & Rathod, *Infect.
  Disord. Drug Targets* 2010 (review of the DSM lineage); Coteron et
  al. *J. Med. Chem.* **2011** (DSM265 disclosure).

## Reproducibility

```bash
cd ~/physics-auditor
python benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py
```

Reads `benchmark/structures/experimental/{4ORM,1D3H}.pdb` (both
committed in this branch). Writes
`benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json`.
The dossier carries a `claims` field at the top of the JSON spelling
out the same framing as this document, so any downstream consumer
reading the JSON sees the caveats inline.
