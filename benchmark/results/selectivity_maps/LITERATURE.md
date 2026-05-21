# Literature references for the PfDHODH/HsDHODH selectivity dossiers

Pointer-style record of the external sources that confirm — or refine — the
findings in this directory's three dossiers (the v0 positional-zip dossier,
the v1 sequence-anchored realigned dossier, and the v2 Kabsch convergent-
validation dossier). Citations only, not quoted text.

## Tier scheme

- **C0** — computational only (single method).
- **C1** — multi-method computational convergence (e.g. v1 sequence + v2 Kabsch + ORO cross-check).
- **C2** — gold-standard tool result or strong multi-source corroboration.
- **C3** — peer-reviewed primary literature confirmed (fetched + cited).

## Reference table — per finding

### HIS185 (Pf) ↔ HIS56 (Hs) — conserved catalytic histidine, binding determinant on both sides

- **Tier:** **C3** (experimental mutagenesis on both species).
- **Sources:**
  - Site-directed His→Ala mutagenesis of human DHODH that explicitly names His56 as the human counterpart of Pf His185; Pf His185 mutation raises IC50 by 500–1000×. ScienceDirect **S0006295297001974** (1998).

### ARG265 (Pf) ↔ ARG136 (Hs) — conserved H-bonding / salt-bridge arginine; Arg265 affects inhibition

- **Tier:** **C3**.
- **Sources:**
  - Pf side (mutagenesis of Arg265, effect on inhibition): ScienceDirect **S0006295297001974** (1998).
  - Hs side (Arg136 listed as binding-site salt-bridge residue): ACS Omega **10.1021/acsomega.5c12978** (2025); BMC Pharmacol. Toxicol. **10.1186/s40360-025-01007-w** (2025).

### HIS185 and ARG265 conserved across all class 2 DHODH enzymes; PHE188 and PHE227 are stacking residues

- **Tier:** **C3**.
- **Sources:**
  - Plasmodium DHODH review **PMC2883174** (Phe227+Phe188 stacking; H-bonds H185+R265 in all three structures examined).

### PHE188 (Pf) ↔ ALA59 (Hs) — spatially-equivalent substitution; stacking selectivity residue on the Pf side; resistance-mutation site

- **Tier:** **C2-strong**, with **structural-margin anchor-dependence noted** (see structural-margin note below).
- **Sources — Pf side:**
  - **PMC2883174** (Phe188 stacking).
  - **PMC4140291** — in vitro resistance-selection study reporting F188I and F188L resistance mutations.
- **Sources — Hs side:**
  - HsDHODH binding-site residue lists including Ala59 — ACS Omega **10.1021/acsomega.5c12978** and **10.1021/acsomega.5c01536**; BMC Pharmacol. Toxicol. **10.1186/s40360-025-01007-w**.
- **Note:** the exact "Pf F188 ≡ Hs A59" identification at this granularity is corroborated independently on each side and via the v2 structural convergence, rather than asserted by a single source the way His185 ≡ His56 was.

### Ranked selectivity residues match experimental PfDHODH inhibitor-resistance mutations

- **Tier:** **C3** (experimental functional validation of the residue set).
- **Source:** in vitro resistance-selection study **PMC4140291** — reports resistance mutations **E182D, F188I, F188L, F227I, I263F, L531F**. All map to residues surfaced by the v1 realigned engine (Glu182, Phe188, Phe227, Ile263, Leu531).

### Structural logic of FMN-anchoring: Class-2 inhibitor tunnel runs from the surface to adjacent to FMN

- **Tier:** **C3** (architecture / DHODH structural-biology literature).
- **Source:** HsDHODH conformational-flexibility structural study — confirms why an FMN-anchored superposition reaches the inhibitor pocket geometrically (the inhibitor tunnel terminates next to the FMN site, so FMN anchoring places the pocket-CA frames adjacent to each other).

### Methodological lead (v2 backlog) — same-compound HsDHODH control structure

- **Finding:** PDB **4OQV** is HsDHODH bound to the SAME compound (DSM338) as 4ORM is bound to.
- **Tier:** **C3** (deposited crystal structure + primary paper).
- **Sources:** RCSB PDB **4OQV**; Deng et al., *J. Med. Chem.* **2014**, 57, 5381–5394 — **10.1021/jm500481t**.
- **Why it matters:** 4ORM-vs-4OQV would remove the chemotype confound that 4ORM-vs-1D3H carries (1D3H is teriflunomide, not DSM338). If the v1 selectivity dossier is rerun on the same-compound pair, the LJ-attribution delta is no longer contaminated by inhibitor geometry, only by residue chemistry/context. High-priority next structural target for the selectivity engine.

## Structural-margin note (from v2 Phase 2b per-residue distances)

The per-cluster-residue Kabsch CA-CA distances refine the convergence story:

- **HIS185 → HIS56**: unambiguous under both anchors. Nearest CA 1.39 Å (FMN) / 1.73 Å (ORO); runner-up 3.75 Å / 3.42 Å. Margins **2.36 Å (FMN) / 1.69 Å (ORO)**. Not cutoff-sensitive.
- **ARG265 → ARG136**: unambiguous under both anchors. Nearest CA 1.43 Å (FMN) / 1.89 Å (ORO); runner-up 3.74 Å / 4.29 Å. Margins **2.31 Å (FMN) / 2.40 Å (ORO)**. Not cutoff-sensitive.
- **PHE188 → ALA59**: **clean under FMN anchor** — nearest 1.76 Å vs runner-up 3.54 Å, margin **1.78 Å**. **Cutoff-sensitive under the ORO anchor alone** — nearest 2.35 Å vs runner-up 2.81 Å, margin **only 0.47 Å**. Pairing is correct and agrees across all methods, but its tight structural margin is FMN-anchor-dependent. This is biologically coherent: Phe188 is the one **substituted** cluster residue, so its local environment diverges most between species, consistent with a softer spatial match. This is *why* two independent anchors matter — FMN resolves what ORO alone leaves near the edge.

Two pocket-wide soft-flags (>2.5 Å nearest distance), both **outside the literature cluster**, recorded for completeness — they do not affect the finding:

- **LEU172 → MET43** under FMN: 3.536 Å nearest, 3.940 Å runner-up.
- **LEU176 → GLN47** under ORO: 2.818 Å nearest, 2.963 Å runner-up.

## Honest self-corrections

Fetched evidence partially walked back two memory-based claims that earlier
work in this project was framed around. Recording them here for intellectual
honesty.

1. **"The inhibitor tunnel is the most conformationally variable part of DHODH."** Stated from memory in earlier framing. Refined by fetched data: the tunnel shows flexibility *between* different inhibitor complexes (different bound ligands push the tunnel walls into different conformations), but any given **bound** complex is fairly rigid — bound HsDHODH studies report RMSF < 1.5 Å. The rigid-body superposition assumption underlying v2 is **better supported** than the original caveat implied, because both 4ORM and 1D3H are bound complexes (the regime where the tunnel is least mobile).

2. **The rigidity caveat overall was stated more strongly than the evidence warranted.** The v2 data — FMN anchor RMSD = 0.26 Å, pocket-mean post-superposition match = 1.24 Å, and the tight per-residue cluster margins from Phase 2b (HIS185 and ARG265 ≈ 2 Å margins, PHE188 1.78 Å under FMN) — combined with the literature RMSF figures, **supports rigidity for the bound state at the cluster positions**. The pre-v2 framing of "structurally pending" remains appropriate as a process gate, but the actual data are consistent with rigidity, not borderline.

## Status of the v1 "structurally pending" framing

The v1 realigned dossier's `pending_followup` framed the conserved-residue
interpretation (HIS185↔HIS56, ARG265↔ARG136, plus the PHE188↔ALA59
substitution) as **sequence-confirmed, structurally pending**, naming
three required confirmation steps:

1. In-repo Kabsch convergent-validation — **done** (v2 Phase 2 + 2b).
2. Published structural-biology cross-check — **done** (this document; tiers C3 across the cluster, C2-strong for Phe188).
3. TM-align gold-standard structural alignment — outstanding (would still be the highest-rigor independent check; remains in the backlog).

The first two of the three named confirmation steps are complete and converge
on the v1 finding. The third (TM-align) remains genuinely outstanding and is
the cleanest residual gate. Both the v1 dossier and the v2 dossier reference
this document; the v1 historical record itself is preserved (its regression
test locks numbers and cluster pairings, not prose).
