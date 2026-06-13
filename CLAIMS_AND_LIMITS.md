# What Physics Auditor Claims — and Does Not

*The claims boundary for this repository. Read alongside `SCOPE_CONTRACT.md` (the
pre-registered scope) and the per-result findings under `benchmark/results/`. Where
this document states a claim, the authoritative numbers live in the cited findings
file and in the code — this is the claims layer, not a restatement of the mechanics.
The source of truth for what is scored is `CHECK_REGISTRY` in
`src/physics_auditor/composite.py`.*

*Pre-registered discrimination bar:* `CHARTER_clash_discrimination_v0.1.md`
adds a **quantitative** native-vs-decoy AUROC bar for the one scored check
(`steric_clashes`), fixed before any discrimination code — addressing audit
gaps 1 and 2, with empirical calibration and the HTML Trust Report explicitly
deferred. It is additive to the qualitative scope here and changes neither the
scored-check count (still one) nor any claim in this document.

## In one line

Physics Auditor is a structural-claims adjudicator: it scores one physics check it
can defend, reports others without scoring them, and withholds its verdict entirely
when it cannot see enough of the structure to judge. An instrument that abstains, not
an engine that always answers.

## What it does

- **Scores exactly one check** — steric clashes. With a single scored check the
  composite's normalized weight is 1.0, so `global_score` equals the clash subscore
  exactly. The composite *structurally refuses* to weight any unscored check (exact
  set-equality guard in `compute_composite`) — enforced by code, not convention.
- **Computes but does not score two** — Lennard-Jones interaction energy (total,
  per-atom, per-residue rollups + hot-pair count) and backbone φ/ψ/ω dihedrals (only
  the counts reach the report; the angle values are not surfaced). Neither is folded
  into the score. Scoring LJ would require inventing a normalization rubric; that is
  deferred, not faked.
- **Withholds the score on incomplete input.** A readiness gate runs first. It flags
  internal missing residues — from SEQRES-vs-ATOM differences, or numeric
  residue-numbering gaps when no SEQRES record is present — and, for ligand-bound
  structures, internal gaps in the pocket region (terminal gaps do not trigger it).
  When the gate returns *unready*, `global_score`, `composite`, and `recommendation`
  are all set to `None`; the raw per-check results are still computed and reported.
  The auditor abstains rather than emit a confident verdict on a structure it cannot
  fully see. This is *residue-level* completeness only — it detects missing residues,
  not missing atoms or side chains, protonation/hydrogens, altloc/occupancy, ligand
  completeness, or atom typing; a structure that passes the gate may still be incomplete
  in those respects.

## What it does NOT prove

- The score is a **steric-clash quality heuristic** — not binding free energy, not
  ΔΔG. The LJ term, where reported, is a short-range contact approximation only; it
  ignores electrostatics, solvation, entropy, and protein motion.
- A passing score does **not** mean a structure is "correct." It means one heuristic
  found no disqualifying clashes on a structure complete enough to judge.
- Per-residue LJ attribution yields **hypotheses consistent with** known
  selectivity-relevant residues — not ligand-conditioned proof of a binding mechanism.
- It runs **one scored check, not eight.** The eight-slot registry is mostly unbuilt
  or unscored (see below).
- Nothing here implies any **wet-lab validation** or experimental confirmation. All
  evidence to date is computational.

## What it has actually demonstrated (with bounds)

**PfDHODH vs HsDHODH** — *Plasmodium falciparum* DHODH (4ORM) against human DHODH
(1D3H), both provenance-verified before parsing. Per-residue LJ attribution recovers
a known antimalarial DHODH selectivity cluster (the HIS185 / PHE188 / ARG265 family).
Bounds, stated plainly: not binding free energy, not a same-compound comparison, and
not a blind result for PfDHODH (the cluster was within the evaluation context). A
sequence-anchored realignment then corrected the original story — HIS185 and ARG265
pair to *conserved* residues (HIS56, ARG136) — spatially near-coincident after
superposition — so *their* contribution traces to pocket context and geometry rather
than substitution — while PHE188 is a genuine Phe→Ala substitution; the cluster
survived the top-10, but the original "substitution explains selectivity" reading did
not survive intact. Two of three confirmation steps (Kabsch, literature) are closed;
TM-align is outstanding. Numbers and residue maps:
`benchmark/results/selectivity_maps/PFDHODH_SELECTIVITY_FINDINGS.md`,
`REALIGNED_FINDINGS.md`, `LITERATURE.md`.

**LDH blind test → W107f confirmation** — PfLDH vs HsLDH-A, prediction locked before
the literature was read. Scored *partial_pass*: it found the T246P/I250P-class
substitutions blind, but missed W107f — because the chosen 4PLZ deposit (a W107fA
mutant) was missing residues 82–93, the substrate-specificity loop; those residues
were physically absent from the coordinates. The miss was input-shaped, not
kernel-shaped. A pre-registered follow-up confirmed it: on a wild-type structure
(1LDG) with the loop ordered, TRP107 was present, within 5 Å of the substrate, and
flagged divergent. A clean, fully-worked instance of the discipline this tool exists
to enforce — lock the prediction, score against the answer, admit the miss, explain
it falsifiably, confirm. Details: `LDH_BLIND_RESULT.md`, `W107F_MECHANISTIC_CHECK.md`.

## Relation to existing physical-validity tooling

Static physical and chemical plausibility is a mature area served by established
tooling. For predicted *ligand poses*, PoseBusters (Buttenschoen et al.,
*Chem. Sci.* 2024, 15, 3130–3139) is a widely used RDKit-based suite that checks
chirality, stereochemistry, bond lengths and angles, ring planarity, internal and
external clashes, and an energy ratio against relaxed conformers (used, e.g., within
the PLINDER evaluation pipeline). For *protein geometry*, MolProbity-style validation
has long covered all-atom clashes, Ramachandran, and rotamer outliers. This auditor's
single scored check — steric clashes — is a standard structure-QC check that overlaps
with, and is no broader than, that established clash validation; it adds nothing those
tools do not already do. The not-implemented checks listed below (chirality, bond
geometry, peptide planarity, …) are likewise long-addressed in that tooling. What this
auditor adds is not broader coverage but a narrower discipline — deriving its
partial-ness from an explicit check registry, and withholding the verdict when
residue-level completeness is insufficient (the readiness gate; see its scope above).
It is not a competitor to PoseBusters or MolProbity on static validity and should not
be described as one.

## Non-claims (explicit)

Physics Auditor must **not** be described as doing any of the following. This list is
binding in the same sense as `SCOPE_CONTRACT.md`'s WILL-NOT list — crossing it
requires a dated amendment, not a quiet change of phrasing.

- computing binding free energy, ΔΔG, or binding affinity
- validating that a structure is correct, or certifying a structure
- running 8 active physics checks (it scores 1, computes-without-scoring 2, has not
  built 5)
- proving a selectivity *mechanism* (it generates residue-level hypotheses)
- implying experimental / wet-lab validation of any result
- supporting mmCIF, or any parser/feature not present in the code
- asserting the identity of a structure that has not passed provenance verification

## Deferred — honest absence, not roadmap

- **Five unbuilt checks** — bond geometry, peptide planarity, chirality, rotamer
  outliers, disulfide-geometry. Each has an unused config-dataclass placeholder; none
  has implementing code. Not to be built as a side effect of other work.
- **LJ scoring and Ramachandran scoring** — LJ computes an interaction energy and
  Ramachandran extracts backbone φ/ψ/ω dihedrals (counts only); neither is scored.
  Wiring a subscore requires a defensible normalization rubric (and, for Ramachandran,
  a reference distribution). Deferred until that rubric exists.
- **True SmDHODH / HsDHODH pocket divergence** — gated on a parasite-side co-crystal
  structure that does not currently exist. The earlier schistosome causality dossier
  was archived as mixed-provenance, not used as evidence.
- **TM-align** — the third, highest-rigor PfDHODH correspondence check, still
  outstanding.
- **`STATE_OF_PHYSICS_AUDITOR.md`** — a deliberately dated snapshot, currently behind
  HEAD; to be refreshed at the consolidation pass.
