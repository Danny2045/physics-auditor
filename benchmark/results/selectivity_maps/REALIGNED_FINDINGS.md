# PfDHODH (4ORM) vs HsDHODH (1D3H) — sequence-anchored realignment

## The question this run asked

Does the headline PfDHODH-vs-HsDHODH selectivity finding from
`PfDHODH_vs_HsDHODH_selectivity.json` survive an *honest* (non-positional-
zip) pocket alignment?

The original dossier paired pocket residues by their position in each
structure's pocket-residue list — i.e. target pocket residue `i` was
attributed against ortholog pocket residue `i`. That works for pockets
of identical sequence and identical residue ordering. It does not work
in the general case: it silently produces a residue-to-residue
correspondence that has nothing to do with homology when pocket sizes
or numbering differ. PfDHODH and HsDHODH have offset numbering and
pockets of different sizes (21 vs 19 residues at the 5 Å cutoff), so
the positional-zip output was a candidate artifact of the alignment
method, not necessarily a real biological finding.

This realignment replaces the positional zip with a sequence-anchored
correspondence and asks which part of the original result was real.

## Method

The v1 correspondence engine
(`src/physics_auditor/causality/correspondence.py`) builds a global
pairwise sequence alignment of the two structures' full chain
sequences with a vendored Needleman–Wunsch implementation (BLOSUM62
substitution matrix, affine gap penalties, no biopython dependency),
then maps each target-pocket residue to its true sequence-homologous
counterpart on the ortholog chain via a sequence-position ↔ engine-
index round-trip. A hard consistency gate refuses to proceed if the
per-chain reconstructed sequence diverges from `Chain.sequence`
(catches interleaved-chain edge cases), and a homology refusal gate
returns `refused=True` rather than producing untrustworthy pairs when
global identity falls below 25%.

The aligner was validated against **eight synthetic known-answer
cases** in `tests/test_correspondence.py` *before* it touched a real
structure:

- two identical sequences → identity mapping;
- N-terminal insertion → correct offset recovery;
- two known point substitutions → 1:1 alignment with identity
  `(L-2)/L`;
- internal three-residue deletion → one length-3 gap, flanks aligned;
- consistency-gate pass on an in-order single-chain structure;
- consistency-gate fire on a structure whose Chain.residues order
  disagrees with the parse order;
- refusal gate fires for two unrelated sequences (poly-Ala vs poly-Trp,
  identity below threshold);
- refusal gate passes and correspondence is built for two
  identical-sequence structures.

Only after all eight tests passed was the realigned run conducted on
the real DHODH pair. This ordering — aligner-first, real-data-second
— is what lets us attribute any change in the DHODH result to honest
science, not to a bug in the new aligner.

## Alignment provenance

| Quantity | Value |
|---|---:|
| Method | `global_sequence_nw` |
| Global sequence identity (PfDHODH vs HsDHODH chains) | **0.4113 (41.1%)** |
| Identity refusal threshold | 0.25 |
| Refused? | no |
| Target pocket residues | 21 |
| Ortholog pocket residues | 19 |
| Target-pocket residues confidently paired to a homolog | **21 / 21** |
| Unpaired target-pocket residues | 0 |

The 41.1% global identity is in the expected band for PfDHODH /
HsDHODH catalytic-domain orthologs and well above the refusal
threshold. The new engine pairs all 21 target-pocket residues to
homologous counterparts (vs the old positional zip, which aligned only
the first 19 by index).

Aggregate pocket energies are identical to the original dossier:

| Quantity | Value |
|---|---:|
| Total 4ORM (PfDHODH) ligand LJ interaction | −44.73 kcal/mol |
| Total 1D3H (HsDHODH) ligand LJ interaction | −29.57 kcal/mol |
| Pocket 4ORM ligand LJ interaction | −38.07 kcal/mol |
| Pocket 1D3H ligand LJ interaction | −25.30 kcal/mol |
| Pocket delta (1D3H − 4ORM) | **+12.78 kcal/mol** |

These are unchanged because they are computed per-residue on each side
independently — the alignment affects only the *attribution* of which
target residue is paired against which ortholog residue, not the
underlying per-residue interaction energies.

## Result: top target-selective residues — old vs realigned

| Rank | Committed (positional-zip) | Δ kcal | Realigned (sequence-anchored) | Δ kcal |
|-----:|----------------------------|-------:|-------------------------------|-------:|
| 1 | HIS185 → ALA59 | +4.339 | **PHE188 → ALA59** | +2.224 |
| 2 | PHE188 → THR63 | +3.321 | **HIS185 → HIS56** | +1.970 |
| 3 | VAL532 → PRO364 | +3.006 | CYS184 → ALA55 | +1.178 |
| 4 | PHE227 → VAL134 | +2.067 | PHE227 → PHE98 | +0.915 |
| 5 | LEU531 → GLY363 | +1.688 | ILE263 → VAL134 | +0.867 |
| 6 | **ARG265 → TYR356** | +1.497 | LEU189 → VAL60 | +0.859 |
| 7 | ILE263 → TYR147 | +1.052 | LEU172 → MET43 | +0.829 |
| 8 | LEU172 → MET43 | +0.829 | GLU182 → GLU53 | +0.678 |
| 9 | LEU176 → GLN47 | +0.640 | **ARG265 → ARG136** | +0.662 |
| 10 | GLY181 → PRO52 | +0.376 | LEU176 → GLN47 | +0.640 |

The literature selectivity cluster from the Phillips/Rathod PfDHODH
series — HIS185, PHE188, ARG265 — survives the realignment: all three
remain in the top-10 target-selective residues. The ranking shuffles
(PHE188 moves to #1, HIS185 to #2, ARG265 from #6 to #9) and the
deltas drop substantially.

### What changed for the literature trio

| Target | Committed pairing & Δ | Realigned pairing & Δ |
|---|---|---|
| HIS185 | → ALA59 (Δ +4.339) — His→Ala substitution | → **HIS56 (Δ +1.970) — sequence-conserved** |
| PHE188 | → THR63 (Δ +3.321) — Phe→Thr substitution | → ALA59 (Δ +2.224) — Phe→Ala substitution |
| ARG265 | → TYR356 (Δ +1.497) — Arg→Tyr substitution | → **ARG136 (Δ +0.662) — sequence-conserved** |

Two of the three cluster residues (HIS185 and ARG265) pair against
**sequence-conserved** ortholog residues under the honest alignment.
The positional-zip dossier reported them as pairing against
chemically dissimilar substitutions (His↔Ala, Arg↔Tyr) — the
substitution narrative was partly a pairing artifact, not a real
amino-acid difference. The realigned deltas are smaller because the
ortholog side now reflects a real homolog rather than a list-position
mismatch.

The same artifact-correction effect is visible elsewhere in the
realigned top-10: PHE227 now pairs PHE→PHE (#4, Δ +0.915) instead of
PHE→VAL (#4, Δ +2.067), and several previously-large deltas
(VAL532→PRO364 at Δ +3.006; LEU531→GLY363 at Δ +1.688) move far down
the ranking because their realigned partners (THR360, LEU359) are
chemically closer.

## Interpretation — what we can say, what we cannot

**What we can say:**

- The literature selectivity cluster (HIS185, PHE188, ARG265) is
  **not** an alignment artifact of the positional zip. All three
  remain top-10 target-selective under an honest pocket-residue
  correspondence.
- The original *mechanism story* — that the cluster derived its
  selectivity from amino-acid substitutions between parasite and human
  (HIS↔ALA, ARG↔TYR) — is partly an artifact. Two of three cluster
  residues are sequence-conserved between PfDHODH and HsDHODH; their
  contribution to selectivity must come from **pocket context and
  geometry**, not from residue chemistry alone.
- An LJ-attribution dossier that pairs residues by list position rather
  than homology can systematically overstate selectivity deltas at
  conserved positions because it pairs a residue against an unrelated
  ortholog residue at the same list index.

**What we cannot yet say:**

- The HIS185↔HIS56 and ARG265↔ARG136 pairings are **sequence-anchored,
  not structure-anchored**. Sequence-conserved positions in an ortholog
  alignment are not guaranteed to occupy the same spatial position in
  the pocket — the alignment can be locally misleading if one of the
  proteins has undergone a loop rearrangement, a hinge motion, or any
  topological reshuffling that decouples sequence identity from
  spatial equivalence.
- Until the pairings are confirmed by an *independent* method, the
  conserved-residue interpretation should be treated as
  **sequence-confirmed, structurally pending**.

## The seam — sequence equivalence is not structure equivalence

This realignment exchanges one assumption (residues at the same list
index are paired) for another (residues at the same global sequence-
alignment column are paired). The second is a much better default and
catches the kinds of artifacts above, but it is still not a
*structural* equivalence. Closing that seam requires confirming
spatially that HIS56 in HsDHODH is in fact the structural counterpart
of HIS185 in PfDHODH at the pocket — i.e. that they sit in the same
place relative to the bound ligand.

Three confirmation steps, in order of leverage:

1. **In-repo Kabsch C-α superposition of the two pockets.** The
   `core/geometry` module already has the primitives. The convergent
   test is: does a structure-based residue correspondence (built by
   minimum-RMSD nearest-neighbour assignment after rigid-body
   alignment of pocket backbones) reproduce the sequence-based
   pairings reported here? Agreement confirms by two independent
   methods. Disagreement means the sequence alignment is locally
   misleading at the pocket and this finding must be revised. This is
   the next development step (v2).
2. **Published structural-biology cross-check.** The 4ORM
   crystallography paper (Deng et al., *J. Med. Chem.* 2014) and the
   PfDHODH selectivity reviews report which catalytic-domain residues
   are conserved and ligand-contacting. The realigned pairings should
   match those annotations.
3. **TM-align (or equivalent gold-standard structural aligner).** An
   external, pre-publication confirmation of residue-level structural
   equivalence between the two structures. Tools like TM-align are
   the established benchmark for residue-level superposition.

Until at least step (1) is done, the conserved-residue interpretation
in this dossier is *sequence-confirmed, structurally pending* — the
exact framing the dossier's `pending_followup` block carries.

## Non-claims

- **LJ-only.** Per-residue values are Lennard-Jones interaction
  contributions only. Electrostatics, hydrogen bonds, solvation, and
  entropy are not modeled. A residue identified here as
  target-selective is selective *in LJ terms*; the full energetic
  picture may differ.
- **Different chemotypes on each side.** PfDHODH carries
  2V6/DSM338 (triazolopyrimidine); HsDHODH carries A26/teriflunomide
  (cinnamoyl-acetamide-style enol). Both occupy the quinone tunnel but
  the chemotypes differ — part of any per-residue delta still reflects
  compound geometry, not residue chemistry.
- **Sequence-anchored, not structure-anchored.** The pairings here
  come from a global sequence alignment. They are not guaranteed to be
  structurally equivalent — see the seam discussion above.
- **No free energy of binding.** Per-residue LJ is a structural-physics
  attribution quantity, not a thermodynamic one.
