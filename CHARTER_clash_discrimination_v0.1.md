# CHARTER — clash-check native-vs-decoy discrimination (v0.1)

**Type:** pre-registration charter for a discrimination claim (a falsifiable bar fixed before any results).
**Status:** PRE-RESULTS. Written and committed before any discrimination/AUROC code runs. No discrimination result exists yet.
**Instrument under test:** the Physics Auditor's single *scored* check — `steric_clashes` (the committed clashscore). This charter governs that check's behavior as a native-vs-decoy discriminator and nothing else.

This closes two of the four audited gaps to "undeniable": **(gap 1)** no pre-registered quantitative bar, and **(gap 2)** no native-vs-decoy discrimination capability. It does **not** address calibration (gap 3) or the HTML Trust Report (gap 4) — both explicitly deferred (§9).

---

## 1. The claim (scoped)

Tested claim: *the committed clashscore discriminates native crystal structures from non-native decoys, and we pre-register exactly where it succeeds and where it fails.*

The deliverable is a **characterized operating envelope for one defensible check**, not a universal discriminator. This is deliberate: the auditor's signature is honest scope and abstention, so "undeniable" here means the check's discrimination claim is pre-registered, measured, and matches (or honestly violates) the registered expectation — not that the check discriminates everything.

**Explicitly NOT claimed by this charter:** composite-instrument discrimination; discrimination by any unscored check (LJ, Ramachandran); binding affinity / ΔΔG; calibration of the clash thresholds; the HTML report. (§9.)

## 2. The metric (direction fixed before results)

Score under test = **clashscore** (higher clashscore → more clashes → predicted *decoy*). Native is expected to carry *low* clashscore.

- **Primary:** AUROC pooling native (label 0) vs decoy (label 1) across the locked target set, with clashscore as the decoy-score. AUROC = 1.0 → clashscore perfectly ranks every decoy above every native; AUROC = 0.5 → no discrimination.
- **Secondary:** native-recognition rate (fraction of targets whose native has the lowest clashscore among {native + its decoys}); and the per-target native-vs-decoy Z-score distribution.

## 3. Two-tier design

### Tier 1 — POSITIVE CONTROL (perturbation decoys; clash MUST fire)
Purpose: prove the scored check is not broken — when structures genuinely have packing defects, clashscore must separate them.
- **Decoy protocol:** Cartesian Gaussian displacement applied to all heavy atoms of each locked native at σ ∈ {0.5, 1.0, 1.5, 2.0} Å; **K_p = 50** decoys per native per σ; **RNG seed = 20260613** (fixed, committed here before any decoy is generated).
- **Pre-registered expectation:** clashscore rises monotonically with σ; AUROC → ~1.0 at σ = 2.0 Å.
- **PASS bar:** AUROC ≥ **0.95** at σ = 2.0 Å.
- **If FAIL (< 0.95):** the clash check itself is defective on gross corruption — STOP and investigate. This would be a major, surprising finding about the check, not a tuning opportunity.

### Tier 2 — THE REAL TEST (3DRobot decoys; clash PREDICTED to fail)
Purpose: test whether clash alone discriminates *realistic, well-packed* non-native folds.
- **Decoy set:** 3DRobot (Deng H, Jia Y, Zhang Y. *Bioinformatics* 2016, 32(3):378–387; PMID 26471454; PMCID PMC5006309), the standard 200-protein non-redundant single-domain set (300 decoys/target), from the Zhang-lab distribution.
- **Pre-registered expectation (with mechanism):** AUROC ≈ **0.5**. 3DRobot decoys are explicitly engineered with enhanced hydrogen-bonding and compactness "to eliminate the possibility of native structure recognition by trivial potentials" (Deng 2016). A clashscore is a trivial steric potential; 3DRobot is adversarial to it *by construction*. Native and decoy clashscores are therefore expected to be similar (both low).
- **Discrimination bar:** AUROC ≥ **0.65** would constitute *meaningful* discrimination of well-packed decoys.
- **Pre-committed null (the expected outcome):** if AUROC < 0.65 — the registered expectation — report: *"clash alone does not discriminate native from well-packed realistic decoys; the instrument's abstention on realistic structures is justified, consistent with 3DRobot's design against trivial potentials."* This is the **expected, pre-registered result, not a failure to hide.**
- **If AUROC ≥ 0.65 (surprising):** report clashscore unexpectedly discriminates 3DRobot decoys — a positive finding to investigate (e.g., a clash-rich decoy subpopulation).

## 4. Locked sets (pre-specified before any clashscore is computed)

- **Native set:** **K = 30** targets drawn from the 3DRobot 200-protein single-domain set by **seed 20260613**, the draw performed and the 30 native PDB IDs **recorded before any clashscore is computed**. (Full 200 is acceptable only via a pre-data amendment; the charter pins K = 30 for bounded, reproducible compute.)
- **Tier 1 decoys:** perturbations (§3) of those 30 natives.
- **Tier 2 decoys:** the 3DRobot decoys for those same 30 targets.
- **Format:** all structures parsed as **PDB** (the auditor has no mmCIF parser; confirmed by audit). 3DRobot decoys are PDB-format. If any structure is not parseable as PDB it is excluded and the exclusion recorded — never silently substituted.
- **Substitution clause:** if the 3DRobot distribution is inaccessible at build time, an equivalent well-packed decoy set is substituted **only via a pre-data amendment** that states the replacement and reason (the 3HBB→3CL9 discipline). 3DRobot is the primary registered set.

## 5. Leakage / researcher-degrees-of-freedom discipline (pre-committed)

- The clash **sigmoid thresholds are FROZEN** at their current committed values and were **not** derived from or tuned to any structure in these sets. They are not adjusted after seeing results.
- Native set, decoy counts, perturbation σ-ladder, and all seeds are **fixed in this document before any clashscore is computed.**
- All **AUROC bars (0.95 / 0.65) are fixed here, before results.**
- The discrimination uses the **existing committed clashscore code, unmodified.** No new physics check is added (consistent with SCOPE_CONTRACT).
- **Reproducibility (the auditor's existing standard):** one command regenerates the full result; dependencies pinned; the 30 native IDs + decoy provenance recorded; the headline AUROC values regression-locked by a committed test, byte-for-byte.

## 6. Pre-committed outcome table

| Tier | Result | Pre-registered reading |
|---|---|---|
| 1 (perturbation) | AUROC ≥ 0.95 @ σ=2.0 | Positive control holds — the scored check fires on real packing defects. |
| 1 (perturbation) | AUROC < 0.95 @ σ=2.0 | The clash check is defective on gross corruption. STOP, investigate — a finding about the check itself. |
| 2 (3DRobot) | AUROC < 0.65 *(expected)* | Clash alone cannot discriminate well-packed decoys. Honest envelope limit; abstention justified; matches 3DRobot-by-design prediction. |
| 2 (3DRobot) | AUROC ≥ 0.65 | Clash unexpectedly discriminates 3DRobot decoys — surprising positive, investigate. |

## 7. Definition of done

Both tiers run; both bars adjudicated; the result committed, one-command-reproducible, deps pinned, AUROC values regression-locked. At that point the auditor's discrimination claim is **pre-registered, measured, and characterized** — gap 1 (a pre-registered bar exists and was cleared/adjudicated) and gap 2 (the native-vs-decoy capability now exists in the codebase) are closed. The check is undeniable **on its registered scope**: it catches packing defects (Tier 1) and honestly abstains on well-packed decoys (Tier 2), exactly as pre-registered.

## 8. What "undeniable on scope" does and does not mean

This makes the **clash check's discrimination claim** undeniable. It does not by itself make the whole instrument "undeniable" per THE_ONE_PLAN's full definition — calibration and the HTML deliverable remain. It is one rung: the discrimination-and-bar rung.

## 9. Explicitly deferred (declared, not hidden)

- **Empirical calibration (gap 3):** the clash sigmoid thresholds remain heuristic anchors; no labeled-data calibration is attempted here. Future rung.
- **HTML Trust Report (gap 4):** the report remains JSON/text; no dashboard is built here. Future rung.
- **Multi-check scoring / composite discrimination:** out of scope; the instrument still scores one check by design.

---

**Operative note:** this charter is additive to SCOPE_CONTRACT.md and CLAIMS_AND_LIMITS.md (it adds a pre-registered *quantitative* bar where those fix qualitative scope). It is committed before any discrimination code is written. A v0.2 is issued only by pre-data amendment (e.g., the §4 substitution clause).

---
**POST-RESULTS CLOSURE (2026-06-13, commit bdfac1f).** This charter was executed. Tier 1 fired §6 row 1 (AUROC 1.0 @ σ=2.0 — PASS, positive control holds). Tier 2 fired §6 row 4 (raw AUROC ≈ 1.0 — surprising positive); the required investigation found the signal is a **coordinate-provenance confound, not fold discrimination** — near-native-fold decoys (~0.5 Å RMSD) clash ~5× more than natives in 30/30 targets, so the clash check has no demonstrated fold-recognition power. The rung is closed on this honest finding (option A): the pre-registered expectation that clash cannot recognize correct folds among realistic structures holds; the high literal AUROC reflects coordinate provenance, not fold quality. Gaps 3 (calibration) and 4 (HTML) remain explicitly deferred. Full writeup: `benchmark/discrimination/DISCRIMINATION_FINDINGS.md`.
