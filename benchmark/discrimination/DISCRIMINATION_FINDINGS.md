# Clash-check discrimination — findings (charter v0.1)

Hand-authored interpretation of `discrimination_results.json` /
`discrimination_results.md`. Governing pre-registration:
`CHARTER_clash_discrimination_v0.1.md` (`6dad2dc`); native draw locked at
`575a4af`. Scoring used the committed `physics_auditor.checks.clashes.check_clashes`
**unmodified**, with frozen default `ClashConfig` (`vdw_tolerance=0.4`,
`ignore_bonded_neighbors=3`, `severe_clash_threshold=0.2`); natives and all
decoys went through one identical preprocessing path (`parse_pdb` defaults →
`infer_bonds_from_topology` → `build_bonded_mask` → `compute_distance_matrix` →
`check_clashes`). Seed 20260613; numpy 2.4.4; jax 0.9.2. 30/30 targets scored,
0 exclusions (6000 Tier-1 + 9000 Tier-2 decoys).

## Headline numbers

| Tier | Metric | Result | Bar | §6 row |
|---|---|---|---|---|
| 1 (perturbation positive control) | AUROC @ σ=2.0 | **1.000000** | ≥ 0.95 | **row 1 — PASS** |
| 2 (3DRobot, the real test) | pooled AUROC | **0.999996** | ≥ 0.65 | **row 4 — surprising positive** |

Tier-1 dose-response (native mean clashscore **8.26**), monotonic as charter §3
predicted:

| σ (Å) | AUROC | mean decoy clashscore |
|------:|------:|----------------------:|
| 0.5 | 1.000000 | 462.34 |
| 1.0 | 1.000000 | 1057.04 |
| 1.5 | 1.000000 | 1425.21 |
| 2.0 | 1.000000 | 1567.59 |

Tier-2 secondaries: native-recognition rate **1.000000** (30/30 — the native
has the lowest clashscore among {native + its 300 decoys} for every target);
native z-score vs the decoy distribution mean **−4.37** (median −4.18, range
[−5.94, −3.02]). Every native sits ~3–6 σ below its decoy cloud.

## Which charter §6 row fired

- **Tier 1 → row 1 (positive control holds).** Perturbing all heavy atoms by
  Gaussian noise drives clashscore up monotonically and separates perturbed
  from native with AUROC 1.0 at every σ. The committed clash check fires
  correctly on real packing defects — it is not broken.
- **Tier 2 → row 4 (surprising positive).** The pre-registered expectation was
  AUROC ≈ 0.5 (charter §3: 3DRobot decoys are "engineered … to eliminate the
  possibility of native structure recognition by trivial potentials," and a
  clashscore is a trivial potential). Instead clashscore separates native from
  3DRobot decoy **near-perfectly** (AUROC 0.999996). Per charter §3/§6 this is
  the surprising-positive branch and triggers investigation — done below.

## Investigation: the surprising positive is a preparation confound, not fold recognition

A near-perfect AUROC does **not** mean the clashscore recognises the correct
fold. The pre-registered near-native control (post-hoc, also in the results
JSON) settles it:

- For each target, take the **single lowest-RMSD 3DRobot decoy** — the one
  closest to the native fold. Mean min-RMSD across the 30 targets is **0.555 Å**
  (max 0.95 Å): these decoys are, geometrically, essentially the native fold.
- Their mean clashscore is **42.29** vs the native mean **8.26** — about **5×
  higher**. The near-native decoy out-clashes its own native in **30/30
  targets**.

If a decoy that is 0.5 Å from the native fold still clashes ~5× more than the
native, then the clashscore is **not** separating "correct fold" from "wrong
fold." It is separating **experimentally-refined coordinates** (the crystal
native, refined against density to near-zero clash) from **computationally-
generated coordinates** (every 3DRobot decoy, including the near-native ones,
carries a residual clash signature from the generation/repacking process —
plausibly aggravated by 3DRobot's deliberate compactness optimisation). The
discriminator keys on *provenance / preparation*, not on fold quality.

This is, precisely, the failure mode the auditor exists to name: a near-perfect
number that means something other than what it appears to mean. Reported as a
fold-quality discriminator, AUROC 0.999996 would be a serious overclaim.

## What we can and cannot say

**Can say.**
- The committed clash check is mechanically sound: it fires on genuine packing
  defects (Tier 1, AUROC 1.0, clean dose-response).
- On this exact benchmark, clashscore separates 3DRobot natives from decoys with
  AUROC ≈ 1.0.

**Cannot say.**
- That clashscore recognises native folds. The near-native control falsifies
  that reading: ~0.5 Å-RMSD decoys are separated just as cleanly, so the signal
  is preparation provenance, not fold correctness.
- That the auditor would discriminate a *refined* near-native model from a
  *refined* wrong model — this benchmark cannot test that, because natives and
  decoys differ in preparation, not only in fold.

**Net.** The charter's underlying scientific expectation — *clash alone does not
recognise correct folds among realistic structures* — is upheld, even though the
literal Tier-2 AUROC came out high rather than ~0.5. The high AUROC is a
confound, and the honest envelope limit stands: a clashscore is a packing-defect
detector, not a fold discriminator. The auditor's abstention on realistic
structures remains justified.

## What would disentangle fold from preparation (deferred, not done here)

- Re-score after applying **identical** light energy-minimisation to *both*
  natives and decoys (removes the refinement asymmetry); see whether any
  native-vs-decoy separation survives.
- Within decoys only, regress clashscore on RMSD-to-native: if clashscore is
  flat in RMSD (as the near-native control suggests), it carries no fold signal.
- Use decoys that are themselves experimentally-refined alternative structures,
  eliminating the crystal-vs-generated provenance gap.

These are out of scope for charter v0.1 (which pre-registered only the two-tier
AUROC test) and would require a dated amendment.

## Reproducibility

```bash
# data: charter §1 archive (https://zhanggroup.org/3DRobot/decoys/3DRobot_set.tar.bz2)
# extracted to ~/3DRobot_data/3DRobot_set/ (or set PHYSICS_AUDITOR_3DROBOT_DIR)
python benchmark/discrimination/run_discrimination.py
```

Deterministic: same seed + committed clashscore → identical AUROCs (verified by
re-run, byte-stable). Headline values are regression-locked in
`tests/test_discrimination_regression.py`.
