# Clash-check native-vs-decoy discrimination — results (charter v0.1)

- Charter: `CHARTER_clash_discrimination_v0.1.md (6dad2dc)`; native draw lock: `575a4af`
- Clashscore source: `physics_auditor.checks.clashes.check_clashes (committed, unmodified)`
- ClashConfig (frozen, untouched): `{'vdw_tolerance': 0.4, 'ignore_bonded_neighbors': 3, 'severe_clash_threshold': 0.2}`
- Preprocessing (identical natives/decoys): parse_pdb defaults (keep_hydrogens=False → H stripped; keep_altloc='A'); then infer_bonds_from_topology → build_bonded_mask → compute_distance_matrix → check_clashes; identical for natives and all decoys.
- Seed: `20260613` | numpy `2.4.4` | jax `0.9.2` | repo `575a4af`
- Targets: 30 scored of 30 requested
- Excluded targets: 0

## Tier 1 — perturbation positive control (clash MUST fire)

RNG: numpy.random.default_rng([20260613, target_index, sigma_index]); target_index over sorted(native_ids), sigma_index over [0.5,1.0,1.5,2.0]; decoy = native_coords + N(0,sigma), float32
K_p=50 per native per σ; native mean clashscore = 8.2582

| σ (Å) | AUROC | mean decoy clashscore | n decoys |
|------:|------:|----------------------:|---------:|
| 0.5 | 1.000000 | 462.3424 | 1500 |
| 1.0 | 1.000000 | 1057.0399 | 1500 |
| 1.5 | 1.000000 | 1425.2087 | 1500 |
| 2.0 | 1.000000 | 1567.5850 | 1500 |

**Tier 1 bar: AUROC ≥ 0.95 at σ=2.0 — observed 1.000000 → PASS.**
Total Tier 1 decoys scored: 6000 (excluded: 0).

## Tier 2 — 3DRobot decoys (the real test; expected AUROC ~0.5)

- **Pooled AUROC = 0.999996** (bar ≥ 0.65 → MEETS bar)
- Native-recognition rate = 1.000000 (30/30 targets where native has the lowest clashscore)
- Native z-score vs decoy distribution ((native_clashscore - decoy_mean) / decoy_std; negative = native below decoy mean):
  mean=-4.365, median=-4.1801, min=-5.9409, max=-3.0172, std=0.7938 (n=30)
- Scored: 30 natives + 9000 decoys; decoys excluded (parse failures): 0

## Post-hoc near-native control (NOT pre-registered)

POST-HOC (not pre-registered): for each target, the single lowest-RMSD 3DRobot decoy (closest to the native fold) vs the native clashscore. If near-native-fold decoys still clash far more than the native, the Tier-2 separation is a preparation/provenance signal (experimentally-refined natives vs computationally-generated decoys), not fold-quality discrimination.

- Lowest-RMSD decoy per target: mean min-RMSD = 0.555 Å (max 0.95 Å) across 30 targets
- Mean native clashscore = 8.2582 vs mean lowest-RMSD-decoy clashscore = 42.2854
- Lowest-RMSD decoy clashscore EXCEEDS its native in 30/30 targets

Reading: if even the closest-to-native decoys clash far more than the native, the Tier-2 separation reflects experimental refinement vs computational generation (a preparation/provenance confound), not native-fold recognition. See DISCRIMINATION_FINDINGS.md.

## Charter §6 pre-committed outcome mapping

- **Tier 1:** AUROC 1.0000 ≥ 0.95 @ σ=2.0 — positive control holds: the scored check fires on real packing defects. [charter §6 row 1]
- **Tier 2:** AUROC 1.0000 ≥ 0.65 — clash UNEXPECTEDLY discriminates 3DRobot decoys: surprising positive, investigate (e.g. a clash-rich decoy subpopulation). [charter §6 row 4]

