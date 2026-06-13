#!/usr/bin/env python3
"""Deterministic native-set draw for the clash-discrimination charter (v0.1 §4).

PRE-SCORING ARTIFACT. This script performs the pre-registered K=30 native draw
from the 3DRobot 200-protein single-domain set and nothing else. It contains
**no** clashscore, AUROC, perturbation, or scoring code of any kind — the
charter's pre-registration wall requires the 30 target IDs to be fixed before
any clashscore is ever computed (CHARTER_clash_discrimination_v0.1.md §4, §5).

Basis: ``sorted_200_targets.txt`` — the 200 canonical 3DRobot target IDs
(``<PDBID><chain>``), enumerated from the distribution's own per-target archive
names (``3DRobot_set/<ID>.tar.bz2``) and sorted lexicographically.

Draw: ``numpy.random.default_rng(20260613).choice(sorted_ids, size=30,
replace=False)``. numpy's Generator selects indices from a PCG64 stream that
depends only on the seed and population size (200), so the result is stable
across numpy versions; the sorted basis maps those indices to IDs.

Run from anywhere::

    python benchmark/discrimination/draw_native_set.py

It re-derives the 30 IDs and (re)writes ``native_set_30.txt`` next to itself.
The output is byte-stable; running it must reproduce the committed lock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SEED = 20260613  # charter v0.1 §4 — fixed before any draw
K = 30  # charter v0.1 §4 — native-set size

HERE = Path(__file__).resolve().parent
BASIS_FILE = HERE / "sorted_200_targets.txt"
OUTPUT_FILE = HERE / "native_set_30.txt"


def load_sorted_basis(path: Path) -> list[str]:
    """Load the 200 canonical target IDs and return them lexicographically sorted.

    Parameters
    ----------
    path : Path
        Path to ``sorted_200_targets.txt`` (one ID per line).

    Returns
    -------
    list[str]
        The 200 unique target IDs, sorted (re-sorted defensively so the draw is
        invariant to any reordering of the on-disk file).
    """
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != 200:
        raise ValueError(f"expected 200 basis IDs, found {len(ids)} in {path}")
    if len(set(ids)) != 200:
        raise ValueError("basis IDs are not unique")
    return sorted(ids)


def draw_native_set(sorted_ids: list[str], seed: int = SEED, k: int = K) -> list[str]:
    """Draw ``k`` native target IDs without replacement, in rng.choice order.

    Parameters
    ----------
    sorted_ids : list[str]
        The lexicographically sorted 200-ID basis.
    seed : int
        RNG seed (default: charter seed 20260613).
    k : int
        Number of targets to draw (default: 30).

    Returns
    -------
    list[str]
        The ``k`` drawn IDs in the order returned by ``rng.choice``.
    """
    rng = np.random.default_rng(seed)
    return [str(x) for x in rng.choice(sorted_ids, size=k, replace=False)]


def main() -> None:
    sorted_ids = load_sorted_basis(BASIS_FILE)
    drawn = draw_native_set(sorted_ids)
    OUTPUT_FILE.write_text("\n".join(drawn) + "\n")
    print(f"numpy {np.__version__}")
    print(f"basis {len(sorted_ids)} IDs from {BASIS_FILE.name}")
    print(f"seed {SEED}, K={K}, method numpy.random.default_rng().choice(replace=False)")
    print(f"wrote {OUTPUT_FILE.name} ({len(drawn)} IDs, draw order):")
    print(" ".join(drawn))


if __name__ == "__main__":
    main()
