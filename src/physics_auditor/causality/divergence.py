"""ESM-2 divergence amplification: full-sequence vs binding-site cosine.

Quantifies how much full-sequence ESM-2 cosine similarity *understates*
binding-site divergence between a target and its ortholog. The slogan
this module operationalizes:

    "SmDHODH and HsDHODH share full-sequence ESM-2 cosine 0.9897 yet
    admit ~30× experimental compound selectivity for the parasite."

If global representations average over the whole sequence, the
selectivity-driving residues — concentrated in the binding pocket — are
diluted into the mean. This module computes three cosines per pair and
reports the gap (the "divergence amplification"):

1. ``full_sequence_cosine``  — mean-pooled ESM-2 embedding over every
   residue of each sequence.
2. ``pocket_meanpool_cosine`` — same forward pass, mean restricted to
   the supplied pocket positions.
3. ``pocket_subseq_cosine``   — a separate forward pass on the pocket
   residues concatenated as a standalone subsequence.

The amplification metric is ``full_sequence_cosine − pocket_meanpool``.
A positive value means the pocket is *more divergent* than the protein
as a whole. We report all three numbers, because pocket_meanpool and
pocket_subseq disagree systematically: meanpool keeps the contextual
information the transformer infused from neighbors, while subseq strips
that context. Reporting both lets the reader see which signal is which.

NON-CLAIMS
----------
- ESM-2 cosines are not free-energy and not equilibrium constants.
  An amplification of 0.05 does not imply 30× selectivity. The
  amplification is a representation-divergence quantity that explains
  *why* a global similarity claim is misleading, not a predictor of
  affinity.
- Pocket indices are caller-supplied 0-based positions into the input
  sequence. The caller is responsible for ensuring those positions
  reference the same protein the sequence describes — this module does
  not verify that.
- Cosines depend on the ESM-2 layer used. The default model (33-layer
  650M-parameter ESM-2) returns final-layer representations; results
  from t12 or t36 will differ. We pin the model name in the report so
  reruns are reproducible.
"""

from __future__ import annotations

import os

# fair-esm imports torch, which links libomp. On macOS/conda environments
# where another OpenMP runtime (numpy via accelerate, jax via XLA) is
# already in-process, the duplicate-libomp guard aborts the interpreter.
# This opt-out is the documented workaround. We set it BEFORE importing
# torch/esm so the directive is honored on the first OpenMP-using import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass

import numpy as np


@dataclass
class DivergenceReport:
    """Result of an ESM-2 divergence-amplification computation for one
    target/ortholog pair.

    Attributes
    ----------
    pair_name : str
        Free-form identifier (e.g. ``"SmDHODH_vs_HsDHODH"``).
    full_sequence_cosine : float
        Cosine similarity of the mean-pooled ESM-2 embedding over every
        residue of each full sequence. The "global" similarity number.
    pocket_meanpool_cosine : float
        Cosine of the mean-pooled embedding restricted to the supplied
        pocket residue indices — same forward pass as full_sequence,
        different mean.
    pocket_subseq_cosine : float
        Cosine of the mean-pooled embedding from a separate forward
        pass on the pocket-only subsequence (no surrounding context).
    divergence_amplification : float
        ``full_sequence_cosine − pocket_meanpool_cosine``. Positive
        values mean the pocket is more divergent than the protein as
        a whole.
    n_pocket_residues_target : int
        Number of pocket positions used on the target side.
    n_pocket_residues_ortholog : int
        Number of pocket positions used on the ortholog side.
    model_name : str
        ESM-2 model used (e.g. ``"esm2_t33_650M_UR50D"``).
    """

    pair_name: str
    full_sequence_cosine: float
    pocket_meanpool_cosine: float
    pocket_subseq_cosine: float
    divergence_amplification: float
    n_pocket_residues_target: int
    n_pocket_residues_ortholog: int
    model_name: str


# Module-level cache of (model, alphabet, repr_layer) by model name.
# ESM-2 t33 is ~2.6 GB on disk and takes 20–30 s to load on M-series CPU;
# we never want to pay that cost more than once per process.
_MODEL_CACHE: dict[str, tuple[object, object, int]] = {}


def _load_model(model_name: str) -> tuple[object, object, int]:
    """Return cached (model, alphabet, repr_layer) for ``model_name``.

    The repr_layer is parsed from the model name (``esm2_t{N}_...`` → N)
    so callers don't have to plumb layer indices through. Models are
    placed in eval mode; CPU is implicit on M-series machines without a
    CUDA build.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        import esm
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise ImportError(
            "fair-esm is not installed. Install with "
            "`pip install -e \".[esm]\"` or run inside the bio-builder env."
        ) from exc

    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(
            f"Unknown ESM-2 model name {model_name!r}. "
            f"Expected something like 'esm2_t33_650M_UR50D'."
        )
    model, alphabet = loader()
    model.eval()

    repr_layer = None
    for part in model_name.split("_"):
        if part.startswith("t") and part[1:].isdigit():
            repr_layer = int(part[1:])
            break
    if repr_layer is None:
        repr_layer = int(getattr(model, "num_layers", 33))

    _MODEL_CACHE[model_name] = (model, alphabet, repr_layer)
    return _MODEL_CACHE[model_name]


def _embed_per_residue(sequence: str, model_name: str) -> np.ndarray:
    """Run ESM-2 forward and return ``(L, D)`` per-residue representations
    with BOS/EOS tokens stripped."""
    import torch

    model, alphabet, repr_layer = _load_model(model_name)
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("seq", sequence)])
    with torch.no_grad():
        result = model(batch_tokens, repr_layers=[repr_layer])
    reps = result["representations"][repr_layer]  # (1, L+2, D)
    # Strip BOS at position 0 and EOS at position L+1.
    per_residue = reps[0, 1 : 1 + len(sequence), :].cpu().numpy()
    return per_residue


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors. Raises on zero-norm inputs
    rather than silently returning NaN."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("Cannot compute cosine similarity on a zero-norm vector.")
    return float(np.dot(a, b) / (na * nb))


def _validate_pocket_idx(
    pocket_idx: list[int], sequence: str, label: str
) -> None:
    """Sanity-check pocket positions. Pocket indices are 0-based positions
    into the input sequence — the same convention compute_selectivity_map
    uses for residue_indices into structure.residues (which for protein-
    only structures coincides with position in the protein sequence).

    Raises
    ------
    ValueError
        If the list is empty, contains a non-integer, or any index lies
        outside ``[0, len(sequence))``.
    """
    if not pocket_idx:
        raise ValueError(f"{label} pocket index list is empty.")
    n = len(sequence)
    for i in pocket_idx:
        if not isinstance(i, (int, np.integer)):
            raise ValueError(
                f"{label} pocket index {i!r} is not an integer "
                f"(got {type(i).__name__})."
            )
        if i < 0 or i >= n:
            raise ValueError(
                f"{label} pocket index {int(i)} out of range for "
                f"sequence of length {n} (expected 0 <= i < {n})."
            )


def compute_divergence(
    target_seq: str,
    ortholog_seq: str,
    target_pocket_idx: list[int],
    ortholog_pocket_idx: list[int],
    model_name: str = "esm2_t33_650M_UR50D",
    pair_name: str = "",
) -> DivergenceReport:
    """Compute ESM-2 divergence amplification for one target/ortholog pair.

    Workflow:
        1. Validate pocket indices against the sequences they reference.
        2. One ESM-2 forward pass per sequence; cache the per-residue
           representations.
        3. Three cosines: full-sequence mean-pool, pocket mean-pool from
           the same forward pass, and pocket-subsequence mean-pool from
           a separate forward pass.
        4. Amplification = full − pocket_meanpool (positive = pocket
           more divergent than the whole protein).

    Parameters
    ----------
    target_seq, ortholog_seq : str
        Protein sequences (single-letter code) for the two sides.
    target_pocket_idx, ortholog_pocket_idx : list[int]
        0-based positions into ``target_seq`` and ``ortholog_seq`` that
        mark the binding pocket. Convention matches
        ``compute_selectivity_map``: positions are indices into the
        protein-residue ordering of the structure, equivalent to
        positions in the protein-only sequence string. Indices must
        satisfy ``0 <= i < len(seq)`` or the function raises.
    model_name : str
        ESM-2 model identifier resolvable via ``esm.pretrained``.
        Default ``"esm2_t33_650M_UR50D"`` (the 33-layer 650M model
        commonly cited in selectivity discussions).
    pair_name : str
        Free-form label recorded on the report.

    Returns
    -------
    DivergenceReport
        See class docstring.
    """
    _validate_pocket_idx(target_pocket_idx, target_seq, "target")
    _validate_pocket_idx(ortholog_pocket_idx, ortholog_seq, "ortholog")

    target_per_res = _embed_per_residue(target_seq, model_name)
    ortholog_per_res = _embed_per_residue(ortholog_seq, model_name)

    full_target = target_per_res.mean(axis=0)
    full_ortholog = ortholog_per_res.mean(axis=0)
    full_cos = _cosine(full_target, full_ortholog)

    pocket_target_mp = target_per_res[np.asarray(target_pocket_idx, dtype=int)].mean(axis=0)
    pocket_ortholog_mp = ortholog_per_res[np.asarray(ortholog_pocket_idx, dtype=int)].mean(axis=0)
    pocket_mp_cos = _cosine(pocket_target_mp, pocket_ortholog_mp)

    target_subseq = "".join(target_seq[i] for i in target_pocket_idx)
    ortholog_subseq = "".join(ortholog_seq[i] for i in ortholog_pocket_idx)
    target_sub_emb = _embed_per_residue(target_subseq, model_name).mean(axis=0)
    ortholog_sub_emb = _embed_per_residue(ortholog_subseq, model_name).mean(axis=0)
    pocket_sub_cos = _cosine(target_sub_emb, ortholog_sub_emb)

    return DivergenceReport(
        pair_name=pair_name,
        full_sequence_cosine=round(full_cos, 6),
        pocket_meanpool_cosine=round(pocket_mp_cos, 6),
        pocket_subseq_cosine=round(pocket_sub_cos, 6),
        divergence_amplification=round(full_cos - pocket_mp_cos, 6),
        n_pocket_residues_target=len(target_pocket_idx),
        n_pocket_residues_ortholog=len(ortholog_pocket_idx),
        model_name=model_name,
    )


def full_sequence_cosine(
    seq_a: str,
    seq_b: str,
    model_name: str = "esm2_t33_650M_UR50D",
) -> float:
    """Mean-pooled ESM-2 cosine between two sequences (no pocket).

    Convenience for the runner: reproduces the headline "full-sequence
    cosine = 0.9897" number from a Kira-style ortholog comparison
    without needing pocket indices.
    """
    a = _embed_per_residue(seq_a, model_name).mean(axis=0)
    b = _embed_per_residue(seq_b, model_name).mean(axis=0)
    return round(_cosine(a, b), 6)


def divergence_report_to_dict(report: DivergenceReport) -> dict:
    """Serialize a DivergenceReport to a JSON-ready dict."""
    return {
        "pair_name": report.pair_name,
        "full_sequence_cosine": report.full_sequence_cosine,
        "pocket_meanpool_cosine": report.pocket_meanpool_cosine,
        "pocket_subseq_cosine": report.pocket_subseq_cosine,
        "divergence_amplification": report.divergence_amplification,
        "n_pocket_residues_target": report.n_pocket_residues_target,
        "n_pocket_residues_ortholog": report.n_pocket_residues_ortholog,
        "model_name": report.model_name,
    }
