"""Exact pattern statistics used by the manuscript MLP.

Label 1 denotes the target healthy lung epithelial cell type; label 0 denotes
all other healthy tissue cell types. State 2 is uncovered, not methylation.
Copied from workflow/core.py in the research implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


@dataclass(frozen=True)
class SampleTensors:
    exact_counts: np.ndarray
    marginal_counts: np.ndarray
    coverage_counts: np.ndarray
    sample_totals: np.ndarray


class PatternSpace:
    """Encoding utilities for a fixed-length 0/1/2 pattern space."""

    def __init__(self, seq_len: int):
        self.seq_len = int(seq_len)
        self.n_patterns = 3 ** self.seq_len
        self.n_masks = 2 ** self.seq_len
        self.ternary_weights = 3 ** np.arange(self.seq_len - 1, -1, -1)
        self.binary_weights = 2 ** np.arange(self.seq_len - 1, -1, -1)

        ids = np.arange(self.n_patterns, dtype=np.int64)
        self.digits = (
            (ids[:, None] // self.ternary_weights[None, :]) % 3
        ).astype(np.int8)
        self.observed_counts = np.sum(self.digits != 2, axis=1).astype(np.int8)
        self.mask_ids = (
            ((self.digits != 2) * self.binary_weights[None, :]).sum(axis=1)
        ).astype(np.int64)

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        out = np.empty(len(sequences), dtype=np.int64)
        for row, sequence in enumerate(sequences):
            text = str(sequence)
            if len(text) != self.seq_len or any(char not in "012" for char in text):
                raise ValueError(f"Invalid length-{self.seq_len} pattern: {text!r}")
            out[row] = sum(int(char) * weight for char, weight in zip(text, self.ternary_weights))
        return out

    def decode(self, pattern_ids: Iterable[int]) -> list[str]:
        return ["".join(map(str, self.digits[int(pattern_id)])) for pattern_id in pattern_ids]


def _marginalize_ternary_counts(exact_counts: np.ndarray, seq_len: int) -> np.ndarray:
    tensor = exact_counts.reshape((exact_counts.shape[0],) + (3,) * seq_len).copy()
    for position in range(seq_len):
        moved = np.moveaxis(tensor, position + 1, -1)
        moved[..., 2] = moved[..., 0] + moved[..., 1] + moved[..., 2]
    return tensor.reshape(exact_counts.shape)


def _coverage_superset_counts(coverage_counts: np.ndarray, seq_len: int) -> np.ndarray:
    tensor = coverage_counts.reshape((coverage_counts.shape[0],) + (2,) * seq_len).copy()
    for position in range(seq_len):
        moved = np.moveaxis(tensor, position + 1, -1)
        moved[..., 0] = moved[..., 0] + moved[..., 1]
    return tensor.reshape(coverage_counts.shape)


def stratified_sample_folds(sample_labels: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Assign each sample to a deterministic class-stratified fold."""
    rng = np.random.default_rng(seed)
    folds = np.full(len(sample_labels), -1, dtype=np.int16)
    for label in (0, 1):
        indices = np.flatnonzero(sample_labels == label)
        if len(indices) < n_splits:
            raise ValueError(f"Class {label} has fewer samples than n_splits={n_splits}")
        rng.shuffle(indices)
        folds[indices] = np.arange(len(indices), dtype=np.int16) % n_splits
    if np.any(folds < 0):
        raise RuntimeError("Some samples were not assigned to a fold")
    return folds


def compute_pattern_statistics(
    tensors: SampleTensors,
    sample_labels: np.ndarray,
    sample_indices: np.ndarray,
    space: PatternSpace,
    alpha: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Compute exact and conditional sample-normalized statistics for all patterns."""
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    pattern_ids = np.arange(space.n_patterns, dtype=np.int64)
    masks = space.mask_ids

    result: dict[str, np.ndarray] = {"pattern_id": pattern_ids}
    class_values: dict[int, dict[str, np.ndarray]] = {}

    for label in (0, 1):
        class_samples = sample_indices[sample_labels[sample_indices] == label]
        if len(class_samples) == 0:
            raise ValueError(f"No class-{label} samples in selected training set")

        numerator = tensors.marginal_counts[class_samples]
        denominator = tensors.coverage_counts[class_samples][:, masks]
        eligible = denominator > 0
        frequencies = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=eligible,
        )
        conditional_probability = np.divide(
            frequencies.sum(axis=0),
            eligible.sum(axis=0),
            out=np.zeros(space.n_patterns, dtype=np.float64),
            where=eligible.sum(axis=0) > 0,
        )

        exact = tensors.exact_counts[class_samples]
        exact_frequency = exact / tensors.sample_totals[class_samples, None]
        class_values[label] = {
            "conditional_probability": conditional_probability,
            "compatible_count": numerator.sum(axis=0, dtype=np.int64),
            "eligible_read_count": denominator.sum(axis=0, dtype=np.int64),
            "eligible_sample_count": eligible.sum(axis=0, dtype=np.int32),
            "support_sample_count": (numerator > 0).sum(axis=0, dtype=np.int32),
            "exact_probability": exact_frequency.mean(axis=0),
            "exact_count": exact.sum(axis=0, dtype=np.int64),
        }

    normal = class_values[0]
    tumor = class_values[1]
    for prefix, values in (("normal", normal), ("tumor", tumor)):
        for name, array in values.items():
            result[f"{prefix}_{name}"] = array

    result["conditional_loglr"] = np.log(tumor["conditional_probability"] + alpha) - np.log(
        normal["conditional_probability"] + alpha
    )
    result["exact_loglr"] = np.log(tumor["exact_probability"] + alpha) - np.log(
        normal["exact_probability"] + alpha
    )
    result["exact_total_count"] = normal["exact_count"] + tumor["exact_count"]

    # A smoothed harmonic support remains positive for one-class-zero patterns,
    # but still down-weights likelihood ratios supported weakly in either class.
    normal_support = normal["compatible_count"].astype(np.float64) + 1.0
    tumor_support = tumor["compatible_count"].astype(np.float64) + 1.0
    harmonic_support = 2.0 / (1.0 / normal_support + 1.0 / tumor_support)
    result["effective_support"] = harmonic_support
    # Kept for backward compatibility. New fitting code computes the same
    # quantity explicitly from ``effective_support`` so that the half-saturation
    # scale can be audited without changing the probability estimator.
    result["reliability_weight"] = expert_fit_weight(harmonic_support, scale=20.0)
    return result


def expert_fit_weight(effective_support: np.ndarray, scale: float = 20.0) -> np.ndarray:
    """Return the MLP fit weight; ``scale`` is its half-saturation point."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("expert-fit scale must be finite and positive")
    support = np.asarray(effective_support, dtype=np.float64)
    if np.any(~np.isfinite(support)) or np.any(support < 0):
        raise ValueError("effective support must be finite and nonnegative")
    return support / (support + float(scale))
