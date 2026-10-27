"""Train the three fixed-marker MLPs on healthy-tissue fragment patterns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import fit_small_mlp, predict_small_mlp
from .patterns import (
    PatternSpace,
    SampleTensors,
    _coverage_superset_counts,
    _marginalize_ternary_counts,
    compute_pattern_statistics,
    expert_fit_weight,
    stratified_sample_folds,
)


SOURCE_GROUPS = {
    "alveolar": "Lung-Alveolar-Epithelial",
    "bronchial": "Lung-Bronchus-Epithelial",
}
SEED = 142
EPOCHS = 30
ALPHA = 1e-4
EXPERT_FIT_SCALE = 20.0


def read_counts(value: str, k: int, expected_depth: int) -> dict[str, int]:
    raw = json.loads(value)
    counts = {str(pattern).replace("?", "2"): int(n) for pattern, n in raw.items()}
    if any(len(p) != k or set(p) - {"0", "1", "2"} or n <= 0
           for p, n in counts.items()):
        raise ValueError("invalid masked pattern counts")
    if sum(counts.values()) != expected_depth:
        raise ValueError("PAT multiplicities do not sum to informative_reads")
    return counts


def reference_tensors(frame: pd.DataFrame, k: int, source: str):
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    if len(frame) != 205 or frame.sample_id.duplicated().any():
        raise ValueError("expected 205 distinct healthy-tissue reference samples")
    space = PatternSpace(k)
    exact = np.zeros((len(frame), space.n_patterns), dtype=np.int64)
    coverage = np.zeros((len(frame), space.n_masks), dtype=np.int64)
    for row_idx, row in enumerate(frame.itertuples(index=False)):
        if int(row.k) != k:
            raise ValueError("marker length mismatch")
        counts = read_counts(row.masked_pattern_counts, k, int(row.informative_reads))
        for pattern, n in counts.items():
            pattern_id = int(space.encode([pattern])[0])
            exact[row_idx, pattern_id] += n
            coverage[row_idx, space.mask_ids[pattern_id]] += n
    totals = exact.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("reference sample has no informative reads")
    labels = frame.group.eq(source).to_numpy(dtype=np.int8)
    if int(labels.sum()) != 3:
        raise ValueError(f"expected three source reference samples for {source}")
    tensors = SampleTensors(
        exact_counts=exact,
        marginal_counts=_marginalize_ternary_counts(exact, k),
        coverage_counts=_coverage_superset_counts(coverage, k),
        sample_totals=totals,
    )
    return frame, labels, space, tensors


def fit_mlp(tensors: SampleTensors, labels: np.ndarray, train_indices: np.ndarray,
            space: PatternSpace, seed: int):
    statistics = compute_pattern_statistics(
        tensors, labels, train_indices, space, alpha=ALPHA
    )
    train_ids = np.flatnonzero(statistics["exact_total_count"] > 0)
    if len(train_ids) == 0:
        raise ValueError("no observed training patterns")
    weights = expert_fit_weight(
        statistics["effective_support"], EXPERT_FIT_SCALE
    )[train_ids]
    model, history = fit_small_mlp(
        train_ids, statistics["conditional_loglr"][train_ids], weights,
        space, epochs=EPOCHS, seed=seed,
    )
    prediction = predict_small_mlp(
        model, np.arange(space.n_patterns, dtype=np.int64), space
    )
    # No observed CpG conveys no tissue-source information.
    prediction[space.observed_counts == 0] = 0.0
    if not np.all(np.isfinite(prediction)):
        raise ValueError("nonfinite MLP prediction")
    return prediction, statistics, history, len(train_ids)


def weighted_score(counts: dict[str, int], prediction: np.ndarray,
                   space: PatternSpace) -> float:
    depth = sum(counts.values())
    if depth == 0:
        return float("nan")
    return sum(n * prediction[int(space.encode([pattern])[0])]
               for pattern, n in counts.items()) / depth


def train_reference(panel: pd.DataFrame, reference: pd.DataFrame,
                    output_dir: Path) -> None:
    required = {"sample_id", "group", "region_id", "k", "informative_reads",
                "masked_pattern_counts"}
    if not required.issubset(reference.columns):
        raise ValueError(f"reference table lacks {sorted(required - set(reference.columns))}")
    if reference.sample_id.nunique() != 205 or len(reference) != 615:
        raise ValueError("expected 205 reference samples x 3 marker regions")
    if reference.duplicated(["sample_id", "region_id"]).any():
        raise ValueError("duplicate sample-marker row")
    if panel.region_id.tolist() != ["M1", "M2", "M3"] or panel.k.tolist() != [7, 6, 7]:
        raise ValueError("unexpected manuscript panel")
    if not reference.groupby("sample_id").region_id.apply(
            lambda x: set(x) == {"M1", "M2", "M3"}).all():
        raise ValueError("reference sample is missing a marker")

    import torch
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_rows, pattern_rows, history_rows = [], [], []
    for marker in panel.itertuples(index=False):
        region, k = str(marker.region_id), int(marker.k)
        source = SOURCE_GROUPS[str(marker.source_compartment)]
        ref, labels, space, tensors = reference_tensors(
            reference.loc[reference.region_id.eq(region)], k, source
        )
        folds = stratified_sample_folds(labels, n_splits=3, seed=SEED)
        for fold in range(3):
            fit_ids = np.flatnonzero(folds != fold)
            prediction, _, history, n_seen = fit_mlp(
                tensors, labels, fit_ids, space, SEED + 100 * fold + k
            )
            for index in np.flatnonzero(folds == fold):
                row = ref.iloc[int(index)]
                counts = read_counts(row.masked_pattern_counts, k,
                                     int(row.informative_reads))
                oof_rows.append({
                    "sample_id": row.sample_id, "group": row.group,
                    "region_id": region, "source": int(labels[index]),
                    "fold": fold + 1,
                    "informative_reads": int(row.informative_reads),
                    "mlp_oof_mean_evidence_ln": weighted_score(counts, prediction, space),
                })
            history_rows.append({
                "region_id": region, "fit": f"oof_{fold + 1}",
                "n_seen_patterns": n_seen,
                "last_weighted_huber": history[-1]["train_weighted_huber"],
            })

        prediction, stats, history, n_seen = fit_mlp(
            tensors, labels, np.arange(len(ref)), space, SEED + 1000 + k
        )
        history_rows.append({
            "region_id": region, "fit": "all_reference",
            "n_seen_patterns": n_seen,
            "last_weighted_huber": history[-1]["train_weighted_huber"],
        })
        for pattern_id in range(space.n_patterns):
            if int(space.observed_counts[pattern_id]) == 0:
                continue
            pattern_rows.append({
                "region_id": region,
                "pattern": space.decode([pattern_id])[0],
                "n_observed_cpg": int(space.observed_counts[pattern_id]),
                "reference_exact_count": int(stats["exact_total_count"][pattern_id]),
                "reference_conditional_lnLR": float(stats["conditional_loglr"][pattern_id]),
                "mlp_pred_evidence_ln": float(prediction[pattern_id]),
                "balanced_compatible_support": float(stats["effective_support"][pattern_id]),
            })

    pd.DataFrame(pattern_rows).to_csv(
        output_dir / "pattern_target_and_prediction.tsv", sep="\t", index=False
    )
    pd.DataFrame(oof_rows).to_csv(
        output_dir / "reference_oof_region_scores.tsv", sep="\t", index=False
    )
    pd.DataFrame(history_rows).to_csv(
        output_dir / "training_summary.tsv", sep="\t", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--reference-patterns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, sep="\t")
    reference = pd.read_csv(args.reference_patterns, sep="\t")
    train_reference(panel, reference, args.output_dir)
    digest = hashlib.sha256(args.reference_patterns.read_bytes()).hexdigest()
    config = {
        "seed": SEED, "epochs": EPOCHS, "alpha": ALPHA,
        "expert_fit_scale": EXPERT_FIT_SCALE,
        "reference_patterns_sha256": digest,
        "model": "one-hot 0/1/2; 64-32 ReLU; AdamW; weighted Huber",
        "target": "sample-averaged, coverage-conditional natural-log source/background ratio",
        "split": "three-fold reference-sample out-of-fold; final fit uses all reference samples",
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
