"""Exploratory ROC analysis of WGBS positive-fragment fractions by marker."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def evaluate(fractions: pd.DataFrame, labels: pd.DataFrame):
    required = {"sample_id", "region_id", "mlp_positive_fraction_all"}
    if not required.issubset(fractions.columns):
        raise ValueError(f"fraction table lacks {sorted(required - set(fractions.columns))}")
    if not {"sample_id", "is_case"}.issubset(labels.columns):
        raise ValueError("labels require sample_id and is_case")
    if labels.sample_id.duplicated().any():
        raise ValueError("duplicate sample label")
    if fractions.duplicated(["sample_id", "region_id"]).any():
        raise ValueError("duplicate sample-marker fraction")
    joined = fractions.merge(labels[["sample_id", "is_case"]],
                             on="sample_id", validate="many_to_one")
    if set(joined.sample_id) != set(labels.sample_id):
        raise ValueError("fraction table does not cover every labeled sample")
    if set(joined.region_id) != {"M1", "M2", "M3"}:
        raise ValueError("expected M1, M2, M3")
    metrics, coordinates = [], []
    for marker, part in joined.groupby("region_id", sort=True):
        y = part.is_case.to_numpy(dtype=int)
        score = part.mlp_positive_fraction_all.to_numpy(dtype=float)
        if set(y) != {0, 1} or not np.isfinite(score).all():
            raise ValueError(f"invalid ROC inputs for {marker}")
        fpr, tpr, thresholds = roc_curve(y, score)
        metrics.append({
            "region_id": marker, "n_case": int(y.sum()),
            "n_healthy": int((1 - y).sum()),
            "roc_auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
        })
        coordinates.extend({
            "region_id": marker, "fpr": float(fp),
            "tpr": float(tp), "threshold": float(threshold),
        } for fp, tp, threshold in zip(fpr, tpr, thresholds))
    return pd.DataFrame(metrics), pd.DataFrame(coordinates)


def plot(coordinates: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for marker, part in coordinates.groupby("region_id", sort=True):
        auc = metrics.set_index("region_id").loc[marker, "roc_auc"]
        ax.plot(part.fpr, part.tpr, marker="o", markersize=3,
                label=f"{marker} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="CSV or TSV: sample_id,is_case; WGBS only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    sep = "\t" if args.labels.suffix.lower() == ".tsv" else ","
    metrics, coordinates = evaluate(
        pd.read_csv(args.fractions, sep="\t"),
        pd.read_csv(args.labels, sep=sep),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "roc_metrics.tsv", sep="\t", index=False)
    coordinates.to_csv(args.output_dir / "roc_coordinates.tsv",
                       sep="\t", index=False)
    if args.plot:
        plot(coordinates, metrics, args.output_dir / "roc.pdf")


if __name__ == "__main__":
    main()
