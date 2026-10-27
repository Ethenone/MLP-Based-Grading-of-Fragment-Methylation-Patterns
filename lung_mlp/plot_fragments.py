"""Plot fragment-level MLP scores and exact-rule matches across cfDNA groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def draw(scores: pd.DataFrame, groups: list[tuple[str, str]],
         output: Path, max_points: int = 12000, seed: int = 20261022) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    required = {"cohort", "region_id", "count", "mlp_fragment_score_ln",
                "strict_match"}
    if not required.issubset(scores.columns):
        raise ValueError(f"fragment table lacks {sorted(required - set(scores.columns))}")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    rng = np.random.default_rng(seed)
    markers = ("M1", "M2", "M3")
    fig, axes = plt.subplots(3, 3, figsize=(10, 7), sharex="row")
    for col, (title, cohort) in enumerate(groups):
        for row, marker in enumerate(markers):
            ax = axes[row, col]
            part = scores.loc[scores.cohort.eq(cohort) & scores.region_id.eq(marker)]
            if part.empty:
                raise ValueError(f"no {cohort}/{marker} fragment rows")
            for strict, color, zorder in (
                (False, "#9bb8d5", 1), (True, "#164b9b", 2)
            ):
                sub = part.loc[part.strict_match.eq(strict)]
                if sub.empty:
                    continue
                counts = sub["count"].to_numpy(dtype=np.int64)
                total = int(counts.sum())
                n = min(total, max_points)
                # Downsampling is display-only. All reported fractions must
                # be computed from full PAT multiplicities in score.py.
                sampled = rng.multinomial(n, counts / total)
                x = np.repeat(sub.mlp_fragment_score_ln.to_numpy(float), sampled)
                y = rng.uniform(-0.35, 0.35, size=len(x))
                ax.scatter(x, y, s=2, c=color, alpha=0.48,
                           linewidths=0, rasterized=True, zorder=zorder)
            ax.axvline(0, color="#444444", linestyle="--", linewidth=0.8)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(marker)
            if row == 2:
                ax.set_xlabel("MLP fragment score")
    legend = [
        Line2D([0], [0], marker="o", linestyle="", color="#164b9b",
               label="Exact selected-site match"),
        Line2D([0], [0], marker="o", linestyle="", color="#9bb8d5",
               label="Other fragment"),
        Line2D([0], [0], linestyle="--", color="#444444",
               label="Score = 0"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-scores", type=Path, required=True)
    parser.add_argument("--wgbs-cohort", required=True)
    parser.add_argument("--tbs-cohort", required=True)
    parser.add_argument("--healthy-cohort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points-per-class", type=int, default=12000)
    args = parser.parse_args()
    detail = pd.read_csv(args.fragment_scores, sep="\t",
                         dtype={"pattern": str})
    draw(detail, [
        ("Cancer WGBS", args.wgbs_cohort),
        ("Cancer TBS", args.tbs_cohort),
        ("Healthy WGBS", args.healthy_cohort),
    ], args.output, max_points=args.max_points_per_class)


if __name__ == "__main__":
    main()
