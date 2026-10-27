"""Compute 0→2, 1→2, and 1→0 site sensitivity of frozen MLP scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .score import load_score_lookup


TRANSITIONS = (("0→2", "0", "2"),
               ("1→2", "1", "2"),
               ("1→0", "1", "0"))


def analyze(detail: pd.DataFrame, panel: pd.DataFrame,
            lookup: dict[tuple[str, str], float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_map = {r.region_id: r for r in panel.itertuples(index=False)}
    # Each fragment's PAT multiplicity weights that sample's estimate.
    # Samples, including low-depth WGBS samples, then receive equal weight.
    accumulated = defaultdict(lambda: [0, 0.0, 0.0])
    for row in detail.itertuples(index=False):
        marker, pattern, count = row.region_id, str(row.pattern), int(row.count)
        if marker not in panel_map or len(pattern) != int(panel_map[marker].k):
            raise ValueError("marker or pattern length mismatch")
        expected = lookup[(marker, pattern)]
        if abs(expected - float(row.mlp_fragment_score_ln)) > 1e-7:
            raise ValueError("fragment scores differ from frozen predictions")
        for site, state in enumerate(pattern):
            for name, before, after in TRANSITIONS:
                if state != before:
                    continue
                edited = pattern[:site] + after + pattern[site + 1:]
                score_after = (0.0 if set(edited) == {"2"}
                               else lookup[(marker, edited)])
                change = score_after - expected
                slot = accumulated[(row.sample_id, marker, site, name)]
                slot[0] += count
                slot[1] += count * change
                slot[2] += count * abs(change)

    per_sample = pd.DataFrame([
        {
            "sample_id": sample, "region_id": marker, "cpg_index0": site,
            "change": change, "reads": reads,
            "mean_signed_delta_ln": total / reads,
            "mean_abs_delta_ln": absolute / reads,
        }
        for (sample, marker, site, change), (reads, total, absolute)
        in accumulated.items()
    ])
    if per_sample.empty:
        raise ValueError("no observed starting states")
    pooled = per_sample.groupby(
        ["region_id", "cpg_index0", "change"], as_index=False
    ).agg(
        n_samples=("sample_id", "nunique"),
        total_reads=("reads", "sum"),
        mean_signed_delta_ln=("mean_signed_delta_ln", "mean"),
        mean_abs_delta_ln=("mean_abs_delta_ln", "mean"),
    )
    flags = []
    for marker in panel.itertuples(index=False):
        selected = {int(x) for x in str(marker.selected_indices0).split(",")}
        for site in range(int(marker.k)):
            flags.append({
                "region_id": marker.region_id,
                "cpg_index0": site,
                "manuscript_selected": site in selected,
            })
    pooled = pooled.merge(pd.DataFrame(flags),
                          on=["region_id", "cpg_index0"], validate="many_to_one")
    return per_sample, pooled


def plot(pooled: pd.DataFrame, panel: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"0→2": "#2166AC", "1→2": "#B2182B", "1→0": "#1B7837"}
    markers = {"0→2": "o", "1→2": "s", "1→0": "^"}
    for marker in panel.itertuples(index=False):
        part = pooled.loc[pooled.region_id.eq(marker.region_id)]
        selected = {int(x) for x in str(marker.selected_indices0).split(",")}
        fig, ax = plt.subplots(figsize=(4.2, 3.1))
        for name, _, _ in TRANSITIONS:
            series = part.loc[part.change.eq(name)].sort_values("cpg_index0")
            ax.plot(series.cpg_index0 + 1, series.mean_signed_delta_ln,
                    color=colors[name], marker=markers[name], label=name)
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(1, int(marker.k) + 1))
        ax.set_xticklabels([f"{i + 1}{'*' if i in selected else ''}"
                            for i in range(int(marker.k))])
        ax.set_xlabel("CpG position within marker")
        ax.set_ylabel("Mean score change (after − before)")
        ax.set_title(marker.region_id)
        ax.legend(frameon=False, ncol=3, fontsize=8)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(output_dir / f"{marker.region_id}_sensitivity.{extension}",
                        dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-scores", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, sep="\t")
    per_sample, pooled = analyze(
        pd.read_csv(args.fragment_scores, sep="\t", dtype={"pattern": str}),
        panel,
        load_score_lookup(args.predictions),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(args.output_dir / "per_sample_sensitivity.tsv",
                      sep="\t", index=False)
    pooled.to_csv(args.output_dir / "pooled_sensitivity.tsv",
                  sep="\t", index=False)
    if args.plot:
        plot(pooled, panel, args.output_dir)


if __name__ == "__main__":
    main()
