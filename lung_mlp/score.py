"""Score fragment patterns and compare the MLP with exact selected-site matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .train import read_counts


def load_score_lookup(path: Path) -> dict[tuple[str, str], float]:
    table = pd.read_csv(path, sep="\t", dtype={"pattern": str})
    required = {"region_id", "pattern", "mlp_pred_evidence_ln"}
    if not required.issubset(table.columns):
        raise ValueError(f"prediction table lacks {sorted(required - set(table.columns))}")
    if table.duplicated(["region_id", "pattern"]).any():
        raise ValueError("duplicate marker-pattern prediction")
    return {
        (r.region_id, r.pattern): float(r.mlp_pred_evidence_ln)
        for r in table.itertuples(index=False)
    }


def compare(patterns: pd.DataFrame, panel: pd.DataFrame,
            lookup: dict[tuple[str, str], float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if patterns.duplicated(["sample_id", "region_id"]).any():
        raise ValueError("duplicate sample-marker row")
    panel_map = {r.region_id: r for r in panel.itertuples(index=False)}
    if set(panel_map) != {"M1", "M2", "M3"}:
        raise ValueError("expected M1, M2, M3")
    detail, summary = [], []
    for row in patterns.itertuples(index=False):
        marker = panel_map[row.region_id]
        k = int(marker.k)
        selected = [int(x) for x in str(marker.selected_indices0).split(",")]
        if len(selected) != int(marker.selected_k):
            raise ValueError("selected CpG count mismatch")
        target = "0" if marker.direction == "low" else "1"
        counts = read_counts(row.masked_pattern_counts, k, int(row.informative_reads))
        all_depth = selected_depth = match_count = positive_all = positive_selected = 0
        for pattern, count in counts.items():
            # All-missing patterns are excluded by extraction; if supplied,
            # their source-evidence score is defined as zero.
            score = (0.0 if set(pattern) == {"2"} else
                     lookup[(row.region_id, pattern)])
            covered = all(pattern[i] != "2" for i in selected)
            matched = covered and all(pattern[i] == target for i in selected)
            positive = score > 0.0
            all_depth += count
            selected_depth += count if covered else 0
            match_count += count if matched else 0
            positive_all += count if positive else 0
            positive_selected += count if (covered and positive) else 0
            detail.append({
                "sample_id": row.sample_id, "cohort": row.cohort,
                "region_id": row.region_id, "pattern": pattern,
                "count": count, "mlp_fragment_score_ln": score,
                "selected_site_covered": covered, "strict_match": matched,
                "mlp_positive": positive,
            })
        if all_depth != int(row.informative_reads):
            raise ValueError("read-count mismatch")
        summary.append({
            "sample_id": row.sample_id, "cohort": row.cohort,
            "region_id": row.region_id,
            "informative_reads": all_depth,
            "selected_site_covered_reads": selected_depth,
            "strict_match_reads": match_count,
            "mlp_positive_all_reads": positive_all,
            "mlp_positive_covered_reads": positive_selected,
            "strict_match_fraction_covered": (
                match_count / selected_depth if selected_depth else np.nan
            ),
            "mlp_positive_fraction_covered": (
                positive_selected / selected_depth if selected_depth else np.nan
            ),
            "mlp_positive_fraction_all": (
                positive_all / all_depth if all_depth else np.nan
            ),
        })
    return pd.DataFrame(detail), pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    detail, summary = compare(
        pd.read_csv(args.patterns, sep="\t"),
        pd.read_csv(args.panel, sep="\t"),
        load_score_lookup(args.predictions),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "fragment_pattern_scores.tsv",
                  sep="\t", index=False)
    summary.to_csv(args.output_dir / "sample_marker_fractions.tsv",
                   sep="\t", index=False)


if __name__ == "__main__":
    main()
