"""Checks for the statistical meaning of masked CpG patterns."""

import json
import unittest

import numpy as np
import pandas as pd

from lung_mlp.patterns import (
    PatternSpace, SampleTensors, _coverage_superset_counts,
    _marginalize_ternary_counts, compute_pattern_statistics,
)
from lung_mlp.score import compare
from lung_mlp.sensitivity import analyze
from lung_mlp.train import read_counts
from lung_mlp.extract_pat import extract_region


class PatternSemanticsTests(unittest.TestCase):
    def test_pat_extraction_keeps_missing_distinct_from_methylation(self):
        class FakeTabix:
            def fetch(self, *_):
                return iter([
                    "chr2\t9\tCTC\t2",   # overlaps indices 10 and 11
                    "chr2\t10\tT.\t3",   # unknown call remains missing
                ])

        region = {"chrom": "chr2", "start_cpg": 10, "end_cpg": 12, "k": 2}
        counts = extract_region(FakeTabix(), region)
        self.assertEqual(counts["01"], 2)
        self.assertNotIn("0?", counts)  # only one observed CpG, filtered

    def test_missing_state_marginalizes_without_double_counting(self):
        space = PatternSpace(2)
        exact = np.zeros((1, space.n_patterns), dtype=np.int64)
        coverage = np.zeros((1, space.n_masks), dtype=np.int64)
        for pattern, count in {"00": 2, "10": 3, "20": 5}.items():
            pid = int(space.encode([pattern])[0])
            exact[0, pid] = count
            coverage[0, space.mask_ids[pid]] += count
        marginal = _marginalize_ternary_counts(exact, 2)
        eligible = _coverage_superset_counts(coverage, 2)
        pid = int(space.encode(["20"])[0])
        self.assertEqual(int(marginal[0, pid]), 10)
        self.assertEqual(int(eligible[0, space.mask_ids[pid]]), 10)
        # Marginal views overlap across different queries, but each
        # observation appears once within the single query "20".

    def test_conditional_target_is_source_over_background(self):
        space = PatternSpace(1)
        exact = np.zeros((2, 3), dtype=np.int64)
        exact[0, space.encode(["0"])[0]] = 9  # background
        exact[0, space.encode(["1"])[0]] = 1
        exact[1, space.encode(["0"])[0]] = 1  # target source
        exact[1, space.encode(["1"])[0]] = 9
        coverage = np.zeros((2, 2), dtype=np.int64)
        coverage[:, 1] = 10
        tensors = SampleTensors(
            exact_counts=exact,
            marginal_counts=_marginalize_ternary_counts(exact, 1),
            coverage_counts=_coverage_superset_counts(coverage, 1),
            sample_totals=np.array([10, 10]),
        )
        stats = compute_pattern_statistics(
            tensors, np.array([0, 1]), np.array([0, 1]), space, alpha=1e-4
        )
        self.assertGreater(stats["conditional_loglr"][space.encode(["1"])[0]], 0)
        self.assertLess(stats["conditional_loglr"][space.encode(["0"])[0]], 0)

    def test_strict_match_and_mlp_use_different_rules(self):
        panel = pd.DataFrame([
            {"region_id": marker, "k": 2, "selected_k": 2,
             "selected_indices0": "0,1", "direction": "low"}
            for marker in ("M1", "M2", "M3")
        ])
        patterns = pd.DataFrame([{
            "sample_id": "example", "cohort": "test", "region_id": "M1",
            "k": 2, "informative_reads": 10,
            "masked_pattern_counts": json.dumps({"00": 2, "02": 3, "10": 5}),
        }])
        lookup = {
            ("M1", "00"): 0.5, ("M1", "02"): 0.2, ("M1", "10"): -0.5,
        }
        detail, summary = compare(patterns, panel, lookup)
        row = summary.iloc[0]
        self.assertEqual(row.informative_reads, 10)
        self.assertEqual(row.selected_site_covered_reads, 7)
        self.assertEqual(row.strict_match_reads, 2)
        self.assertEqual(row.mlp_positive_all_reads, 5)
        self.assertAlmostEqual(row.strict_match_fraction_covered, 2 / 7)
        self.assertAlmostEqual(row.mlp_positive_fraction_all, 0.5)
        self.assertTrue(detail.loc[detail.pattern.eq("02"), "mlp_positive"].iloc[0])

    def test_sensitivity_changes_only_observed_start_state(self):
        panel = pd.DataFrame([
            {"region_id": "M1", "k": 2, "selected_indices0": "0,1"},
        ])
        detail = pd.DataFrame([
            {"sample_id": "a", "region_id": "M1", "pattern": "10",
             "count": 2, "mlp_fragment_score_ln": -1.0},
        ])
        lookup = {
            ("M1", "10"): -1.0, ("M1", "20"): 0.0,
            ("M1", "00"): 1.0, ("M1", "12"): -1.5,
        }
        _, pooled = analyze(detail, panel, lookup)
        first = pooled.loc[pooled.cpg_index0.eq(0)].set_index("change")
        self.assertAlmostEqual(first.loc["1→2", "mean_signed_delta_ln"], 1.0)
        self.assertAlmostEqual(first.loc["1→0", "mean_signed_delta_ln"], 2.0)
        self.assertNotIn("0→2", first.index)
        second = pooled.loc[pooled.cpg_index0.eq(1)].set_index("change")
        self.assertAlmostEqual(second.loc["0→2", "mean_signed_delta_ln"], -0.5)

    def test_read_count_validation(self):
        with self.assertRaises(ValueError):
            read_counts('{"0?": 3}', 2, 4)


if __name__ == "__main__":
    unittest.main()
