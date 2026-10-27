"""Convert indexed hg38 PAT records into fixed-marker masked pattern counts.

This is the feature conversion used by the MLP. FASTQ alignment and PAT
generation are performed separately with standard sequencing tools.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


MAX_PAT_LEN = 150
MIN_OBSERVED_CPGS = 2
MIN_OBSERVED_FRACTION = 0.4


def load_regions(panel: pd.DataFrame) -> list[dict]:
    needed = {"region_id", "chrom", "start_cpg", "end_cpg", "k"}
    if not needed.issubset(panel.columns):
        raise ValueError(f"panel lacks {sorted(needed - set(panel.columns))}")
    if panel.region_id.duplicated().any():
        raise ValueError("duplicate marker")
    regions = panel[list(needed)].to_dict("records")
    for region in regions:
        region["start_cpg"] = int(region["start_cpg"])
        region["end_cpg"] = int(region["end_cpg"])
        region["k"] = int(region["k"])
        if region["end_cpg"] - region["start_cpg"] != region["k"]:
            raise ValueError(f"inconsistent half-open CpG interval: {region}")
    return regions


def filtered_counter(counter: Counter, k: int) -> Counter:
    return Counter({
        pattern: n for pattern, n in counter.items()
        if sum(char != "?" for char in pattern) >= MIN_OBSERVED_CPGS
        and sum(char != "?" for char in pattern) / k >= MIN_OBSERVED_FRACTION
    })


def extract_region(tabix, region: dict) -> Counter:
    chrom = region["chrom"]
    start = region["start_cpg"]
    end = region["end_cpg"]
    k = region["k"]
    counter = Counter()
    # CSI indexes the 1-based PAT start coordinate. Broad fetch followed by
    # exact half-open overlap recovers fragments that start before the marker.
    fetch_start = max(0, start - MAX_PAT_LEN - 1)
    fetch_end = end + 1
    try:
        lines = tabix.fetch(chrom, fetch_start, fetch_end)
    except ValueError:
        lines = ()
    for line in lines:
        fields = line.rstrip().split("\t")
        if len(fields) < 4:
            continue
        read_start = int(fields[1])
        raw = fields[2]
        multiplicity = int(fields[3])
        if multiplicity <= 0:
            raise ValueError("nonpositive PAT multiplicity")
        read_end = read_start + len(raw)
        ov0 = max(read_start, start)
        ov1 = min(read_end, end)
        if ov0 >= ov1:
            continue
        chars = ["?"] * k
        for ci in range(ov0, ov1):
            char = raw[ci - read_start]
            if char == "C":
                chars[ci - start] = "1"
            elif char == "T":
                chars[ci - start] = "0"
        counter["".join(chars)] += multiplicity
    return filtered_counter(counter, k)


def extract_sample(sample, regions: list[dict]) -> list[dict]:
    try:
        import pysam
    except ImportError as exc:
        raise RuntimeError("pysam is required to read indexed PAT files") from exc
    pat = Path(sample.pat_path)
    if not pat.is_file() or not Path(str(pat) + ".csi").is_file():
        raise FileNotFoundError(f"PAT or CSI missing for {sample.sample_id}: {pat}")
    rows = []
    with pysam.TabixFile(str(pat), index=str(pat) + ".csi") as tabix:
        for region in regions:
            counter = extract_region(tabix, region)
            k = region["k"]
            depth = sum(counter.values())
            observed = sum(n * sum(c != "?" for c in p)
                           for p, n in counter.items())
            rows.append({
                "sample_id": sample.sample_id,
                "group": sample.group,
                "cohort": sample.cohort,
                "region_id": region["region_id"],
                "k": k,
                "informative_reads": depth,
                "mean_observed_fraction": observed / (depth * k) if depth else float("nan"),
                "masked_pattern_counts": json.dumps(
                    dict(sorted(counter.items())), separators=(",", ":"
                    )
                ),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV: sample_id,group,cohort,pat_path")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output TSV or TSV.GZ")
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, sep="\t")
    regions = load_regions(panel)
    manifest = pd.read_csv(args.manifest, dtype=str)
    required = {"sample_id", "group", "cohort", "pat_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifest lacks {sorted(required - set(manifest.columns))}")
    if manifest.sample_id.duplicated().any():
        raise ValueError("duplicate sample ID")
    base = args.manifest.resolve().parent
    manifest["pat_path"] = manifest.pat_path.map(
        lambda p: str((base / p).resolve()) if not Path(p).is_absolute() else p
    )
    rows = [row for sample in manifest.itertuples(index=False)
            for row in extract_sample(sample, regions)]
    result = pd.DataFrame(rows).sort_values(["sample_id", "region_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
