# MLP-Based Grading of Fragment Methylation Patterns

Code accompanying the lung epithelial cfDNA manuscript. This repository contains
the **MLP-only** analysis of three fixed methylation marker intervals (M1–M3).
Marker discovery was performed with wgbstools and is described in the manuscript.
The repository does not contain the broader SABLE mixture-of-experts work or the
excluded cancer-tissue exploratory analysis.

## What the model estimates

One MLP is fitted for each marker using fragment patterns from 205 **healthy
tissue** reference samples. M1/M2 compare alveolar epithelial cells with all
other healthy tissue samples; M3 compares bronchial epithelial cells with all
other healthy tissue samples. The target is the sample-averaged,
coverage-conditional natural-log ratio of target-cell to background-cell
pattern frequencies, with a pseudocount of `1e-4`. Pattern support controls
the training weight. Input states are:

- `0`: observed unmethylated CpG;
- `1`: observed methylated CpG;
- `2`: CpG not covered by the fragment.

State `2` is an observation mask, not a third biological methylation state.
For each marker, a one-hot-encoded pattern enters a 64–32 ReLU MLP. The
network is trained with weighted Huber loss for 30 epochs. Three-fold
out-of-fold checks split by **reference sample**, and the frozen model is
then refitted on all reference samples. Cancer and healthy cfDNA do not enter
model training. The output is tissue-source evidence, **not** a cancer
probability or a calibrated clinical test result.

The analysis compares `score > 0` fragment fractions with the manuscript's
exact selected-site rule: all selected CpGs must be observed and all `0`
for M1/M2 or all `1` for M3. PAT multiplicity is retained in every
fragment fraction. For targeted bisulfite sequencing, amplified reads are
not assumed to be independent cfDNA molecules.

## Contents

| Path | Purpose |
| --- | --- |
| `data/panel_hg38.tsv` | Frozen M1–M3 hg38 intervals and selected CpGs |
| `data/healthy_reference_patterns.tsv.gz` | Compact pattern counts from the 205 public healthy-tissue samples |
| `data/frozen_pattern_scores.tsv` | Frozen MLP score for every informative ternary pattern |
| `lung_mlp/extract_pat.py` | Indexed PAT to fixed-marker masked pattern counts |
| `lung_mlp/patterns.py` | Coverage-conditional, sample-averaged statistical target |
| `lung_mlp/model.py` | Exact MLP architecture and training loop |
| `lung_mlp/train.py` | Reference-only three-fold check and final fit |
| `lung_mlp/score.py` | Fragment scores, positive fractions, exact-rule comparison |
| `lung_mlp/sensitivity.py` | Site-wise 0→2, 1→2 and 1→0 score changes |
| `lung_mlp/plot_fragments.py` | Fragment-score distributions with exact-rule points marked |
| `lung_mlp/evaluate.py` | Exploratory WGBS 3-case versus 23-healthy ROC |

The included reference pattern table allows the training to be replayed
without downloading the full public WGBS dataset. The frozen prediction
table allows exact re-scoring without rerunning
PyTorch. It includes all 5,100 patterns with at least one observed CpG
across the 7/6/7-site markers. The all-missing pattern has score zero.

## Installation

Use Python 3.10 or later. Install the packages in `requirements.txt` in a
clean environment:

```bash
python -m pip install -r requirements.txt
```

Run from the repository root with `python -m lung_mlp.<module>`.
PyTorch versions and CPU/GPU backends can cause small floating-point
differences after refitting; `data/frozen_pattern_scores.tsv` preserves the
scores used for the manuscript analysis.

## Input files

PAT files are bgzip-compressed, CSI-indexed, four-column files containing
chromosome, **1-based hg38 CpG index**, C/T/. pattern and multiplicity.
The CpG index must use the same hg38 wgbstools reference as the panel.
FASTQ alignment, methylation calling and `bam2pat` are upstream standard
sequencing steps and are not included here.

For PAT extraction, prepare a CSV with these columns:

```text
sample_id,group,cohort,pat_path
reference_001,Lung-Alveolar-Epithelial,reference,/path/to/reference_001.pat.gz
case_001,case,WGBS_case,/path/to/case_001.pat.gz
```

Use one row per biological sample or sequencing run. Each `pat_path` needs a
matching `.csi` index. For M1/M2, the reference table must contain three
`Lung-Alveolar-Epithelial` samples; for M3, three
`Lung-Bronchus-Epithelial` samples. Other healthy tissues are the
background. Do not combine replicate reads as independent donors.
For a downloaded PAT without a reliable index, rebuild its CSI on a **copy**
before extraction:

```python
import pysam
pysam.tabix_index("sample.pat.gz", seq_col=0, start_col=1, end_col=1,
                  csi=True, force=True)
```

### Train from the included public-reference subset

```bash
python -m lung_mlp.train \
  --panel data/panel_hg38.tsv \
  --reference-patterns data/healthy_reference_patterns.tsv.gz \
  --output-dir work/train
```

On the local verification environment, the reference counts and targets
replayed exactly (target difference below `3e-16`), and all 5,100 MLP
scores differed from the frozen table by at most `4.9e-6` due to the
PyTorch runtime.

### Re-extract patterns from PAT files

```bash
python -m lung_mlp.extract_pat \
  --panel data/panel_hg38.tsv \
  --manifest reference_manifest.csv \
  --output work/reference_patterns.tsv.gz
```

The training output includes `pattern_target_and_prediction.tsv`, out-of-fold
reference scores, training summaries and a configuration file. The code
expects exactly 205 distinct reference samples, each scored at all three
markers.

### Score cfDNA and compare the rules

```bash
python -m lung_mlp.extract_pat \
  --panel data/panel_hg38.tsv \
  --manifest cfdna_manifest.csv \
  --output work/cfdna_patterns.tsv.gz

python -m lung_mlp.score \
  --panel data/panel_hg38.tsv \
  --patterns work/cfdna_patterns.tsv.gz \
  --predictions data/frozen_pattern_scores.tsv \
  --output-dir work/cfdna
```

`fragment_pattern_scores.tsv` contains one row per distinct sample-pattern
with its multiplicity and model score. `sample_marker_fractions.tsv` contains
the full selected-site denominator, exact-rule numerator, MLP-positive
numerators and their fractions. `mlp_positive_fraction_all` is the fraction
of informative reads with fragment score strictly greater than zero.

### Site sensitivity and exploratory ROC

```bash
python -m lung_mlp.sensitivity \
  --fragment-scores work/cfdna/fragment_pattern_scores.tsv \
  --panel data/panel_hg38.tsv \
  --predictions data/frozen_pattern_scores.tsv \
  --output-dir work/sensitivity --plot

python -m lung_mlp.evaluate \
  --fractions work/wgbs_only/sample_marker_fractions.tsv \
  --labels wgbs_case_healthy_labels.csv \
  --output-dir work/roc --plot
```

The sensitivity analysis changes one *observed* site at a time while holding
the rest of that fragment fixed; within-sample changes are weighted by PAT
multiplicity, and sample means are then averaged equally. The ROC input must
contain only WGBS cancer cases and WGBS healthy cfDNA controls. The label
file has `sample_id,is_case` with `1` for a case and `0` for healthy. Do
not mix targeted cancer sequencing with WGBS healthy controls when estimating
diagnostic performance.

For a descriptive three-column fragment-score plot, run
`python -m lung_mlp.plot_fragments --help` and pass the cohort labels from
your cfDNA manifest. Display points are sampled with a fixed seed; numerical
fractions always use the full counts in `sample_marker_fractions.tsv`.

## Data and interpretation

The healthy tissue reference was derived from public WGBS data in
[GSE186458](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186458).
The paper-specific clinical sequencing GEO accession can be added after it
is assigned. This repository contains no identifiable patient data, local
server paths, credentials or raw FASTQ files.

The exploratory 3-versus-23 WGBS ROC is a ranking result on a small cohort.
It does not establish a clinical threshold or sensitivity/specificity.
Targeted samples have no assay-matched healthy targeted controls in this
study.
