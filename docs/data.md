# Data

[Back to README](../README.md)

## No malware is redistributed

This repository contains no malware in any form: no binaries, no hashes that
could be used to look binaries up, and no raw sandbox traces. The only input is
the extracted feature representation the MLRan authors published, specifically
their 483-feature RFE set of binary indicators together with the accompanying
metadata file.

Anyone reproducing this work must obtain the dataset from the MLRan authors
under their terms. We do not redistribute it.

## Obtaining MLRan

Three files are read, pinned in `sift/paths.py`:

```
mlran/6_experiments/FS_MLRan_Datasets/MLRan_X_train_RFE.csv
mlran/6_experiments/FS_MLRan_Datasets/MLRan_X_test_RFE.csv
mlran/6_experiments/FS_MLRan_Datasets/mlran_dataset_metadata.csv
```

Clone the MLRan repository to `mlran/` at the root of this repository. That
directory is ignored by git. `sift/paths.py` locates it relative to the package
file, so no module manipulates `sys.path` and nothing depends on the working
directory.

Cite the dataset:

> F. C. Onwuegbuche, S. O. Adelodun, A. D. Jurcut, and L. Pasquale, "MLRan: A
> behavioural dataset for ransomware analysis and detection," *Journal of Network
> and Computer Applications*, vol. 250, art. no. 104475, 2026,
> doi:10.1016/j.jnca.2026.104475.

## Preprocessing

One fixed chain, applied identically to all 16 lattice configurations, in
`sift.data.build_panel`:

| Step | Remaining |
|---|---|
| Features joined to metadata | 4,880 |
| Drop samples activating no feature | 4,742 |
| Collapse identical feature vectors, keep the earliest submission | 3,605 |
| Filter to the 2012 to 2024 frame | 3,311 |
| Keep ransomware from families of at least 20 samples | 1,425, 34 classes |

**None of these steps is a control.** They run before the lattice and identically
within it. The deduplication step in particular is not control B1 and not the
supplementary fifth control; those are separate interventions. `sift/CONTRACT.md`
section 1 spells out the distinction, because conflating them is the most likely
way to misread the decomposition.

`scripts/analysis/dataset_summary.py` prints this chain and writes it to
`results/evidence/dataset_summary.parquet`, so the numbers above are checkable
rather than assertions in prose.

## Two time axes

Each sample carries two dates: the first submission date and the compile
timestamp. They disagree often enough to matter, and the disagreement is the
raw material of control B2. `results/evidence/dataset_summary.parquet` section
`axis_disagreement` quantifies it.

The frame ends in 2024, which is covered only through June. That truncates the
cut 2023 test window to 31 samples, which is why every table marks the 2023 row
descriptive only.

## An inherited limitation we cannot remove

The 483 features were selected by the dataset authors using RFE on their own
split. That selection therefore carries look-ahead we have no way to undo.

The consequence is specific and it constrains one result: control B1 measures
only the leakage we deliberately introduce on top, not the leakage already baked
into the feature set. A reader should not take φ_B1 as the total cost of feature
selection leakage on this dataset.
