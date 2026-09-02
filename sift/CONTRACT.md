# The SIFT contract

Every function signature, column name and constant in this document is binding.
Modules and tests cite it by section number, so the numbering of sections 1 to 7
is stable and must not be renumbered.

Code is English throughout. Docstrings follow the NumPy convention and signatures
carry type annotations. Comments explain why, never restate the line below them.

---

## 1. The four controls, and one confusion to avoid

The four players of the lattice are these four controls, not the preprocessing
steps.

```python
CONTROL_NAMES: tuple[str, ...] = ("a1_prior", "a2_labels", "b1_fs", "b2_axis")
```

| Flag | Off means | On means |
|---|---|---|
| `a1_prior` | leave the window class distribution alone | weight it to match the random reference |
| `a2_labels` | the test set keeps families never seen before | restrict to families present in the same cut training set |
| `b1_fs` | select features by mutual information over the whole panel | select on the cut training set only |
| `b2_axis` | split on `Year`, the compile timestamp | split on `first_submission_date` |

Stated explicitly because it has been misread: `drop_empty` and `min_class_20`
are **not** controls. They are fixed preprocessing, applied identically to all 16
configurations, and never appear in `ControlFlags`.

**One opt-in extension, C1.** `dedup_exact` used to sit in that same group. It
was pulled out into a fifth, protocol-side control `c1_dedup` because the effect
is measurable: depending on the cut, 0.4 to 28.7% of test-window samples have an
exact duplicate in the training window on the family panel, and turning the step
off moves the reported gap by 0.069, the same order as phi_A2 at 0.123. Sources:
`results/evidence/item5_dedup_overlap.parquet` and `item5_dedup_gap_move.parquet`.

This is opt-in and changes nothing by default. `ControlFlags`, `CONTROL_NAMES`,
`LATTICE_CELLS` and the four-flag `config_id` encoding in
`results/metrics.parquet` are unchanged, and the four-control lattice remains the
default. The five-control lattice runs only when `ExtendedControlFlags`,
`build_panel_variants`, `build_jobs_c1` and `run_lattice_c1` are called by name,
and it writes to `results/c1/` with a 5-bit `config_id`, where `config_id % 16`
recovers the old four-control index.

**The application order is mandatory.** The controls do not commute, so v(S) is
only defined once an order is fixed. That order is part of the method and is
stated in the paper.

```python
CONTROL_ORDER: tuple[str, ...] = ("b2_axis", "a2_labels", "a1_prior", "b1_fs")
```

The reasoning: `b2_axis` decides which sample lands in which set, so it goes
first; `a2_labels` then filters samples; `a1_prior` reweights without changing
which samples are present; `b1_fs` selects features on a training set that is by
then fixed.

For the five-control lattice the order is
`CONTROL_ORDER_C1 = ("c1_dedup", "b2_axis", "a2_labels", "a1_prior", "b1_fs")`.
`c1_dedup` leads because it decides **which samples exist at all**, while the
other four only rearrange or reweight a fixed set; `b2_axis` cannot split windows
before it is known whether a group of identical vectors is one sample or five.
The order is guaranteed structurally rather than by convention: C1 is applied
inside `build_panel_variants`, which sits above `apply_controls`.

**The lattice must have all 16 cells.** Silently dropping a cell is forbidden. If
a cell yields a test set below threshold, `run_lattice` raises loudly rather than
continuing, because the exact Shapley value needs all 16 values. Choose
cut-points so that the tightest cell still survives.

---

## 2. Module layout

Flat, one module per stage, no sub-packages. Imports run one way only: `config`
imports nothing from `sift`, and `experiment` may import anything.

```
sift/paths.py        directory anchors, no logic
sift/config.py       frozen dataclasses only
sift/seeding.py      deterministic seed derivation
sift/data.py         raw load through to the analysis panel
sift/features.py     feature columns, id to name mapping, selection
sift/splits.py       random and temporal splits, consistency assertions
sift/controls.py     A1 A2 B1 B2 and the application order
sift/models.py       seeded estimator factory
sift/metrics.py      metric computation
sift/cache.py        content-addressed fit cache
sift/manifest.py     run manifest and panel matching
sift/experiment.py   one lattice cell, and the whole lattice
sift/shapley.py      exact Shapley, Harsanyi dividends, Owen value
sift/drift.py        domain classifier and the residual probes
sift/iwrisk.py       importance-weighted risk
sift/driftsim.py     synthetic drift panels with a ground truth
sift/recovery.py     injection panels for the recovery check
sift/reporting.py    tables and figures
sift/mock.py         synthetic panel so tests run without MLRan
```

Installed with `pip install -e .` from `pyproject.toml`. `sys.path.append` is
forbidden anywhere in the package or the tests: `sift/paths.py` resolves every
directory relative to its own file. Only `reporting` and `recovery` write files;
no module on the computation path does.

---

## 3. Public signatures

```python
# config.py
@dataclass(frozen=True)
class ControlFlags:
    a1_prior: bool = False
    a2_labels: bool = False
    b1_fs: bool = False
    b2_axis: bool = False

    @classmethod
    def from_index(cls, index: int) -> "ControlFlags": ...   # 0..15
    def to_index(self) -> int: ...
    def active(self) -> frozenset[str]: ...

@dataclass(frozen=True)
class ExperimentConfig:
    panel: PanelSpec
    split: SplitSpec
    flags: ControlFlags
    model_name: str
    seed: int
    target: str = "ransomware_family"
    n_features: int = 200

    def fingerprint(self) -> str: ...

# seeding.py
def derive_seed(base_seed: int, *parts: str | int) -> int: ...
def make_rng(seed: int) -> numpy.random.Generator: ...

# data.py
def load_mlran(data_dir: Path = MLRAN_DIR) -> pandas.DataFrame: ...
def build_panel(raw: pandas.DataFrame, spec: PanelSpec) -> tuple[pandas.DataFrame, dict[str, int]]: ...

# splits.py
def make_split(panel, spec: SplitSpec, time_column: str, seed: int) -> tuple[Index, Index]: ...
def assert_temporal_consistency(panel, train_idx, test_idx, time_column: str) -> None: ...

# controls.py
def apply_controls(panel, cfg: ExperimentConfig) -> ControlledData: ...

# experiment.py
def run_cell(panel, cfg: ExperimentConfig) -> dict[str, object]: ...
def run_lattice(panel, base, model_names, seeds, cut_years) -> pandas.DataFrame: ...

# shapley.py
def exact_shapley(values: Mapping[frozenset[str], float]) -> dict[str, float]: ...
def harsanyi_dividends(values: Mapping[frozenset[str], float]) -> dict[frozenset[str], float]: ...
def owen_value(values, groups: Sequence[Sequence[str]]) -> dict[str, float]: ...
```

`build_panel` returns `provenance` alongside the panel, that is the row count
surviving each filtering step. It returns it rather than printing it.

---

## 4. Seeding

Every random object receives an explicitly derived seed. `np.random.seed` is
forbidden, so is `random_state=None`, so is a hand-pasted constant inside a
function.

```python
seed = derive_seed(base_seed, "model", model_name, cut_year)
```

We deliberately go against the scikit-learn recommendation to pass a
`RandomState` instance to estimators. The reason: results must coincide exactly
per (configuration, model, seed) triple, so that a difference between two of the
16 cells is attributable to a control rather than to chance. The compensation is
averaging over an explicit seed list. The paper states this.

**The temporal design also runs multiple seeds.** The temporal split is
deterministic, and measurement shows only **two** of the four models are seed
sensitive: RandomForest and the MLP. Logistic Regression uses lbfgs and is
deterministic; LightGBM at defaults has `bagging_fraction` and `feature_fraction`
of 1.0 and is deterministic too. So for those two models the seed variance is
exactly zero, and their intervals must come from a bootstrap over the prediction
table rather than from the seed list. Both designs still run over the same list
of 5 seeds. Otherwise Delta would be the difference between a five-run mean and a
single point, with the temporal side's variance undefined.

---

## 5. Metric

```python
labels = numpy.unique(y_true)          # pinned explicitly, never left as None
f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0.0)
```

Pinning `labels` and setting `zero_division` explicitly are both mandatory. Left
at their defaults, sklearn averages over the union of the labels present in
`y_true` and `y_pred`; the temporal window holds markedly fewer families than the
random test set, so the two designs would average over different class counts and
would not be comparable.

Record `k_test`, the number of classes averaged over, on every result row.

Report `balanced_accuracy` and `mcc` alongside so a reader can see the conclusion
does not hinge on one metric choice. During piloting there was a case with
macro-F1 at 0.244 against balanced accuracy at 0.471, a factor of two, which is
why both are shown.

---

## 6. Result schema

**`results/predictions.parquet`**, one row per test sample per fit.

| Column | Type | Meaning |
|---|---|---|
| `fit_id` | string | join key to the metrics table |
| `sample_id` | int32 | MLRan sample identifier |
| `grp_id` | int32 | duplicate group, needed when comparing across configurations |
| `y_true` | int16 | true family code |
| `y_pred` | int16 | predicted family code |
| `test_year` | int16 | year of the sample |

**`results/metrics.parquet`**, one row per fit: `fit_id`, `run_id`, `config_id`
from 0 to 15, the four control flags as booleans, `design`, `cut`, `seed`,
`model`, `n_train`, `n_test`, `k_train`, `k_test`, the four metrics, and
`fit_seconds`.

The four metric columns are a convenience and a consistency check. **Every number
that reaches the paper, and every confidence interval, is recomputed from
`predictions.parquet`.**

zstd level 3.

---

## 7. Cache and parallelism

**`joblib.Memory` is not used.** It identifies a cache entry by function name plus
a pickle of the arguments, and the panel is a DataFrame whose pickle is not
stable. It also detects changes only in the decorated function body, not in the
helpers it calls. We use a content-addressed cache of our own instead.

The cache key is a blake2b over canonicalised JSON containing: the schema
version, the task name, the MLRan commit and the sha256 of the three data files,
the four control flags, the design, the cut, the seed, the model name, the model
parameters, **a hash of all of `sift/*.py`**, and the library versions. `n_jobs`
and `verbose` are excluded, having been checked not to change predictions.

The cache sits at fit level, not at data-preparation level. Measured: 0.5 s to
load and 0.3 s to preprocess, so rebuilding for all 16 configurations costs about
64 s. Caching there buys nothing. Caching at fit level buys the ability to resume
after a failure part-way through.

**Parallelism parameters, taken from measurement rather than from the core
count:**

```python
Parallel(n_jobs=6, backend="loky", batch_size=1, pre_dispatch="2*n_jobs")
RandomForestClassifier(..., n_jobs=1)
LGBMClassifier(..., n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
```

The reference machine has 4 physical cores and 8 logical threads. The measured
speedup ceiling is 1.99x at `n_jobs=6`, worse at both 4 and 8. The bottleneck is
memory bandwidth and L3 contention rather than thread count: setting
`OMP_NUM_THREADS=1` changes nothing. Every estimator is given `n_jobs=1`.

Worth noting: LightGBM runs faster single-threaded, 18.0 s at `n_jobs=1` against
31.5 s at `n_jobs=8`, with identical predictions at every setting.

Queue longest-first, LightGBM before the others and larger cuts first, since fit
time ranges from 0.8 to 23 s.

---

## 8. Measured budget

| Model | Time per fit |
|---|---|
| LogReg | 3.8 s |
| RandomForest | 3.6 s |
| LightGBM | 18.0 s |
| MLP | 0.8 s |

LightGBM accounts for 69 percent of the total cost.

The value function is a difference: `Delta(S) = M_reference(S) - M_temporal(S)`
and `v(S) = Delta(empty) - Delta(S)`, with `random_fully_matched` as the
reference arm. Both arms move with the coalition, so the game needs all 16
coalitions on **both**. The `random` and `random_matched` designs exist only to
report the three-way split of the gap; they are not moves in the game and run at
the empty coalition only.

Each (cut, model, seed) triple therefore needs 34 fits rather than 64:

```
(16 + 16 + 1 + 1) x 5 cuts x 4 models x 5 seeds = 3,400 fits
```

Measured wall-clock for the full lattice is 1.60 hours; the five-control lattice
at 6,800 fits takes 3.09 hours. Total result size is about 8 MB excluding the
refittable cache.
