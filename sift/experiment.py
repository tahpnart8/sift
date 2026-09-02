"""Execution layer: one lattice cell, and the full 16-coalition lattice.

Two invariants drive every design decision in this module.
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed

from sift import cache, manifest
from sift import metrics as metrics_module
from sift import models as models_module
from sift.config import (
    CONTROL_NAMES_C1,
    ControlFlags,
    ExperimentConfig,
    ExtendedControlFlags,
    SplitSpec,
    config_for_cell_c1,
    extended_flags_of,
)
from sift.controls import CONTROL_ORDER_C1, apply_controls
from sift.data import PanelVariants
from sift.seeding import derive_seed

try:  # CONTROL_NAMES is fixed by the contract; its home module is not.
    from sift.config import CONTROL_NAMES
except ImportError:  # pragma: no cover
    from sift.controls import CONTROL_NAMES  # type: ignore[no-redef]


# --- Constants fixed by the contract ----------------------------------------
N_COALITIONS: int = 16

# What is declared here is the *role* each design plays, a property of the
# value function rather than of the split machinery.
#
#     Delta(S) = M_reference(S) - M_temporal(S)
#     v(S)     = Delta(empty) - Delta(S)
#     R        = Delta(N)
#
# so v(empty) = 0 by construction and sum(phi_i) = v(N) gives the identity
# Delta(empty) = sum(phi_i) + R. Both arms move with the coalition, so the game
# needs all 16 coalitions on BOTH arms, and only on those two.
LATTICE_DESIGNS: tuple[str, ...] = ("temporal", "random_fully_matched")

# These two exist solely to report the three-way decomposition of the gap
# (size effect, class-composition effect, remainder). They are measurements
# about the reported gap, not moves in the game, so they are run at the
# baseline coalition only and must never reach exact_shapley.
REFERENCE_DESIGNS: tuple[str, ...] = ("random", "random_matched")

#: The empty coalition, the only one the reference designs are run at.
BASELINE_CONFIG_ID: int = 0

#: Values of the ``role`` column in metrics.parquet.
LATTICE_ROLE: str = "lattice"
REFERENCE_ROLE: str = "reference"


#: A cell yielding fewer test samples than this cannot support a macro-F1 over
#: 32 classes. Hitting it is a design error in the cut-point choice, not a
#: condition to route around.
#:
#: Overridable per call for the AUT slot table, where A2 shrinks the first slot
#: to 16 test samples. The caller lowers it explicitly rather than the guard
#: being weakened for everyone.
MIN_TEST_SAMPLES: int = 20

#: Measured on the reference machine, used only to order the work queue.
#: Keys are the identifiers in :data:`sift.models.MODEL_NAMES`.
MEASURED_FIT_SECONDS: dict[str, float] = {
    "lightgbm": 18.0,
    "logreg": 3.8,
    "random_forest": 3.6,
    "mlp": 0.8,
}
_DEFAULT_FIT_SECONDS: float = 5.0

#: n_jobs=6 measured 1.99x against 1.72x at 4 and 1.60x at 8. The ceiling is
#: memory bandwidth, not thread count, so this is an empirical value.
PARALLEL_KWARGS: dict[str, Any] = {
    "n_jobs": 6,
    "backend": "loky",
    "batch_size": 1,
    "pre_dispatch": "2*n_jobs",
}

COMPRESSION: str = "zstd"
COMPRESSION_LEVEL: int = 3

#: Panel columns the predictions table joins on.
REQUIRED_PANEL_COLUMNS: tuple[str, ...] = ("sample_id", "grp_id")

#: Transient key marking a record served from the cache; never persisted.
FROM_CACHE_KEY: str = "_from_cache"

METRIC_COLUMNS: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "accuracy",
)

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "macro_f1": ("macro_f1", "f1_macro", "macro_f1_score"),
    "balanced_accuracy": ("balanced_accuracy", "bal_acc", "balanced_acc"),
    "mcc": ("mcc", "matthews_corrcoef", "matthews"),
    "accuracy": ("accuracy", "acc"),
}


class ContractError(RuntimeError):
    """Raised when a collaborating module does not match CONTRACT.md."""


def assert_design_partition() -> None:
    """Check the role assignment still covers exactly ``SplitSpec.DESIGNS``.

    Raises
    ------
    ContractError
        When a design is unclassified, classified twice, or unknown.
    """
    declared = set(SplitSpec.DESIGNS)
    lattice, reference = set(LATTICE_DESIGNS), set(REFERENCE_DESIGNS)
    overlap = lattice & reference
    if overlap:
        raise ContractError("designs classified as both roles: " + repr(sorted(overlap)))
    assigned = lattice | reference
    if assigned != declared:
        raise ContractError(
            "design roles do not partition SplitSpec.DESIGNS. Unclassified: "
            + repr(sorted(declared - assigned))
            + "; unknown: "
            + repr(sorted(assigned - declared))
            + ". Assign every design a role in experiment.py before running."
        )


def design_role(design: str) -> str:
    """Return ``LATTICE_ROLE`` or ``REFERENCE_ROLE`` for a design name."""
    if design in LATTICE_DESIGNS:
        return LATTICE_ROLE
    if design in REFERENCE_DESIGNS:
        return REFERENCE_ROLE
    raise ContractError("design " + repr(design) + " has no declared role")


class LatticeCellError(RuntimeError):
    """Raised when one lattice cell cannot be evaluated."""


class IncompleteLatticeError(RuntimeError):
    """Raised when a group does not carry all sixteen coalitions."""


# --- Parquet schemas, exactly as CONTRACT section 6 -------------------------
PREDICTIONS_SCHEMA: pa.Schema = pa.schema(
    [
        ("fit_id", pa.string()),
        ("sample_id", pa.int32()),
        ("grp_id", pa.int32()),
        ("y_true", pa.int16()),
        ("y_pred", pa.int16()),
        ("test_year", pa.int16()),
    ]
)

METRICS_SCHEMA: pa.Schema = pa.schema(
    [
        ("fit_id", pa.string()),
        ("run_id", pa.string()),
        ("config_id", pa.int8()),
        ("a1_prior", pa.bool_()),
        ("a2_labels", pa.bool_()),
        ("b1_fs", pa.bool_()),
        ("b2_axis", pa.bool_()),
        ("design", pa.string()),
        ("role", pa.string()),
        ("cut", pa.int16()),
        ("seed", pa.int16()),
        ("model", pa.string()),
        ("n_train", pa.int32()),
        ("n_test", pa.int32()),
        ("k_train", pa.int16()),
        ("k_test", pa.int16()),
        ("macro_f1", pa.float64()),
        ("balanced_accuracy", pa.float64()),
        ("mcc", pa.float64()),
        ("accuracy", pa.float64()),
        ("fit_seconds", pa.float32()),
    ]
)

METRICS_COLUMN_ORDER: tuple[str, ...] = tuple(METRICS_SCHEMA.names)
PREDICTIONS_COLUMN_ORDER: tuple[str, ...] = tuple(PREDICTIONS_SCHEMA.names)


# --- Adapters over signatures CONTRACT.md leaves unspecified ----------------
_MODEL_FACTORY_NAMES: tuple[str, ...] = ("build_model", "make_model", "make_estimator")
_METRIC_FUNCTION_NAMES: tuple[str, ...] = ("compute_metrics", "evaluate", "all_metrics")
_DESIGN_FIELD_NAMES: tuple[str, ...] = ("design", "kind", "mode", "strategy")
_CUT_FIELD_NAMES: tuple[str, ...] = ("cut_year", "cut", "cut_point", "year")

#: Attributes ``apply_controls`` must expose on its ControlledData result.
REQUIRED_CONTROLLED_ATTRS: tuple[str, ...] = (
    "train",
    "test",
    "columns",
    "time_column",
    "train_weights",
)


def _resolve(module: Any, candidates: Sequence[str], role: str) -> Any:
    """Find the first attribute of ``module`` named in ``candidates``.

    Parameters
    ----------
    module : Any
        Module to search.
    candidates : Sequence of str
        Acceptable attribute names, in preference order.
    role : str
        Human-readable role, used in the error message.

    Returns
    -------
    Any
        The resolved attribute.

    Raises
    ------
    ContractError
        When no candidate is present.
    """
    for name in candidates:
        found = getattr(module, name, None)
        if callable(found):
            return found
    public = sorted(n for n in dir(module) if not n.startswith("_"))
    raise ContractError(
        "CONTRACT.md does not pin the "
        + role
        + " signature and none of "
        + repr(tuple(candidates))
        + " exists in "
        + module.__name__
        + ". Public names present: "
        + repr(public)
    )


def _dataclass_field(spec: Any, candidates: Sequence[str], role: str) -> str:
    """Find the field of a dataclass instance matching one of ``candidates``."""
    if not dataclasses.is_dataclass(spec):
        raise ContractError(role + " expects a dataclass, got " + type(spec).__name__)
    names = {field.name for field in dataclasses.fields(spec)}
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ContractError(
        "SplitSpec has no "
        + role
        + " field among "
        + repr(tuple(candidates))
        + ". Fields present: "
        + repr(sorted(names))
    )


def _require_attrs(obj: Any, required: Sequence[str], role: str) -> None:
    """Assert that an object exposes every required attribute."""
    missing = [name for name in required if not hasattr(obj, name)]
    if missing:
        public = sorted(n for n in dir(obj) if not n.startswith("_"))
        raise ContractError(
            role
            + " is missing "
            + repr(missing)
            + ". Attributes present: "
            + repr(public)
        )


def _normalise_metrics(raw: Mapping[str, Any]) -> dict[str, float]:
    """Map a metrics dict onto the four contract column names.

    Parameters
    ----------
    raw : Mapping
        Output of the metrics module.

    Returns
    -------
    dict of str to float
        Exactly the keys in :data:`METRIC_COLUMNS`.

    Raises
    ------
    ContractError
        When a required metric is absent under any known alias.
    """
    resolved: dict[str, float] = {}
    for column, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            if alias in raw:
                resolved[column] = float(raw[alias])
                break
        else:
            raise ContractError(
                "metrics module returned no value for "
                + column
                + " under any of "
                + repr(aliases)
                + ". Keys returned: "
                + repr(sorted(raw))
            )
    return resolved


# --- Config construction ----------------------------------------------------
def split_spec_for(base_split: Any, design: str, cut_year: int) -> Any:
    """Derive a SplitSpec for one design and cut from a base spec.

    Parameters
    ----------
    base_split : Any
        Template SplitSpec carried by the base ExperimentConfig.
    design : str
        Either ``random`` or ``temporal``.
    cut_year : int
        Cut year for this cell.

    Returns
    -------
    Any
        A new SplitSpec instance.
    """
    design_field = _dataclass_field(base_split, _DESIGN_FIELD_NAMES, "design")
    cut_field = _dataclass_field(base_split, _CUT_FIELD_NAMES, "cut")
    return dataclasses.replace(
        base_split, **{design_field: design, cut_field: int(cut_year)}
    )


def describe_split(split_spec: Any) -> tuple[str, int]:
    """Read design and cut year back out of a SplitSpec.

    Parameters
    ----------
    split_spec : Any
        SplitSpec instance.

    Returns
    -------
    tuple of (str, int)
        Design name and cut year.
    """
    design_field = _dataclass_field(split_spec, _DESIGN_FIELD_NAMES, "design")
    cut_field = _dataclass_field(split_spec, _CUT_FIELD_NAMES, "cut")
    return str(getattr(split_spec, design_field)), int(getattr(split_spec, cut_field))


def flag_mapping(flags: ControlFlags) -> dict[str, bool]:
    """Return the four controls as an ordered plain dict.

    Parameters
    ----------
    flags : ControlFlags
        Control flag set.

    Returns
    -------
    dict of str to bool
        Keys in :data:`CONTROL_NAMES` order.
    """
    return {name: bool(getattr(flags, name)) for name in CONTROL_NAMES}


def panel_fingerprint(panel: pd.DataFrame) -> str:
    """Return the content fingerprint of the panel actually passed in.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.

    Returns
    -------
    str
        32-character digest identifying this exact panel.

    Notes
    -----
    Deliberately not memoised on ``panel.attrs``. Memoising made the digest a
    property the frame *carried* rather than one derived from its contents, so
    a derived panel inheriting a parent's ``attrs`` would key every fit on an
    identity it does not have -- the same class of collision that keying on the
    raw files alone produced. Hashing costs about 90 ms; callers that need the
    saving pass the digest explicitly via ``run_cell(..., panel_digest=...)``.
    """
    return cache.frame_fingerprint(panel)


def label_categories(panel: pd.DataFrame, target: str) -> np.ndarray:
    """Return the panel-wide, sorted label vocabulary.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    target : str
        Label column, normally ``ransomware_family``.

    Returns
    -------
    numpy.ndarray
        Unique labels in ascending order.
    """
    return np.sort(panel[target].dropna().unique())


def _encode(values: pd.Series, categories: np.ndarray) -> np.ndarray:
    codes = pd.Categorical(values, categories=categories).codes
    if (codes < 0).any():
        raise ContractError(
            "labels outside the panel vocabulary appeared in a split: "
            + repr(sorted(set(values[codes < 0])))
        )
    return codes.astype(np.int16)


def describe_cell(cfg: ExperimentConfig) -> str:
    """Render a one-line identifier for a lattice cell, for error messages."""
    design, cut = describe_split(cfg.split)
    active = sorted(flag for flag, on in flag_mapping(cfg.flags).items() if on)
    return (
        "config_id="
        + str(cfg.flags.to_index())
        + " active="
        + (",".join(active) if active else "none")
        + " design="
        + design
        + " cut="
        + str(cut)
        + " model="
        + cfg.model_name
        + " seed="
        + str(cfg.seed)
    )


# --- One cell ---------------------------------------------------------------
def run_cell(
    panel: pd.DataFrame,
    cfg: ExperimentConfig,
    panel_digest: str | None = None,
    min_test_samples: int | None = None,
) -> dict[str, object]:
    """Evaluate a single lattice cell, using the fit cache when possible.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel produced by ``build_panel``.
    cfg : ExperimentConfig
        Fully specified cell: controls, split, model and seed.
    panel_digest : str, optional
        Fingerprint of ``panel``, when the caller has already computed it. Omitted, it
        is derived from ``panel`` here.
    min_test_samples : int, optional
        Floor on test-set size, defaulting to :data:`MIN_TEST_SAMPLES`. Lower
        it only for a measurement that has been ruled to need a smaller cell,
        such as the full-coalition AUT slot table.

    Returns
    -------
    dict
        One tidy metrics row plus a ``predictions`` sub-dict of numpy arrays holding
        ``sample_id``, ``grp_id``, ``y_true``, ``y_pred`` and ``test_year``.

    Raises
    ------
    LatticeCellError
        When the cell produces fewer than :data:`MIN_TEST_SAMPLES` test rows,
        or when an active control cannot be applied to the chosen estimator.
    ContractError
        When a collaborating module deviates from CONTRACT.md.
    """
    missing = [c for c in REQUIRED_PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise ContractError(
            "panel is missing " + repr(missing) + "; predictions.parquet requires them"
        )

    design, cut = describe_split(cfg.split)
    flags = flag_mapping(cfg.flags)

    model_factory = _resolve(models_module, _MODEL_FACTORY_NAMES, "model factory")
    metric_function = _resolve(metrics_module, _METRIC_FUNCTION_NAMES, "metrics")

    model_seed = derive_seed(cfg.seed, "model", cfg.model_name, cut)
    estimator = model_factory(cfg.model_name, model_seed)

    identifier = cache.fit_id(
        cache.build_payload(
            panel_fingerprint=(
                panel_digest if panel_digest is not None else panel_fingerprint(panel)
            ),
            panel_spec=dataclasses.asdict(cfg.panel),
            flags=flags,
            design=design,
            cut=cut,
            seed=int(cfg.seed),
            model_name=cfg.model_name,
            model_params=estimator.get_params(),
            target=cfg.target,
            n_features=int(cfg.n_features),
        )
    )
    cached = cache.get(identifier)
    if cached is not None:
        # Transient marker, stripped before the metrics table is written. A
        # cached record keeps its original fit_seconds, so elapsed time cannot
        # be used to tell a hit from a miss after the fact.
        cached[FROM_CACHE_KEY] = True
        return cached

    controlled = apply_controls(panel, cfg)
    _require_attrs(controlled, REQUIRED_CONTROLLED_ATTRS, "ControlledData")

    categories = label_categories(panel, cfg.target)
    columns = list(controlled.columns)
    train, test = controlled.train, controlled.test
    X_train = train.loc[:, columns].to_numpy(dtype=np.float32)
    X_test = test.loc[:, columns].to_numpy(dtype=np.float32)
    y_train = _encode(train[cfg.target], categories)
    y_test = _encode(test[cfg.target], categories)

    floor = MIN_TEST_SAMPLES if min_test_samples is None else int(min_test_samples)
    if len(y_test) < floor:
        raise LatticeCellError(
            "lattice cell yields only "
            + str(len(y_test))
            + " test samples, below the floor of "
            + str(floor)
            + " -> "
            + describe_cell(cfg)
            + ". Exact Shapley needs all 16 coalitions, so choose cut years "
            "where the strictest cell survives rather than dropping this one."
        )

    # A1 is an intervention on the measurement, not on the learner, so it
    # belongs in the metric, where it reweights the test window to the reference
    # prior. Reweighting the training set would be a mitigation, a different
    # question, and inexpressible for estimators whose fit takes no
    # sample_weight.
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    y_pred = np.asarray(estimator.predict(X_test))

    scores = _normalise_metrics(
        dict(metric_function(y_test, y_pred, sample_weight=controlled.test_weights))
    )

    # test_year is read off the axis this cell actually split on, so it names
    # the sample's position in the test window rather than an unrelated year.
    year_column = controlled.time_column
    if year_column not in test.columns:
        raise ContractError(
            "controlled.time_column=" + str(year_column) + " is not a panel column"
        )

    record: dict[str, object] = {
        "fit_id": identifier,
        "config_id": int(cfg.flags.to_index()),
        **flags,
        "design": design,
        "role": design_role(design),
        "cut": int(cut),
        "seed": int(cfg.seed),
        "model": cfg.model_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "k_train": int(np.unique(y_train).size),
        "k_test": int(np.unique(y_test).size),
        **scores,
        "fit_seconds": float(fit_seconds),
        "predictions": {
            "sample_id": test["sample_id"].to_numpy(dtype=np.int32),
            "grp_id": test["grp_id"].to_numpy(dtype=np.int32),
            "y_true": np.asarray(y_test, dtype=np.int16),
            "y_pred": y_pred.astype(np.int16),
            "test_year": test[year_column].to_numpy(dtype=np.int16),
        },
    }
    cache.put(identifier, record)
    return record


# --- Worker plumbing --------------------------------------------------------
@lru_cache(maxsize=2)
def _panel_from_pickle(path: str) -> pd.DataFrame:
    # loky reuses worker processes, so the panel is unpickled once per worker
    # rather than once per task. Passing the DataFrame directly would pickle
    # roughly 6 MB per task, which at batch_size=1 dominates the fit itself.
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _run_one(
    panel_path: str, cfg: ExperimentConfig, panel_digest: str
) -> dict[str, object]:
    """Worker entry point: load the shared panel, then evaluate one cell."""
    return run_cell(_panel_from_pickle(panel_path), cfg, panel_digest=panel_digest)


def _estimated_seconds(model_name: str, cut_year: int) -> float:
    """Rank a job by expected cost so the queue runs longest-first."""
    base = MEASURED_FIT_SECONDS.get(model_name, _DEFAULT_FIT_SECONDS)
    return base * max(int(cut_year) - 2011, 1)


def build_jobs(
    base: ExperimentConfig,
    model_names: Sequence[str],
    seeds: Sequence[int],
    cut_years: Sequence[int],
) -> list[ExperimentConfig]:
    """Enumerate every cell of the lattice, ordered longest-job-first.

    Parameters
    ----------
    base : ExperimentConfig
        Template supplying panel spec, target and feature budget.
    model_names : Sequence of str
        Estimators to run.
    seeds : Sequence of int
        Base seeds. Both designs use the same list, because three of the four
        models are stochastic and a single temporal point would leave the
        temporal side of Delta without a variance.
    cut_years : Sequence of int
        Temporal cut points.

    Returns
    -------
    list of ExperimentConfig
        ``34 * len(cuts) * len(models) * len(seeds)`` configs.

    Notes
    -----
    The 34 is deliberate and is not a loop bound that happens to come out that
    way. Per (cut, model, seed) the scope is:
    """
    assert_design_partition()

    jobs: list[ExperimentConfig] = []
    for cut_year in cut_years:
        for design in LATTICE_DESIGNS:
            split = split_spec_for(base.split, design, cut_year)
            for index in range(N_COALITIONS):
                flags = ControlFlags.from_index(index)
                for model_name in model_names:
                    for seed in seeds:
                        jobs.append(
                            dataclasses.replace(
                                base,
                                split=split,
                                flags=flags,
                                model_name=model_name,
                                seed=int(seed),
                            )
                        )
        for design in REFERENCE_DESIGNS:
            split = split_spec_for(base.split, design, cut_year)
            baseline = ControlFlags.from_index(BASELINE_CONFIG_ID)
            for model_name in model_names:
                for seed in seeds:
                    jobs.append(
                        dataclasses.replace(
                            base,
                            split=split,
                            flags=baseline,
                            model_name=model_name,
                            seed=int(seed),
                        )
                    )
    jobs.sort(
        key=lambda cfg: _estimated_seconds(cfg.model_name, describe_split(cfg.split)[1]),
        reverse=True,
    )
    return jobs


# --- Table assembly and completeness checks ---------------------------------
def _metrics_frame(records: Sequence[Mapping[str, Any]], run_id: str) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            k: v
            for k, v in record.items()
            if k != "predictions" and not k.startswith("_")
        }
        row["run_id"] = run_id
        rows.append(row)
    frame = pd.DataFrame(rows)
    missing = [c for c in METRICS_COLUMN_ORDER if c not in frame.columns]
    if missing:
        raise ContractError("metrics frame is missing columns " + repr(missing))
    return frame.loc[:, list(METRICS_COLUMN_ORDER)]


def _predictions_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    blocks = []
    for record in records:
        payload = record["predictions"]
        block = pd.DataFrame(payload)
        block.insert(0, "fit_id", record["fit_id"])
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    return frame.loc[:, list(PREDICTIONS_COLUMN_ORDER)]


def lattice_cells(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows that are moves in the game.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Assembled metrics table.

    Returns
    -------
    pandas.DataFrame
        Rows on the two lattice designs. This is the frame to build a value
        function from; passing the whole table to ``exact_shapley`` would mix
        in reference measurements taken at the baseline coalition only.
    """
    return metrics.loc[metrics["role"] == LATTICE_ROLE]


def reference_cells(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return only the baseline reference rows used for the gap decomposition."""
    return metrics.loc[metrics["role"] == REFERENCE_ROLE]


def assert_lattice_complete(metrics: pd.DataFrame) -> None:
    """Verify the table is a total value function plus its reference rows.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Assembled metrics table.

    Raises
    ------
    IncompleteLatticeError
        When a lattice design is short of, or duplicates, a coalition, or when
        a reference design carries anything other than the baseline coalition.
    """
    for design in LATTICE_DESIGNS:
        arm = metrics.loc[metrics["design"] == design]
        if arm.empty:
            raise IncompleteLatticeError(
                "lattice design " + design + " produced no rows at all"
            )
        missing = set(range(N_COALITIONS)) - set(arm["config_id"].unique().tolist())
        if missing:
            raise IncompleteLatticeError(
                design + " is missing coalitions " + repr(sorted(missing))
            )
        counts = arm.groupby(["cut", "model", "seed"])["config_id"].nunique()
        short = counts[counts != N_COALITIONS]
        if not short.empty:
            raise IncompleteLatticeError(
                "these " + design + " groups do not carry all 16 coalitions:"
                + chr(10) + short.to_string()
            )

    for design in REFERENCE_DESIGNS:
        arm = metrics.loc[metrics["design"] == design]
        if arm.empty:
            continue
        stray = set(arm["config_id"].unique().tolist()) - {BASELINE_CONFIG_ID}
        if stray:
            raise IncompleteLatticeError(
                "reference design " + design + " carries non-baseline coalitions "
                + repr(sorted(stray))
                + ". Reference rows are measurements about the gap, not game "
                "cells, and must exist at the baseline only."
            )

    mislabelled = metrics.loc[metrics["role"] != metrics["design"].map(design_role)]
    if not mislabelled.empty:
        raise IncompleteLatticeError(
            "role column disagrees with design role for "
            + str(len(mislabelled)) + " rows"
        )

    duplicated = metrics.duplicated(
        subset=["cut", "design", "model", "seed", "config_id"]
    )
    if duplicated.any():
        raise IncompleteLatticeError(
            "duplicate cells found:" + chr(10)
            + metrics.loc[duplicated, ["cut", "design", "model", "seed", "config_id"]]
            .to_string(index=False)
        )



def _write_parquet(frame: pd.DataFrame, schema: pa.Schema, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(
        table, path, compression=COMPRESSION, compression_level=COMPRESSION_LEVEL
    )
    return path


# --- Full lattice -----------------------------------------------------------
def run_lattice(
    panel: pd.DataFrame,
    base: ExperimentConfig,
    model_names: Sequence[str],
    seeds: Sequence[int],
    cut_years: Sequence[int],
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the complete 16-coalition lattice and persist all three artefacts.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    base : ExperimentConfig
        Template config; split, flags, model and seed are overridden per cell.
    model_names : Sequence of str
        Estimators to run.
    seeds : Sequence of int
        Base seeds, applied to both designs.
    cut_years : Sequence of int
        Temporal cut points.
    results_dir : Path, optional
        Output directory. Defaults to ``results/``.

    Returns
    -------
    pandas.DataFrame
        The metrics table in tidy long format, one row per fit.

    Raises
    ------
    LatticeCellError
        Propagated from any cell that cannot be evaluated.
    IncompleteLatticeError
        When the assembled table is not a total value function.
    """
    out_dir = results_dir or cache.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = manifest.new_run_id()
    started_utc = manifest.utc_now()
    started = time.perf_counter()

    # Refuse a cache directory written under an older key composition before
    # any work is dispatched, rather than after an hour of fits.
    cache.assert_cache_compatible()

    jobs = build_jobs(base, model_names, seeds, cut_years)

    # Compute once here so no worker pays the 90 ms hash, and so every fit in
    # this run is keyed on the same panel identity.
    panel_digest = panel_fingerprint(panel)

    # Pin before any worker starts, so an edit landing mid-run cannot give
    # early and late fits different cache keys. The git state is captured at
    # the same instant and for the same reason: both describe the code the
    # fits actually ran under, not the code present when the run finished.
    pinned_code_version = cache.pin_code_version()
    git_at_start = manifest.git_info(cache.PROJECT_ROOT)

    # A run killed mid-flight leaves its panel pickle behind, because the
    # finally block never executes. Sweep them so they do not accumulate.
    for stale in out_dir.glob(".panel_worker_*.pkl"):
        stale.unlink(missing_ok=True)

    panel_path = out_dir / (".panel_worker_" + run_id + ".pkl")
    panel.to_pickle(panel_path)
    try:
        records: list[dict[str, object]] = Parallel(**PARALLEL_KWARGS)(
            delayed(_run_one)(str(panel_path), cfg, panel_digest) for cfg in jobs
        )
    finally:
        panel_path.unlink(missing_ok=True)
        _panel_from_pickle.cache_clear()
        cache.unpin_code_version()

    metrics_frame = _metrics_frame(records, run_id)
    assert_lattice_complete(metrics_frame)
    predictions_frame = _predictions_frame(records)

    _write_parquet(predictions_frame, PREDICTIONS_SCHEMA, out_dir / "predictions.parquet")
    _write_parquet(metrics_frame, METRICS_SCHEMA, out_dir / "metrics.parquet")

    wall_seconds = time.perf_counter() - started
    n_cached = sum(1 for record in records if record.get(FROM_CACHE_KEY))
    manifest.write_manifest(
        manifest.build_manifest(
            run_id=run_id,
            started_utc=started_utc,
            ended_utc=manifest.utc_now(),
            wall_seconds=wall_seconds,
            code_version=pinned_code_version,
            code_version_scope=cache.fit_path_modules(),
            git_at_start=git_at_start,
            counts={
                "fits_planned": len(jobs),
                "fits_recorded": len(records),
                "fits_from_cache": n_cached,
                "prediction_rows": len(predictions_frame),
            },
            threading={
                **PARALLEL_KWARGS,
                "estimator_n_jobs": 1,
                "note": "n_jobs=6 measured 1.99x; bottleneck is memory bandwidth",
            },
            rng_policy={
                "seeds": list(map(int, seeds)),
                "derivation": 'derive_seed(base_seed, "model", model_name, cut_year)',
                "designs_share_seed_list": True,
            },
            lattice={
                "control_names": list(CONTROL_NAMES),
                "n_coalitions": N_COALITIONS,
                "lattice_designs": list(LATTICE_DESIGNS),
                "reference_designs": list(REFERENCE_DESIGNS),
                "baseline_config_id": BASELINE_CONFIG_ID,
                "fits_per_cut_model_seed": 2 * N_COALITIONS + len(REFERENCE_DESIGNS),
                "value_function": "gap-based: v(S) = Delta(empty) - Delta(S)",
                "panel_fingerprint": panel_digest,
                "cut_years": list(map(int, cut_years)),
                "models": list(model_names),
                "min_test_samples": MIN_TEST_SAMPLES,
            },
        ),
        out_dir / "run_manifest.json",
    )
    return metrics_frame


# --- C1: the opt-in five-control lattice ------------------------------------
#
# Nothing above this line changed; the entry points below are reached only by a
# caller that asks for them by name.
#
# The extension rests on one fact: C1 is a property of the *panel*, not a step
# inside the pipeline. ``config_for_cell_c1`` turns an extended coalition into
# an ordinary four-control ``ExperimentConfig`` whose ``PanelSpec`` carries the
# state of C1, so every fit still goes through the unmodified ``run_cell``,
# ``apply_controls`` and cache path. Only ``config_id`` is relabelled, from 4
# bits to 5.

#: Number of coalitions in the extended lattice.
N_COALITIONS_C1: int = 2 ** len(CONTROL_NAMES_C1)

#: Extended ``config_id`` values the reference designs may carry: the baseline
#: coalition at each state of C1.
#:
#: Once C1 is a player the baseline is ``c1_dedup=False``, so index 0 is the one
#: the game needs. Index 16 is the same measurement on the deduplicated panel;
#: it is kept because it is already in the fit cache and dropping it would stop
#: the extended run reproducing the published three-way split.
BASELINE_CONFIG_IDS_C1: tuple[int, ...] = (
    BASELINE_CONFIG_ID,
    BASELINE_CONFIG_ID + N_COALITIONS,
)

#: Fits per (cut, model, seed) group in an extended run: 32 coalitions on each
#: of the two lattice designs, plus the two baselines on each of the two
#: reference designs.
JOBS_PER_GROUP_C1: int = 2 * N_COALITIONS_C1 + len(REFERENCE_DESIGNS) * 2


def _schema_with_c1(schema: pa.Schema) -> pa.Schema:
    """Return ``schema`` with a ``c1_dedup`` field inserted after ``b2_axis``."""
    fields = list(schema)
    at = [field.name for field in fields].index("b2_axis") + 1
    return pa.schema(fields[:at] + [pa.field("c1_dedup", pa.bool_())] + fields[at:])


METRICS_SCHEMA_C1: pa.Schema = _schema_with_c1(METRICS_SCHEMA)
"""``METRICS_SCHEMA`` with ``c1_dedup`` inserted after the four control flags.

A separate schema, written to a separate directory. ``results/metrics.parquet``
keeps its four-flag ``config_id`` and is not touched by anything here.
"""

METRICS_COLUMN_ORDER_C1: tuple[str, ...] = tuple(METRICS_SCHEMA_C1.names)


def flag_mapping_c1(flags: ExtendedControlFlags) -> dict[str, bool]:
    """Return the five controls as an ordered plain dict.

    Parameters
    ----------
    flags : sift.config.ExtendedControlFlags
        Extended control flag set.

    Returns
    -------
    dict of str to bool
        Keys in :data:`sift.config.CONTROL_NAMES_C1` order.
    """
    return {name: bool(getattr(flags, name)) for name in CONTROL_NAMES_C1}


def _stamp_c1(record: Mapping[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    """Relabel a four-control record as the extended cell it actually is.

    ``run_cell`` writes ``config_id = cfg.flags.to_index()``, which is the
    four-control index and collides between the two C1 states. The extended
    index and the ``c1_dedup`` column are written here instead of inside
    ``run_cell``, so the four-control path keeps producing byte-identical rows.
    """
    flags = extended_flags_of(cfg)
    stamped = dict(record)
    stamped["config_id"] = flags.to_index()
    stamped.update(flag_mapping_c1(flags))
    return stamped


def run_cell_c1(
    panels: PanelVariants,
    base: ExperimentConfig,
    flags: ExtendedControlFlags,
    panel_digests: Mapping[bool, str] | None = None,
    min_test_samples: int | None = None,
) -> dict[str, object]:
    """Evaluate a single cell of the extended lattice.

    Parameters
    ----------
    panels : sift.data.PanelVariants
        Both C1 variants of the panel.
    base : ExperimentConfig
        Template config; its ``flags`` and ``panel.dedup_exact`` are overridden.
    flags : sift.config.ExtendedControlFlags
        The extended coalition.
    panel_digests : Mapping of {bool: str}, optional
        Precomputed fingerprints of the two variants, keyed by the state of C1.
    min_test_samples : int, optional
        Floor on test-set size, as in :func:`run_cell`.

    Returns
    -------
    dict
        A metrics row as :func:`run_cell` returns it, with ``config_id`` in
        ``range(32)`` and an added ``c1_dedup`` column.
    """
    cfg = config_for_cell_c1(base, flags)
    digest = None if panel_digests is None else panel_digests.get(flags.c1_dedup)
    record = run_cell(
        panels.select(flags.c1_dedup),
        cfg,
        panel_digest=digest,
        min_test_samples=min_test_samples,
    )
    return _stamp_c1(record, cfg)


def build_jobs_c1(
    base: ExperimentConfig,
    model_names: Sequence[str],
    seeds: Sequence[int],
    cut_years: Sequence[int],
) -> list[ExperimentConfig]:
    """Enumerate every cell of the extended lattice, ordered longest-job-first.

    Parameters
    ----------
    base : ExperimentConfig
        Template supplying panel spec, target and feature budget.
    model_names, seeds, cut_years : Sequence
        As in :func:`build_jobs`.

    Returns
    -------
    list of ExperimentConfig
        ``68 * len(cuts) * len(models) * len(seeds)`` configs, each already projected
        onto the four-control form by :func:`sift.config.config_for_cell_c1`.
    """
    assert_design_partition()

    jobs: list[ExperimentConfig] = []
    for cut_year in cut_years:
        for design in LATTICE_DESIGNS:
            split = split_spec_for(base.split, design, cut_year)
            for index in range(N_COALITIONS_C1):
                flags = ExtendedControlFlags.from_index(index)
                for model_name in model_names:
                    for seed in seeds:
                        jobs.append(
                            config_for_cell_c1(
                                dataclasses.replace(
                                    base,
                                    split=split,
                                    model_name=model_name,
                                    seed=int(seed),
                                ),
                                flags,
                            )
                        )
        for design in REFERENCE_DESIGNS:
            split = split_spec_for(base.split, design, cut_year)
            for c1_dedup in (False, True):
                flags = ExtendedControlFlags.from_flags(
                    ControlFlags.from_index(BASELINE_CONFIG_ID), c1_dedup
                )
                for model_name in model_names:
                    for seed in seeds:
                        jobs.append(
                            config_for_cell_c1(
                                dataclasses.replace(
                                    base,
                                    split=split,
                                    model_name=model_name,
                                    seed=int(seed),
                                ),
                                flags,
                            )
                        )
    jobs.sort(
        key=lambda cfg: _estimated_seconds(cfg.model_name, describe_split(cfg.split)[1]),
        reverse=True,
    )
    return jobs


def _metrics_frame_c1(records: Sequence[Mapping[str, Any]], run_id: str) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            k: v
            for k, v in record.items()
            if k != "predictions" and not k.startswith("_")
        }
        row["run_id"] = run_id
        rows.append(row)
    frame = pd.DataFrame(rows)
    missing = [c for c in METRICS_COLUMN_ORDER_C1 if c not in frame.columns]
    if missing:
        raise ContractError("metrics frame is missing columns " + repr(missing))
    return frame.loc[:, list(METRICS_COLUMN_ORDER_C1)]


def assert_lattice_complete_c1(metrics: pd.DataFrame) -> None:
    """Verify an extended metrics table is a total 32-coalition value function.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Assembled extended metrics table.

    Raises
    ------
    IncompleteLatticeError
        When a lattice design is short of, or duplicates, a coalition, or when
        a reference design carries anything but the two baseline rows.
    """
    for design in LATTICE_DESIGNS:
        arm = metrics.loc[metrics["design"] == design]
        if arm.empty:
            raise IncompleteLatticeError(
                "lattice design " + design + " produced no rows at all"
            )
        missing = set(range(N_COALITIONS_C1)) - set(arm["config_id"].unique().tolist())
        if missing:
            raise IncompleteLatticeError(
                design + " is missing coalitions " + repr(sorted(missing))
            )
        counts = arm.groupby(["cut", "model", "seed"])["config_id"].nunique()
        short = counts[counts != N_COALITIONS_C1]
        if not short.empty:
            raise IncompleteLatticeError(
                "these " + design + " groups do not carry all 32 coalitions:"
                + chr(10) + short.to_string()
            )

    for design in REFERENCE_DESIGNS:
        arm = metrics.loc[metrics["design"] == design]
        if arm.empty:
            continue
        stray = set(arm["config_id"].unique().tolist()) - set(BASELINE_CONFIG_IDS_C1)
        if stray:
            raise IncompleteLatticeError(
                "reference design " + design + " carries non-baseline coalitions "
                + repr(sorted(stray))
                + ". Reference rows are measurements about the gap, not game "
                "cells, and must exist at the baseline only."
            )

    mislabelled = metrics.loc[metrics["role"] != metrics["design"].map(design_role)]
    if not mislabelled.empty:
        raise IncompleteLatticeError(
            "role column disagrees with design role for "
            + str(len(mislabelled)) + " rows"
        )

    duplicated = metrics.duplicated(
        subset=["cut", "design", "model", "seed", "config_id"]
    )
    if duplicated.any():
        raise IncompleteLatticeError(
            "duplicate cells found:" + chr(10)
            + metrics.loc[duplicated, ["cut", "design", "model", "seed", "config_id"]]
            .to_string(index=False)
        )


def _run_one_c1(
    panel_path: str, cfg: ExperimentConfig, panel_digest: str
) -> dict[str, object]:
    """Worker entry point for the extended lattice.

    ``_panel_from_pickle`` is memoised with ``maxsize=2``, which is exactly the
    number of panel variants, so each worker unpickles each variant once.
    """
    record = run_cell(_panel_from_pickle(panel_path), cfg, panel_digest=panel_digest)
    return _stamp_c1(record, cfg)


def run_lattice_c1(
    panels: PanelVariants,
    base: ExperimentConfig,
    model_names: Sequence[str],
    seeds: Sequence[int],
    cut_years: Sequence[int],
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the complete 32-coalition lattice and persist all three artefacts.

    Parameters
    ----------
    panels : sift.data.PanelVariants
        Both C1 variants of the panel.
    base : ExperimentConfig
        Template config; split, flags, panel spec, model and seed are overridden
        per cell.
    model_names, seeds, cut_years : Sequence
        As in :func:`run_lattice`.
    results_dir : Path, optional
        Output directory. Defaults to ``results/c1/``, deliberately **not**
        ``results/``: the four-control ``metrics.parquet`` is a published
        artefact with a four-flag ``config_id`` and must not be overwritten by
        a table whose ``config_id`` means something else.

    Returns
    -------
    pandas.DataFrame
        The extended metrics table, one row per fit.

    Raises
    ------
    LatticeCellError
        Propagated from any cell that cannot be evaluated. Cells with
        ``c1_dedup=False`` hold more samples than their deduplicated twin, so
        the test-size floor is easier to clear, not harder.
    IncompleteLatticeError
        When the assembled table is not a total value function.
    """
    out_dir = results_dir or (cache.RESULTS_DIR / "c1")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = manifest.new_run_id()
    started_utc = manifest.utc_now()
    started = time.perf_counter()

    cache.assert_cache_compatible()

    jobs = build_jobs_c1(base, model_names, seeds, cut_years)

    digests = {
        True: panel_fingerprint(panels.with_dedup),
        False: panel_fingerprint(panels.without_dedup),
    }
    if digests[True] == digests[False]:
        raise ContractError(
            "the two C1 panel variants have the same fingerprint, so every fit "
            "of one would be served the other's cached record. The raw frame "
            "carries no exact duplicates, and C1 is not measurable on it."
        )

    pinned_code_version = cache.pin_code_version()
    git_at_start = manifest.git_info(cache.PROJECT_ROOT)

    for stale in out_dir.glob(".panel_worker_*.pkl"):
        stale.unlink(missing_ok=True)

    paths = {
        state: out_dir / (".panel_worker_" + run_id + ("_dedup" if state else "_raw") + ".pkl")
        for state in (True, False)
    }
    panels.with_dedup.to_pickle(paths[True])
    panels.without_dedup.to_pickle(paths[False])
    try:
        records: list[dict[str, object]] = Parallel(**PARALLEL_KWARGS)(
            delayed(_run_one_c1)(
                str(paths[cfg.panel.dedup_exact]),
                cfg,
                digests[cfg.panel.dedup_exact],
            )
            for cfg in jobs
        )
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)
        _panel_from_pickle.cache_clear()
        cache.unpin_code_version()

    metrics_frame = _metrics_frame_c1(records, run_id)
    assert_lattice_complete_c1(metrics_frame)
    predictions_frame = _predictions_frame(records)

    _write_parquet(predictions_frame, PREDICTIONS_SCHEMA, out_dir / "predictions.parquet")
    _write_parquet(metrics_frame, METRICS_SCHEMA_C1, out_dir / "metrics.parquet")

    wall_seconds = time.perf_counter() - started
    n_cached = sum(1 for record in records if record.get(FROM_CACHE_KEY))
    manifest.write_manifest(
        manifest.build_manifest(
            run_id=run_id,
            started_utc=started_utc,
            ended_utc=manifest.utc_now(),
            wall_seconds=wall_seconds,
            code_version=pinned_code_version,
            code_version_scope=cache.fit_path_modules(),
            git_at_start=git_at_start,
            counts={
                "fits_planned": len(jobs),
                "fits_recorded": len(records),
                "fits_from_cache": n_cached,
                "prediction_rows": len(predictions_frame),
            },
            threading={
                **PARALLEL_KWARGS,
                "estimator_n_jobs": 1,
                "note": "n_jobs=6 measured 1.99x; bottleneck is memory bandwidth",
            },
            rng_policy={
                "seeds": list(map(int, seeds)),
                "derivation": 'derive_seed(base_seed, "model", model_name, cut_year)',
                "designs_share_seed_list": True,
            },
            lattice={
                "control_names": list(CONTROL_NAMES_C1),
                "control_order": list(CONTROL_ORDER_C1),
                "n_coalitions": N_COALITIONS_C1,
                "lattice_designs": list(LATTICE_DESIGNS),
                "reference_designs": list(REFERENCE_DESIGNS),
                "baseline_config_id": BASELINE_CONFIG_ID,
                "baseline_config_ids": list(BASELINE_CONFIG_IDS_C1),
                "fits_per_cut_model_seed": JOBS_PER_GROUP_C1,
                "value_function": "gap-based: v(S) = Delta(empty) - Delta(S)",
                "panel_fingerprint": digests[True],
                "panel_fingerprint_no_dedup": digests[False],
                "cut_years": list(map(int, cut_years)),
                "models": list(model_names),
                "min_test_samples": MIN_TEST_SAMPLES,
            },
        ),
        out_dir / "run_manifest.json",
    )
    return metrics_frame
