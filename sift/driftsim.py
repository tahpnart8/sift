"""Synthetic concept-drift panels that give the residual R a ground truth.

The residual ``R = Delta(N)`` is defined in :mod:`sift.shapley` as what the four
controls cannot explain. Whether any of it is genuine concept drift is exactly
what the framework cannot settle on real data, because drift magnitude is not
observable on real MLRan features: there is no column that says how much
``P(y | x)`` moved between 2015 and 2021.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy
import pandas

from sift import cache
from sift.config import CONTROL_NAMES, ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import apply_controls
from sift.experiment import (
    LATTICE_DESIGNS,
    design_role,
    panel_fingerprint,
    run_cell,
    split_spec_for,
)
from sift.features import feature_columns
from sift.metrics import bootstrap_distribution
from sift.paths import MLRAN_DIR, RESULTS_DIR
from sift.seeding import derive_seed, make_rng
from sift.shapley import decompose, gap_values

__all__ = [
    "DEFAULT_LEVELS",
    "DEFAULT_SEEDS",
    "DRIFTSIM_DIR",
    "RIGHT_ANGLE",
    "DriftSpec",
    "assert_controls_are_null",
    "base_config",
    "drift_weights",
    "feature_probabilities",
    "generate_panel",
    "mlran_marginals",
    "panel_spec_for",
    "residual_bootstrap",
    "rotation_basis",
    "rule_disagreement",
    "run_ladder",
    "run_level",
    "write_results",
]

#: A right angle. The drift level is ``theta / RIGHT_ANGLE``, so it runs from 0
#: (rule unchanged) to 1 (rule orthogonal to the original, hence fully replaced).
RIGHT_ANGLE: float = float(numpy.pi / 2.0)

#: The five injected drift levels of the ladder.
DEFAULT_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Three seeds. ``logreg`` with the lbfgs solver is exactly deterministic given
#: a training matrix, so the seed list varies the *matched random draw* and
#: nothing else; the temporal arm is identical across all three by construction.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: The single model. Chosen because it is deterministic, so any movement in R
#: across the ladder is the injected drift rather than optimiser noise.
DEFAULT_MODEL: str = "logreg"

#: Output directory. Nothing outside it is written.
DRIFTSIM_DIR: Path = RESULTS_DIR / "driftsim"

#: Metric the gap is measured on, matching the main experiment.
METRIC: str = "macro_f1"

#: Tolerance for the null-by-construction assertions. The A1 weight vector is
#: exactly one when a window carries exactly the reference base rate; the matched
#: random draw can round its stratified allocation by a sample, so a small slack
#: is allowed and the realised deviation is recorded rather than assumed away.
NULL_TOLERANCE: float = 1e-6


# --- Specification ----------------------------------------------------------
@dataclass(frozen=True)
class DriftSpec:
    """Everything that defines one synthetic drift panel.

    Parameters
    ----------
    drift_level : float, default 0.0
        Injected drift, ``theta / (pi / 2)``, in ``[0, 1]``.
    n_features : int, default 100
        Number of binary features ``d``.
    samples_per_year : int, default 300
        Rows per year, identical for every year. Holding it constant removes a second
        confound that the lattice does not model: a window whose sample count moves with
        time would make the temporal and matched designs differ in ways unrelated to
        drift.
    year_min, year_max : int, default 2012 and 2023
        Inclusive bounds of the synthetic panel's year range.
    cut_year : int, default 2018
        The single cut point. Rows dated before it carry the early rule; rows
        dated at or after it carry the rotated rule.
    test_window : int, default 3
        Width of the temporal test window in years.
    base_rate : float, default 0.5
        Positive rate, held exactly constant in every year.
    signal_sd : float, default 3.0
        ``||w||``, which in standardised coordinates is exactly the standard deviation
        of the linear predictor.
    seed : int, default 0
        Seed of the generator. Distinct from the experiment seeds, which vary
        the split draw rather than the data.
    """

    drift_level: float = 0.0
    n_features: int = 100
    samples_per_year: int = 300
    year_min: int = 2012
    year_max: int = 2023
    cut_year: int = 2018
    test_window: int = 3
    base_rate: float = 0.5
    signal_sd: float = 3.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.drift_level <= 1.0:
            raise ValueError(f"drift_level must lie in [0, 1], got {self.drift_level}")
        if self.n_features < 2:
            raise ValueError(f"n_features must be at least 2, got {self.n_features}")
        if self.samples_per_year < 2 or self.samples_per_year % 2:
            raise ValueError(
                f"samples_per_year must be a positive even number, got "
                f"{self.samples_per_year}"
            )
        if self.year_min >= self.year_max:
            raise ValueError(f"year_min {self.year_min} must precede year_max {self.year_max}")
        if not self.year_min < self.cut_year <= self.year_max:
            raise ValueError(
                f"cut_year {self.cut_year} must leave both a training and a test window "
                f"inside [{self.year_min}, {self.year_max}]"
            )
        if self.cut_year + self.test_window - 1 > self.year_max:
            raise ValueError(
                f"the test window {self.cut_year}-{self.cut_year + self.test_window - 1} "
                f"runs past year_max {self.year_max}"
            )
        if not 0.0 < self.base_rate < 1.0:
            raise ValueError(f"base_rate must lie in (0, 1), got {self.base_rate}")
        if self.signal_sd <= 0.0:
            raise ValueError(f"signal_sd must be positive, got {self.signal_sd}")
        if not float(self.base_rate * self.samples_per_year).is_integer():
            raise ValueError(
                f"base_rate {self.base_rate} times samples_per_year "
                f"{self.samples_per_year} is not an integer, so no year can carry the "
                "reference prior exactly and control A1 would not be null"
            )

    @property
    def theta(self) -> float:
        """Rotation angle in radians."""
        return float(self.drift_level) * RIGHT_ANGLE

    @property
    def years(self) -> numpy.ndarray:
        """Every year the panel spans, ascending."""
        return numpy.arange(self.year_min, self.year_max + 1, dtype=numpy.int64)

    @property
    def n_samples(self) -> int:
        """Total rows in the panel."""
        return int(self.years.size) * int(self.samples_per_year)


# --- Bernoulli parameters from the real panel -------------------------------
@lru_cache(maxsize=1)
def mlran_marginals(data_dir: Path = MLRAN_DIR) -> numpy.ndarray:
    """Return the marginal activation frequency of every real MLRan feature.

    The binary task is used rather than the family task, because it retains
    goodware and therefore the full sparsity range the release actually carries.

    Parameters
    ----------
    data_dir : pathlib.Path, default :data:`sift.paths.MLRAN_DIR`
        Directory holding the MLRan release.

    Returns
    -------
    numpy.ndarray
        One frequency per feature column, in panel column order.
    """
    from sift.data import build_panel, load_mlran

    panel, _ = build_panel(load_mlran(data_dir), PanelSpec(task="binary"))
    columns = feature_columns(panel)
    return panel[columns].to_numpy(dtype=numpy.float64).mean(axis=0)


def feature_probabilities(
    spec: DriftSpec,
    marginals: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Draw ``d`` Bernoulli parameters from the real marginal frequencies.

    Sampling from the real marginals rather than truncating to the first ``d``
    of them keeps the sparsity *distribution* realistic, which is what the
    generator needs: a logistic model over features that are all near one half
    would be far easier than the real problem.

    Parameters
    ----------
    spec : DriftSpec
        Specification; ``n_features`` and ``seed`` are read.
    marginals : numpy.ndarray, optional
        Frequencies to sample from. Defaults to :func:`mlran_marginals`, which
        requires the MLRan release; pass an array to run without it.

    Returns
    -------
    numpy.ndarray
        ``spec.n_features`` probabilities, each clipped into ``[0.01, 0.99]`` so
        that no column is constant and every standardised column is finite.
    """
    source = numpy.asarray(mlran_marginals() if marginals is None else marginals, dtype=float)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("marginals must be a non-empty one-dimensional array")
    rng = make_rng(derive_seed(spec.seed, "driftsim", "marginals", spec.n_features))
    drawn = rng.choice(source, size=spec.n_features, replace=source.size < spec.n_features)
    return numpy.clip(numpy.sort(drawn), 0.01, 0.99)


# --- The rotation -----------------------------------------------------------
def rotation_basis(n_features: int, seed: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return the orthonormal plane the weight vector rotates in.

    Parameters
    ----------
    n_features : int
        Dimension of the parameter space.
    seed : int
        Seed for the draw.

    Returns
    -------
    u, v : numpy.ndarray
        Two unit vectors with ``u . v == 0``. The rotation path
        ``cos(theta) u + sin(theta) v`` stays on the unit sphere, so scaling it
        by ``||w||`` holds the norm constant at every angle.
    """
    rng = make_rng(derive_seed(seed, "driftsim", "rotation", n_features))
    u = rng.standard_normal(n_features)
    u /= numpy.linalg.norm(u)
    v = rng.standard_normal(n_features)
    v -= u * float(v @ u)
    v /= numpy.linalg.norm(v)
    return u, v


def drift_weights(spec: DriftSpec) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return the early and late weight vectors of one drift level.

    Parameters
    ----------
    spec : DriftSpec
        Specification; ``n_features``, ``signal_sd``, ``seed`` and
        ``drift_level`` are read.

    Returns
    -------
    w_early, w_late : numpy.ndarray
        Two vectors of identical Euclidean norm ``spec.signal_sd``, separated by
        ``spec.theta`` radians.
    """
    u, v = rotation_basis(spec.n_features, spec.seed)
    theta = spec.theta
    w_early = spec.signal_sd * u
    w_late = spec.signal_sd * (numpy.cos(theta) * u + numpy.sin(theta) * v)
    return w_early, w_late


def rule_disagreement(
    spec: DriftSpec,
    probabilities: numpy.ndarray | None = None,
    n_draw: int = 20_000,
) -> float:
    """Measure the injected drift directly, as a rule-flip rate.

    This is the ground truth stated in the units the reader cares about: the
    share of the feature space on which the early and late decision rules
    disagree about the more likely label. It is a property of the generator, not
    of any fit, and it is reported alongside ``R`` so that the ladder can be read
    without trusting the angle parameterisation.

    Parameters
    ----------
    spec : DriftSpec
        Specification.
    probabilities : numpy.ndarray, optional
        Bernoulli parameters. Defaults to :func:`feature_probabilities`.
    n_draw : int, default 20000
        Monte Carlo sample size.

    Returns
    -------
    float
        Fraction of drawn feature vectors whose predicted class flips.
    """
    p = feature_probabilities(spec) if probabilities is None else numpy.asarray(probabilities)
    rng = make_rng(derive_seed(spec.seed, "driftsim", "disagreement"))
    x = (rng.random((n_draw, p.size)) < p).astype(numpy.float64)
    z = (x - p) / numpy.sqrt(p * (1.0 - p))
    w_early, w_late = drift_weights(spec)
    return float(numpy.mean((z @ w_early > 0.0) != (z @ w_late > 0.0)))


# --- The panel --------------------------------------------------------------
def generate_panel(
    spec: DriftSpec,
    probabilities: numpy.ndarray | None = None,
) -> pandas.DataFrame:
    """Generate one synthetic drift panel.

    Parameters
    ----------
    spec : DriftSpec
        Specification.
    probabilities : numpy.ndarray, optional
        Bernoulli parameters, one per feature. Defaults to
        :func:`feature_probabilities`, which reads the real MLRan marginals.

    Returns
    -------
    pandas.DataFrame
        A panel carrying the columns :mod:`sift.controls` and
        :mod:`sift.experiment` require, plus ``spec.n_features`` binary feature
        columns named by numeric identifier as in the real release.

    Notes
    -----
    Labels use the latent-variable form of the logistic model,
    ``y = 1[w . z + e > tau]`` with ``e`` standard logistic. ``tau`` is the
    within-year empirical quantile at ``1 - base_rate``, which is the logistic
    intercept estimated on the realised sample rather than assumed from the
    population. That is what makes the base rate exactly constant across years
    and therefore makes control A1 null; a fixed intercept would leave a
    sampling wobble in the class prior that A1 would legitimately pick up.
    """
    p = feature_probabilities(spec) if probabilities is None else numpy.asarray(probabilities)
    if p.size != spec.n_features:
        raise ValueError(
            f"probabilities holds {p.size} entries but spec.n_features is {spec.n_features}"
        )

    # The drift level is deliberately absent from this seed. Every level then
    # shares one feature matrix and one draw of logistic noise, so two panels
    # differ in the late-window labels and in nothing else: the ladder is a
    # paired design rather than five unrelated datasets.
    rng = make_rng(derive_seed(spec.seed, "driftsim", "panel"))
    n = spec.n_samples
    years = numpy.repeat(spec.years, spec.samples_per_year)

    x = (rng.random((n, spec.n_features)) < p).astype(numpy.int8)
    z = (x.astype(numpy.float64) - p) / numpy.sqrt(p * (1.0 - p))

    w_early, w_late = drift_weights(spec)
    late = years >= spec.cut_year
    latent = numpy.where(late, z @ w_late, z @ w_early)

    # Standard logistic noise: this is exactly the logistic link, written in the
    # latent form so the intercept can be set by a quantile.
    noise = numpy.log(rng.random(n)) - numpy.log1p(-rng.random(n))
    score = latent + noise

    n_positive = int(round(spec.base_rate * spec.samples_per_year))
    y = numpy.zeros(n, dtype=numpy.int64)
    for year in spec.years:
        rows = numpy.flatnonzero(years == year)
        order = numpy.argsort(-score[rows], kind="stable")
        y[rows[order[:n_positive]]] = 1

    features = pandas.DataFrame(x, columns=[str(i) for i in range(spec.n_features)])
    meta = pandas.DataFrame(
        {
            "sample_id": numpy.arange(1, n + 1, dtype=numpy.int64),
            "sample_type": y,
            # Two families, both present in every year in equal number, so
            # control A2 has no unseen family to remove.
            "ransomware_family": numpy.where(y == 1, "ransomware", "goodware"),
            "first_submission_date_year": years,
            # Control B2 is null by construction: the two axes are the same
            # column under two names, so they induce identical splits.
            "Year": years,
            "n_active": x.sum(axis=1).astype(numpy.int64),
            "grp_id": numpy.arange(n, dtype=numpy.int64),
        }
    )
    return pandas.concat([meta, features], axis=1)


def panel_spec_for(spec: DriftSpec) -> PanelSpec:
    """Return the :class:`~sift.config.PanelSpec` describing a synthetic panel.

    Parameters
    ----------
    spec : DriftSpec
        Specification.

    Returns
    -------
    sift.config.PanelSpec
        Binary task, window matching the synthetic year range, and
        ``prior_reference_rate`` equal to the injected base rate so that control
        A1 targets the prior the data already carries.
    """
    return PanelSpec(
        task="binary",
        year_min=spec.year_min,
        year_max=spec.year_max,
        prior_reference_rate=spec.base_rate,
    )


def base_config(spec: DriftSpec, model_name: str = DEFAULT_MODEL) -> ExperimentConfig:
    """Return the template configuration for one synthetic panel.

    Parameters
    ----------
    spec : DriftSpec
        Specification.
    model_name : str, default 'logreg'
        Estimator key.

    Returns
    -------
    sift.config.ExperimentConfig
        Template whose ``n_features`` equals the full feature count, which makes
        control B1 null: ``select_features`` returns every column whatever rows
        it is fitted on.
    """
    return ExperimentConfig(
        panel=panel_spec_for(spec),
        split=SplitSpec(
            design="temporal",
            cut_year=spec.cut_year,
            test_window=spec.test_window,
        ),
        flags=ControlFlags(),
        model_name=model_name,
        seed=0,
        target="sample_type",
        n_features=spec.n_features,
    )


# --- The null-by-construction guarantee, checked rather than asserted in prose ---
def assert_controls_are_null(
    panel: pandas.DataFrame,
    spec: DriftSpec,
    model_name: str = DEFAULT_MODEL,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    tolerance: float = NULL_TOLERANCE,
) -> dict[str, float]:
    """Verify that no control changes anything on this panel.

    A false positive in the decomposition is only interpretable if the panel
    really carries no confound, so the claim is checked on the realised data
    rather than argued from the construction. For every design and seed the
    sixteen coalitions must agree on the training half, the test half, the
    selected columns and the temporal axis values, and the A1 weight vector must
    be unity.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of :func:`generate_panel`.
    spec : DriftSpec
        Specification the panel was generated from.
    model_name : str, default 'logreg'
        Estimator key, needed only to complete the configuration.
    seeds : Sequence of int
        Seeds to check.
    tolerance : float
        Largest weight deviation from one that is accepted.

    Returns
    -------
    dict
        ``max_weight_deviation``, the largest ``|w - 1|`` observed over every
        coalition, design and seed.

    Raises
    ------
    AssertionError
        If any coalition changes the split, the feature set or the axis values.
    """
    base = base_config(spec, model_name)
    worst = 0.0
    for design in LATTICE_DESIGNS:
        for seed in seeds:
            reference: tuple | None = None
            for config_id in range(2 ** len(CONTROL_NAMES)):
                cfg = replace(
                    base,
                    split=split_spec_for(base.split, design, spec.cut_year),
                    flags=ControlFlags.from_index(config_id),
                    seed=int(seed),
                )
                controlled = apply_controls(panel, cfg)
                if controlled.test_weights is not None:
                    worst = max(
                        worst, float(numpy.abs(controlled.test_weights - 1.0).max())
                    )
                fingerprint = (
                    tuple(controlled.train["sample_id"]),
                    tuple(controlled.test["sample_id"]),
                    controlled.columns,
                    tuple(panel.loc[controlled.test_idx, controlled.time_column]),
                )
                if reference is None:
                    reference = fingerprint
                elif fingerprint != reference:
                    raise AssertionError(
                        f"coalition {config_id} changed the {design} split at drift level "
                        f"{spec.drift_level}; the panel is not confound-free"
                    )
    if worst > tolerance:
        raise AssertionError(
            f"control A1 produced weights deviating from one by {worst:.3e}, above the "
            f"tolerance {tolerance:.1e}; the class prior is not exactly constant"
        )
    return {"max_weight_deviation": worst}


# --- Running one level ------------------------------------------------------
@contextmanager
def _cache_dir(path: Path) -> Iterator[Path]:
    """Point the fit cache at a private directory for the duration of the block.

    The synthetic fits are keyed on a panel fingerprint that no real run can
    produce, so they could safely share the main cache. They are kept apart
    anyway: ``results/fit_cache`` holds the fits behind the reported lattice, and
    a reader auditing it should not have to sort several hundred synthetic
    entries out of it.
    """
    original = cache.CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    cache.CACHE_DIR = path
    try:
        yield path
    finally:
        cache.CACHE_DIR = original


def _cell_rows(
    panel: pandas.DataFrame,
    spec: DriftSpec,
    model_name: str,
    seeds: Sequence[int],
) -> tuple[list[dict[str, object]], dict[tuple[str, int], dict[str, numpy.ndarray]]]:
    """Fit every coalition on both arms and return tidy rows plus predictions.

    Returns
    -------
    rows : list of dict
        One row per fit, carrying the columns :func:`sift.shapley.gap_values`
        requires.
    predictions : dict
        Stored predictions of the full coalition only, keyed by
        ``(design, seed)``, for the residual bootstrap.
    """
    base = base_config(spec, model_name)
    digest = panel_fingerprint(panel)
    full_coalition = 2 ** len(CONTROL_NAMES) - 1

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, int], dict[str, numpy.ndarray]] = {}
    for design in LATTICE_DESIGNS:
        for seed in seeds:
            for config_id in range(2 ** len(CONTROL_NAMES)):
                flags = ControlFlags.from_index(config_id)
                cfg = replace(
                    base,
                    split=split_spec_for(base.split, design, spec.cut_year),
                    flags=flags,
                    seed=int(seed),
                )
                record = run_cell(panel, cfg, panel_digest=digest)
                rows.append(
                    {
                        "drift_level": float(spec.drift_level),
                        "config_id": int(config_id),
                        "a1_prior": bool(flags.a1_prior),
                        "a2_labels": bool(flags.a2_labels),
                        "b1_fs": bool(flags.b1_fs),
                        "b2_axis": bool(flags.b2_axis),
                        "design": design,
                        "role": design_role(design),
                        "cut": int(spec.cut_year),
                        "seed": int(seed),
                        "model": model_name,
                        "n_train": int(record["n_train"]),
                        "n_test": int(record["n_test"]),
                        METRIC: float(record[METRIC]),
                        "accuracy": float(record["accuracy"]),
                    }
                )
                if config_id == full_coalition:
                    stored = record["predictions"]
                    predictions[(design, int(seed))] = {
                        "sample_id": numpy.asarray(stored["sample_id"]),
                        "y_true": numpy.asarray(stored["y_true"]),
                        "y_pred": numpy.asarray(stored["y_pred"]),
                    }
    return rows, predictions


def _full_coalition_weights(
    panel: pandas.DataFrame,
    spec: DriftSpec,
    model_name: str,
    design: str,
    seed: int,
    sample_id: numpy.ndarray,
) -> numpy.ndarray | None:
    """Recover the A1 test weights of the full coalition, aligned to ``sample_id``.

    ``run_cell`` stores predictions but not the weight vector the point estimate
    used, and a bootstrap that dropped the weights would describe a different
    estimand from the estimate it accompanies.
    """
    base = base_config(spec, model_name)
    cfg = replace(
        base,
        split=split_spec_for(base.split, design, spec.cut_year),
        flags=ControlFlags.from_index(2 ** len(CONTROL_NAMES) - 1),
        seed=int(seed),
    )
    controlled = apply_controls(panel, cfg)
    if controlled.test_weights is None:
        return None
    lookup = pandas.Series(
        controlled.test_weights, index=controlled.test["sample_id"].to_numpy()
    )
    return lookup.loc[sample_id].to_numpy(dtype=numpy.float64)


def residual_bootstrap(
    panel: pandas.DataFrame,
    spec: DriftSpec,
    predictions: dict[tuple[str, int], dict[str, numpy.ndarray]],
    model_name: str,
    seeds: Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, numpy.ndarray]:
    """Percentile bootstrap interval for the residual ``R = Delta(N)``.

    Both arms of the full coalition are resampled over their own test rows, the
    two resampled scores are subtracted, and the difference is averaged across
    seeds at each bootstrap index so that the interval describes the same
    seed-averaged quantity the point estimate reports. Resampling is over test
    rows, so the interval covers sampling variation inside these windows and not
    variation across hypothetical panels.

    Parameters
    ----------
    panel : pandas.DataFrame
        The synthetic panel.
    spec : DriftSpec
        Specification.
    predictions : dict
        Full-coalition predictions from :func:`_cell_rows`.
    model_name : str
        Estimator key.
    seeds : Sequence of int
        Seeds contributing to the average.
    n_boot : int, default 1000
        Resamples per arm per seed.
    alpha : float, default 0.05
        Two-sided miscoverage.

    Returns
    -------
    low, high : float
        Percentile bounds of ``R``.
    draws : numpy.ndarray
        The ``n_boot`` seed-averaged difference draws.
    """
    temporal_design, reference_design = "temporal", "random_fully_matched"
    per_seed: list[numpy.ndarray] = []
    for seed in seeds:
        arm_draws: dict[str, numpy.ndarray] = {}
        for design in (temporal_design, reference_design):
            stored = predictions[(design, int(seed))]
            weights = _full_coalition_weights(
                panel, spec, model_name, design, int(seed), stored["sample_id"]
            )
            arm_draws[design] = bootstrap_distribution(
                stored["y_true"],
                stored["y_pred"],
                metric=METRIC,
                n_boot=n_boot,
                seed=derive_seed(spec.seed, "driftsim", "boot", design, int(seed)),
                sample_weight=weights,
            )
        per_seed.append(arm_draws[reference_design] - arm_draws[temporal_design])
    draws = numpy.mean(numpy.vstack(per_seed), axis=0)
    low, high = numpy.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high), draws


def run_level(
    spec: DriftSpec,
    model_name: str = DEFAULT_MODEL,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_boot: int = 1000,
    check_nullity: bool = True,
    probabilities: numpy.ndarray | None = None,
) -> dict[str, object]:
    """Run the lattice at one drift level and decompose the gap.

    Parameters
    ----------
    spec : DriftSpec
        Specification, including the drift level.
    model_name : str, default 'logreg'
        Estimator key.
    seeds : Sequence of int
        Base seeds.
    n_boot : int, default 1000
        Bootstrap resamples for the residual interval.
    check_nullity : bool, default True
        Verify on the realised panel that no coalition changes the split.
    probabilities : numpy.ndarray, optional
        Bernoulli parameters. Defaults to the real MLRan marginals.

    Returns
    -------
    dict
        ``rows`` (per-fit records), ``shapley`` (per-control values),
        ``residual``, ``residual_ci``, ``delta``, ``disagreement``,
        ``max_weight_deviation``, ``checks_passed`` and ``seconds``.
    """
    started = time.perf_counter()
    if probabilities is None:
        probabilities = feature_probabilities(spec)
    panel = generate_panel(spec, probabilities)

    nullity = {"max_weight_deviation": float("nan")}
    if check_nullity:
        nullity = assert_controls_are_null(panel, spec, model_name, seeds)

    rows, predictions = _cell_rows(panel, spec, model_name, seeds)
    frame = pandas.DataFrame(rows)
    values = gap_values(frame, model=model_name, cut=spec.cut_year, metric_column=METRIC)
    # decompose is handed Delta(empty) so that residual = Delta(empty) - v(N)
    # reduces to Delta(N), the part the four controls leave unexplained.
    result = decompose(_baseline_gap(frame, model_name, spec.cut_year), values)
    result.raise_if_any_check_failed()

    low, high, _ = residual_bootstrap(
        panel, spec, predictions, model_name, seeds, n_boot=n_boot
    )
    return {
        "rows": rows,
        "shapley": dict(result.shapley),
        "residual": float(result.residual),
        "residual_ci": (low, high),
        "delta": float(result.delta),
        "disagreement": rule_disagreement(spec, probabilities),
        "max_weight_deviation": float(nullity["max_weight_deviation"]),
        "checks_passed": bool(result.all_checks_passed),
        "seconds": time.perf_counter() - started,
    }


def _baseline_gap(frame: pandas.DataFrame, model_name: str, cut: int) -> float:
    """Return ``Delta(empty)``, the gap at the naive coalition."""
    naive = (
        (frame["model"] == model_name) & (frame["cut"] == cut) & (frame["config_id"] == 0)
    )
    block = frame[naive]
    temporal = block[block["design"] == "temporal"][METRIC].mean()
    reference = block[block["design"] == "random_fully_matched"][METRIC].mean()
    return float(reference) - float(temporal)


# --- The ladder -------------------------------------------------------------
def run_ladder(
    levels: Sequence[float] = DEFAULT_LEVELS,
    template: DriftSpec | None = None,
    model_name: str = DEFAULT_MODEL,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_boot: int = 1000,
    results_dir: Path | None = None,
    probabilities: numpy.ndarray | None = None,
) -> dict[str, pandas.DataFrame]:
    """Run every drift level and write the three result tables.

    Parameters
    ----------
    levels : Sequence of float
        Injected drift levels.
    template : DriftSpec, optional
        Specification whose ``drift_level`` is replaced per level.
    model_name : str, default 'logreg'
        Estimator key.
    seeds : Sequence of int
        Base seeds.
    n_boot : int, default 1000
        Bootstrap resamples.
    results_dir : pathlib.Path, optional
        Output directory, defaulting to :data:`DRIFTSIM_DIR`.
    probabilities : numpy.ndarray, optional
        Bernoulli parameters, shared by every level so that the levels differ in the
        label rule alone.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``shapley``, ``residual`` and ``fits``.
    """
    out_dir = DRIFTSIM_DIR if results_dir is None else Path(results_dir)
    base_spec = DriftSpec() if template is None else template
    if probabilities is None:
        probabilities = feature_probabilities(base_spec)

    shapley_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    with _cache_dir(out_dir / "fit_cache"):
        for level in levels:
            spec = replace(base_spec, drift_level=float(level))
            outcome = run_level(
                spec, model_name, seeds, n_boot=n_boot, probabilities=probabilities
            )
            fit_rows.extend(outcome["rows"])
            for control in CONTROL_NAMES:
                shapley_rows.append(
                    {
                        "drift_level": float(level),
                        "theta_radians": spec.theta,
                        "control": control,
                        "phi": float(outcome["shapley"][control]),
                        "model": model_name,
                        "cut_year": int(spec.cut_year),
                        "n_seeds": len(seeds),
                    }
                )
            low, high = outcome["residual_ci"]
            residual_rows.append(
                {
                    "drift_level": float(level),
                    "theta_radians": spec.theta,
                    "rule_disagreement": float(outcome["disagreement"]),
                    "delta_naive": float(outcome["delta"]),
                    "residual": float(outcome["residual"]),
                    "residual_lo": float(low),
                    "residual_hi": float(high),
                    "sum_phi": float(sum(outcome["shapley"].values())),
                    "max_abs_phi": float(max(abs(v) for v in outcome["shapley"].values())),
                    "max_weight_deviation": float(outcome["max_weight_deviation"]),
                    "checks_passed": bool(outcome["checks_passed"]),
                    "model": model_name,
                    "cut_year": int(spec.cut_year),
                    "n_seeds": len(seeds),
                    "seconds": float(outcome["seconds"]),
                }
            )

    tables = {
        "shapley": pandas.DataFrame(shapley_rows),
        "residual": pandas.DataFrame(residual_rows),
        "fits": pandas.DataFrame(fit_rows),
    }
    write_results(tables, out_dir)
    return tables


def write_results(
    tables: dict[str, pandas.DataFrame],
    results_dir: Path | None = None,
) -> dict[str, Path]:
    """Write the ladder tables to parquet.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        Output of :func:`run_ladder`.
    results_dir : pathlib.Path, optional
        Output directory, defaulting to :data:`DRIFTSIM_DIR`.

    Returns
    -------
    dict of str to pathlib.Path
        Where each table landed.
    """
    out_dir = DRIFTSIM_DIR if results_dir is None else Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        written[name] = path
    return written


def main() -> None:
    """Run the default ladder and print the pass/fail table."""
    started = time.perf_counter()
    tables = run_ladder()
    residual = tables["residual"]
    shapley = tables["shapley"]

    print("\ndrift  R        95% CI                sum_phi    max|phi|   flip_rate")
    for row in residual.itertuples():
        print(
            f"{row.drift_level:<6.2f} {row.residual:+.4f}  "
            f"[{row.residual_lo:+.4f}, {row.residual_hi:+.4f}]  "
            f"{row.sum_phi:+.5f}   {row.max_abs_phi:.5f}    {row.rule_disagreement:.3f}"
        )

    residuals = residual["residual"].to_numpy()
    monotone = bool(numpy.all(numpy.diff(residuals) > 0.0))
    zero_row = residual[residual["drift_level"] == 0.0]
    covers_zero = bool(
        (zero_row["residual_lo"] <= 0.0).all() and (zero_row["residual_hi"] >= 0.0).all()
    )
    print(f"\nmonotone in drift level : {monotone}")
    print(f"level 0 interval covers 0: {covers_zero}")
    print(f"largest |phi| anywhere   : {shapley['phi'].abs().max():.6f}")
    print(f"total runtime seconds    : {time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
