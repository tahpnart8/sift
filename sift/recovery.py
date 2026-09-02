"""Recovery and selectivity check for the SIFT decomposition.

The lattice machinery in :mod:`sift.shapley` is verified arithmetically: the
weights sum to one, efficiency holds, and the Moebius cross-check reproduces
every ``phi_i``. None of that establishes *construct validity*. Arithmetic is
silent on whether ``phi_a2`` measures novelty rather than, say, a class-prior
difference that happens to travel with it. This module closes that gap by the
only route available: build panels in which the ground truth is known by
construction, run the same decomposition, and ask whether the attribution lands
on the control that was actually injected.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy
import pandas
import pyarrow
import pyarrow.parquet

from sift import cache
from sift.config import CONTROL_NAMES, ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.controls import match_class_prior
from sift.experiment import (
    LATTICE_DESIGNS,
    LATTICE_ROLE,
    N_COALITIONS,
    label_categories,
    panel_fingerprint,
    run_cell,
)
from sift.features import feature_columns, select_features
from sift.metrics import bootstrap_distribution
from sift.paths import RESULTS_DIR
from sift.seeding import derive_seed, make_rng
from sift.shapley import decompose, exact_shapley, harsanyi_dividends
from sift.splits import make_split, temporal_split

__all__ = [
    "A1_ALIGNMENTS",
    "DECOY_PREFIX",
    "InjectionSpec",
    "PANEL_INJECTED",
    "PANEL_NAMES",
    "RECOVERY_DIR",
    "RecoveryPanel",
    "TIER_GROUPS",
    "ZONES",
    "arm_prior_gap",
    "build_recovery_panel",
    "confound_diagnostics",
    "per_class_dispersion",
    "lattice_metrics",
    "null_panel",
    "recover",
    "recovery_cache",
    "run_recovery",
    "rebuild_test_weights",
    "write_recovery",
]

#: Output directory. Deliberately a sibling of ``results/`` rather than a
#: subdirectory of it, so a recovery run can never be mistaken for, or overwrite,
#: the production lattice.
RECOVERY_DIR: Path = RESULTS_DIR / "recovery"

#: Fit cache for recovery panels. Separate from ``results/fit_cache`` because
#: these panels are synthetic and must never share a directory with the fits the
#: paper reports, even though the panel fingerprint already keys them apart.
RECOVERY_CACHE_DIR: Path = RECOVERY_DIR / "fit_cache"

#: The three temporal zones a row can be dealt into.
ZONES: tuple[str, ...] = ("train", "test", "post")

#: Prefix of the injected feature columns of the ``b1_only`` panel. Chosen so
#: that :func:`sift.features.feature_columns` picks them up as features, since
#: it treats every non-metadata column as one.
DECOY_PREFIX: str = "decoy_"

#: Panel names, in reporting order.
PANEL_NAMES: tuple[str, ...] = (
    "null",
    "a1_only",
    "a2_only",
    "b1_only",
    "b2_only",
    "a2_b2",
)

#: Ground truth: which controls are injected in each panel.
PANEL_INJECTED: dict[str, tuple[str, ...]] = {
    "null": (),
    "a1_only": ("a1_prior",),
    "a2_only": ("a2_labels",),
    "b1_only": ("b1_fs",),
    "b2_only": ("b2_axis",),
    "a2_b2": ("a2_labels", "b2_axis"),
}

#: The two-tier taxonomy, passed through to :func:`sift.shapley.decompose` so the
#: Owen checks run alongside the Shapley ones.
TIER_GROUPS: tuple[tuple[str, ...], ...] = (("a1_prior", "a2_labels"), ("b1_fs", "b2_axis"))

#: How the A1 target distribution chooses which families to boost.
#:
#: ``"name"`` boosts every second family in alphabetical order, an arbitrary
#: pairing with respect to how hard a family is to classify, which is the
#: property the recovery check turned out to depend on. ``"size"`` boosts the
#: smaller half instead, so the window over-represents the families with the
#: least training support. See :func:`_skewed_distribution`.
A1_ALIGNMENTS: tuple[str, ...] = ("name", "size")

#: What ``injected_magnitude`` means for each control.
MAGNITUDE_KIND: dict[str, str] = {
    "a1_prior": "tv_distance_window_vs_corpus",
    "a2_labels": "share_of_test_rows_in_unseen_families",
    "b1_fs": "decoy_columns_over_feature_budget",
    "b2_axis": "share_of_rows_with_disagreeing_axes",
}


@dataclass(frozen=True)
class InjectionSpec:
    """Everything that fixes a recovery run.

    Parameters
    ----------
    cut : int, default 2018
        The single cut point. One cut only; the check is about attribution, not
        about how attribution moves with time.
    test_window : int, default 3
        Width of the test window in years.
    n_train, n_test : int, default 520 and 500
        Target sizes of the training half and the test window. They are set directly
        rather than inherited from the MLRan year histogram because the remaining rows
        form the reservoir that lets a class-composition skew be injected into one half
        without forcing the opposite skew into the other.
    a1_skew : float, default 3.0
        Odds multiplier applied to half the families when building the target
        class distribution of the test window.
    a1_align : {'name', 'size'}, default 'name'
        Which half of the families the multiplier is applied to. Under ``'name'`` it is
        every second family in alphabetical order, which is the original setting and is
        uncorrelated with how hard a family is.
    a2_families : int, default 6
        Number of families dealt entirely into the test window.
    b1_decoys : int, default 100
        Number of decoy feature columns added.
    b1_decoy_levels : int, default 4
        Number of distinct non-zero values a decoy takes inside the test window.
    b1_decoy_noise : float, default 0.1
        Probability that a decoy takes a non-zero value outside the test window.
    b2_skew : float, default 3.0
        Odds multiplier applied to half the families when building the target
        class composition of the secondary-axis training half.
    model_name : str, default 'logreg'
        The one model. Logistic regression with the lbfgs solver returns
        identical predictions for any seed, so the seed axis carries no model
        noise and the bootstrap is the only interval that means anything.
    seeds : tuple of int, default (0, 1, 2)
        Base seeds. They vary the split draws of the reference arm, not the
        model.
    n_features : int, default 200
        Feature budget, held at the production value.
    n_boot : int, default 300
        Bootstrap resamples of the test window per fit.
    alpha : float, default 0.05
        Two-sided miscoverage of the reported interval.
    bootstrap_seed : int, default 20260827
        Seed handed to every fit's resampler, so that two fits over the same
        test window are resampled by the same index draw and their difference is
        paired rather than independent.
    """

    cut: int = 2018
    test_window: int = 3
    n_train: int = 520
    n_test: int = 500
    a1_skew: float = 3.0
    a1_align: str = "name"
    a2_families: int = 6
    b1_decoys: int = 100
    b1_decoy_levels: int = 4
    b1_decoy_noise: float = 0.1
    b2_skew: float = 3.0
    model_name: str = "logreg"
    seeds: tuple[int, ...] = (0, 1, 2)
    n_features: int = 200
    n_boot: int = 300
    alpha: float = 0.05
    bootstrap_seed: int = 20260827

    def __post_init__(self) -> None:
        if self.test_window < 1:
            raise ValueError(f"test_window must be at least 1, got {self.test_window}")
        if self.n_train < 1 or self.n_test < 1:
            raise ValueError("n_train and n_test must both be positive")
        if self.a1_skew <= 1.0 or self.b2_skew <= 1.0:
            raise ValueError("skew multipliers must exceed one to inject anything")
        if self.a1_align not in A1_ALIGNMENTS:
            raise ValueError(
                f"a1_align must be one of {A1_ALIGNMENTS}, got {self.a1_align!r}"
            )
        if self.a2_families < 1:
            raise ValueError(f"a2_families must be at least 1, got {self.a2_families}")
        if self.b1_decoys < 1:
            raise ValueError(f"b1_decoys must be at least 1, got {self.b1_decoys}")
        if self.b1_decoy_levels < 2:
            raise ValueError(
                f"b1_decoy_levels must be at least 2, got {self.b1_decoy_levels}"
            )
        if not 0.0 <= self.b1_decoy_noise < 1.0:
            raise ValueError(
                f"b1_decoy_noise must lie in [0, 1), got {self.b1_decoy_noise}"
            )
        if not self.seeds:
            raise ValueError("at least one seed is required")

    @property
    def last_test_year(self) -> int:
        """Last year inside the test window."""
        return self.cut + self.test_window - 1

    def split_spec(self, design: str) -> SplitSpec:
        """Return the :class:`~sift.config.SplitSpec` for one arm of the game."""
        return SplitSpec(design=design, cut_year=self.cut, test_window=self.test_window)


@dataclass(frozen=True)
class RecoveryPanel:
    """One semi-synthetic panel together with its ground truth.

    Attributes
    ----------
    name : str
        One of :data:`PANEL_NAMES`.
    panel : pandas.DataFrame
        The panel itself, with both year columns re-assigned and, for the
        ``b1_only`` panel, the decoy columns appended.
    injected : tuple of str
        Names of the controls that were actually injected. The ground truth.
    magnitude : dict of {str: float}
        Measured magnitude of all four confounds on this panel, whether injected or not,
        keyed by control name.
    diagnostics : dict of {str: float}
        Panel-level descriptive numbers: half sizes, class counts, and the
        secondary-axis measurements where they differ from the primary ones.
    """

    name: str
    panel: pandas.DataFrame
    injected: tuple[str, ...]
    magnitude: dict[str, float]
    diagnostics: dict[str, float]


# --- Zone dealing -----------------------------------------------------------


def _zone_years(spec: InjectionSpec, years: pandas.Series) -> dict[str, numpy.ndarray]:
    """Return the years belonging to each zone, taken from the panel's own axis.

    Parameters
    ----------
    spec : InjectionSpec
        Cut point and window width.
    years : pandas.Series
        The original primary-axis years, used only for their distinct values.

    Returns
    -------
    dict of {str: numpy.ndarray}
        Sorted year values per zone.

    Raises
    ------
    ValueError
        If any zone would be empty, which would make the deal unrealisable.
    """
    distinct = numpy.sort(pandas.unique(years.to_numpy()))
    zones = {
        "train": distinct[distinct < spec.cut],
        "test": distinct[(distinct >= spec.cut) & (distinct <= spec.last_test_year)],
        "post": distinct[distinct > spec.last_test_year],
    }
    empty = [name for name, values in zones.items() if values.size == 0]
    if empty:
        raise ValueError(
            f"cut {spec.cut} with a {spec.test_window}-year window leaves zone(s) "
            f"{empty} without a single year in the panel's range"
        )
    return zones


def _zone_weights(
    zone_values: Mapping[str, numpy.ndarray], years: pandas.Series
) -> dict[str, numpy.ndarray]:
    """Return within-zone year probabilities taken from the panel's own histogram."""
    counts = years.value_counts()
    weights: dict[str, numpy.ndarray] = {}
    for zone, values in zone_values.items():
        raw = numpy.array([float(counts.get(value, 0.0)) for value in values])
        weights[zone] = raw / raw.sum() if raw.sum() > 0 else numpy.full(values.size, 1.0 / values.size)
    return weights


def _deal_zones(
    panel: pandas.DataFrame,
    target: str,
    test_share: Mapping[Any, float],
    train_share: Mapping[Any, float],
    rng: numpy.random.Generator,
    priority: str = "test",
) -> pandas.Series:
    """Deal every row into one of the three zones, family by family.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel to deal. Never modified.
    target : str
        Label column defining the families.
    test_share, train_share : Mapping
        Per-family fraction of that family's rows to place in the test window and in the
        training half.
    rng : numpy.random.Generator
        Source of the within-family shuffle.
    priority : {'test', 'train'}, default 'test'
        Which half is filled first when the two requested shares exceed one. The
        injections that skew a half's composition set this to that half, so the
        skew is realised exactly and the rounding loss falls on the other.

    Returns
    -------
    pandas.Series
        Zone label per row, aligned to ``panel.index``.
    """
    if priority not in ("test", "train"):
        raise ValueError(f"priority must be 'test' or 'train', got {priority!r}")

    zones = pandas.Series("post", index=panel.index, dtype=object)
    for family, positions in panel.groupby(target, sort=True).indices.items():
        labels = panel.index.to_numpy()[positions]
        labels = labels[rng.permutation(labels.size)]
        size = labels.size
        wanted = {
            "test": int(round(size * float(test_share[family]))),
            "train": int(round(size * float(train_share[family]))),
        }
        first, second = (priority, "train" if priority == "test" else "test")
        n_first = max(0, min(wanted[first], size))
        n_second = max(0, min(wanted[second], size - n_first))
        zones.loc[labels[:n_first]] = first
        zones.loc[labels[n_first : n_first + n_second]] = second
    return zones


def _years_from_zones(
    zones: pandas.Series,
    spec: InjectionSpec,
    template_years: pandas.Series,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Turn zone labels into concrete years drawn from the panel's own histogram.

    Only the zone matters to :func:`sift.splits.temporal_split`; the year within
    a zone is cosmetic and is drawn so the resulting histogram still resembles
    MLRan's rather than being flat.
    """
    values = _zone_years(spec, template_years)
    weights = _zone_weights(values, template_years)
    out = numpy.empty(len(zones), dtype=numpy.int64)
    zone_array = zones.to_numpy()
    for zone in ZONES:
        mask = zone_array == zone
        n = int(mask.sum())
        if n:
            out[mask] = rng.choice(values[zone], size=n, p=weights[zone])
    return out


def _uniform_shares(
    families: Sequence[Any], value: float
) -> dict[Any, float]:
    """Return the same share for every family."""
    return {family: float(value) for family in families}


def _skewed_distribution(
    prior: pandas.Series, skew: float, align: str = "name"
) -> pandas.Series:
    """Return a class distribution that departs from ``prior`` by a fixed odds ratio.

    Half the families have their mass multiplied by ``skew`` and the result is
    renormalised. Which half is deterministic rather than random, so that two
    runs of the same specification target the same distribution and the injected
    magnitude is reproducible.

    Parameters
    ----------
    prior : pandas.Series
        Corpus class distribution, summing to one.
    skew : float
        Odds multiplier applied to the boosted half.
    align : {'name', 'size'}, default 'name'
        ``'name'`` boosts every second family in alphabetical order. ``'size'``
        boosts the smaller half of the families.

    Returns
    -------
    pandas.Series
        Target distribution over the same index, summing to one.

    Notes
    -----
    The two alignments can produce the same total-variation distance from
    ``prior`` while differing entirely in what a class-prior control can do about
    them, and that distinction is the substance of this parameter rather than a
    detail of it.
    """
    if align not in A1_ALIGNMENTS:
        raise ValueError(f"align must be one of {A1_ALIGNMENTS}, got {align!r}")
    ordered = prior.sort_index()
    if align == "name":
        boosted = numpy.arange(ordered.size) % 2 == 0
    else:
        # Rank by mass, which is rank by family size since the prior is the size
        # histogram normalised. Ties break by name through the stable sort, so
        # the boosted set is a function of the panel and not of the sort.
        rank = numpy.argsort(numpy.argsort(ordered.to_numpy(), kind="stable"), kind="stable")
        boosted = rank < ordered.size // 2
    multiplier = numpy.where(boosted, skew, 1.0)
    raw = ordered.to_numpy() * multiplier
    return pandas.Series(raw / raw.sum(), index=ordered.index)


def _shares_for_distribution(
    distribution: pandas.Series,
    sizes: pandas.Series,
    n_rows: int,
    cap: Mapping[Any, float] | None = None,
) -> dict[Any, float]:
    """Convert a target class distribution into a per-family row share.

    Family ``f`` must supply ``n_rows * distribution[f]`` rows, which is a share
    of ``n_rows * distribution[f] / sizes[f]`` of that family.
    """
    shares: dict[Any, float] = {}
    for family in sizes.index:
        wanted = float(n_rows) * float(distribution[family]) / float(sizes[family])
        ceiling = 1.0 if cap is None else float(cap[family])
        shares[family] = float(min(max(wanted, 0.0), ceiling))
    return shares


# --- Panel construction -----------------------------------------------------


def _blank_panel(panel: pandas.DataFrame) -> pandas.DataFrame:
    """Return a detached copy ready to have its temporal axes re-written.

    ``attrs`` is cleared for the reason :func:`sift.data.build_panel` clears it:
    pandas propagates it across ``copy``, so anything the caller attached to the
    real panel would ride into a synthetic one and be read as a property of it.
    """
    out = panel.copy()
    out.attrs.clear()
    return out


def _apply_axis(
    frame: pandas.DataFrame,
    column: str,
    zones: pandas.Series,
    spec: InjectionSpec,
    template_years: pandas.Series,
    rng: numpy.random.Generator,
) -> None:
    """Write one temporal axis in place from a zone assignment."""
    frame[column] = _years_from_zones(zones, spec, template_years, rng)


def null_panel(
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
    seed: int = 0,
) -> pandas.DataFrame:
    """Return the panel with its temporal signal destroyed.

    Both axes are re-dealt from the same zone assignment, so the two agree
    everywhere and control B2 has nothing to correct. Within every family the
    deal is a uniform random shuffle over the three zones in fixed proportions,
    so the year columns are independent of the label, of the features, and of one
    another.

    Parameters
    ----------
    panel : pandas.DataFrame
        The real MLRan analysis panel. Never modified.
    panel_spec : sift.config.PanelSpec
        Supplies the target column and the two axis names.
    spec : InjectionSpec
        Cut point, window width and target half sizes.
    seed : int, default 0
        Seed for the deal.

    Returns
    -------
    pandas.DataFrame
        A new panel. Only the two year columns differ from the input.
    """
    target = panel_spec.target_column
    rng = make_rng(derive_seed(seed, "recovery", "null"))
    total = len(panel)
    families = pandas.Index(sorted(panel[target].unique()))

    zones = _deal_zones(
        panel,
        target,
        _uniform_shares(families, spec.n_test / total),
        _uniform_shares(families, spec.n_train / total),
        rng,
    )
    out = _blank_panel(panel)
    template = panel[panel_spec.primary_axis]
    _apply_axis(out, panel_spec.primary_axis, zones, spec, template, rng)
    out[panel_spec.secondary_axis] = out[panel_spec.primary_axis].to_numpy()
    return out


def _deal_a1(
    panel: pandas.DataFrame, panel_spec: PanelSpec, spec: InjectionSpec, rng
) -> pandas.Series:
    """Zone deal for the A1 injection: skewed window, corpus-prior training half."""
    target = panel_spec.target_column
    total = len(panel)
    sizes = panel[target].value_counts().sort_index()
    prior = sizes / float(total)
    skewed = _skewed_distribution(prior, spec.a1_skew, spec.a1_align)

    train_share = _uniform_shares(sizes.index, spec.n_train / total)
    cap = {family: 1.0 - train_share[family] for family in sizes.index}
    test_share = _shares_for_distribution(skewed, sizes, spec.n_test, cap)
    return _deal_zones(panel, target, test_share, train_share, rng, priority="test")


def _deal_a2(
    panel: pandas.DataFrame, panel_spec: PanelSpec, spec: InjectionSpec, rng
) -> tuple[pandas.Series, tuple[str, ...]]:
    """Zone deal for the A2 injection: ``k`` families held out of the training side."""
    target = panel_spec.target_column
    total = len(panel)
    sizes = panel[target].value_counts().sort_index()
    ordered = sizes.sort_values(kind="stable")
    novel = tuple(str(name) for name in ordered.index[: spec.a2_families])
    novel_rows = int(sizes.loc[list(novel)].sum())

    remaining = total - novel_rows
    if remaining <= 0 or novel_rows >= spec.n_test:
        raise ValueError(
            f"the {spec.a2_families} held-out families supply {novel_rows} rows, which "
            f"does not fit inside a test window of {spec.n_test}"
        )
    test_rest = (spec.n_test - novel_rows) / remaining
    train_rest = spec.n_train / remaining
    if test_rest + train_rest > 1.0:
        raise ValueError(
            "holding out "
            + str(spec.a2_families)
            + " families leaves too few rows to fill both halves"
        )

    test_share = {
        family: 1.0 if family in novel else test_rest for family in sizes.index
    }
    train_share = {
        family: 0.0 if family in novel else train_rest for family in sizes.index
    }
    zones = _deal_zones(panel, target, test_share, train_share, rng, priority="test")
    return zones, novel


def _deal_b2(
    panel: pandas.DataFrame, panel_spec: PanelSpec, spec: InjectionSpec, rng
) -> pandas.Series:
    """Zone deal for the B2 injection: skewed training half, corpus-prior window.

    The mirror image of :func:`_deal_a1`. The test window keeps the corpus class
    prior, so control A1 has nothing to correct, while the training half is
    dealt towards a skewed composition: some families arrive with a third of the
    training support a random split would have given them and some with three
    times as much. Every family keeps a presence in the training half, so control
    A2 has nothing to correct either.
    """
    target = panel_spec.target_column
    total = len(panel)
    sizes = panel[target].value_counts().sort_index()
    prior = sizes / float(total)
    skewed = _skewed_distribution(prior, spec.b2_skew)

    test_share = _uniform_shares(sizes.index, spec.n_test / total)
    cap = {family: 1.0 - test_share[family] for family in sizes.index}
    train_share = _shares_for_distribution(skewed, sizes, spec.n_train, cap)
    return _deal_zones(panel, target, test_share, train_share, rng, priority="train")


def _add_decoys(
    frame: pandas.DataFrame,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
    rng: numpy.random.Generator,
) -> pandas.DataFrame:
    """Return the B1 decoy columns as a frame aligned to ``frame.index``.

    A decoy is a small-integer column that, on every row inside the test window,
    is a fixed random many-to-one map of the class, and off the window is zero
    apart from a low rate of uniform noise. Fitted on the whole panel the
    mutual-information selector sees one of the strongest features present;
    fitted on the training half alone it sees noise. The model, trained on
    pre-cut rows, can never use one, so every budget slot a decoy takes is a slot
    the temporal arm loses.
    """
    target = panel_spec.target_column
    years = frame[panel_spec.primary_axis].to_numpy()
    window = (years >= spec.cut) & (years <= spec.last_test_year)
    codes = pandas.factorize(frame[target], sort=True)[0]
    n_classes = int(codes.max()) + 1
    n_rows = len(frame)
    levels = spec.b1_decoy_levels

    columns: dict[str, numpy.ndarray] = {}
    for index in range(spec.b1_decoys):
        assignment = rng.integers(1, levels + 1, size=n_classes)
        column = numpy.where(
            rng.random(n_rows) < spec.b1_decoy_noise,
            rng.integers(1, levels + 1, size=n_rows),
            0,
        )
        column[window] = assignment[codes[window]]
        columns[f"{DECOY_PREFIX}{index:03d}"] = column.astype(numpy.int8)
    return pandas.DataFrame(columns, index=frame.index)


def build_recovery_panel(
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    name: str,
    spec: InjectionSpec,
    seed: int = 0,
) -> RecoveryPanel:
    """Build one recovery panel and measure what was actually injected into it.

    Parameters
    ----------
    panel : pandas.DataFrame
        The real MLRan analysis panel. Never modified.
    panel_spec : sift.config.PanelSpec
        Panel specification; the target column and both axis names are read.
    name : str
        One of :data:`PANEL_NAMES`.
    spec : InjectionSpec
        Injection magnitudes and cut point.
    seed : int, default 0
        Seed for the deal. Fixed across panels so that two panels differ by their
        injection and not by their shuffle.

    Returns
    -------
    RecoveryPanel
        The panel, its ground truth, and the measured magnitude of all four
        confounds.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised panel.
    """
    if name not in PANEL_NAMES:
        raise ValueError(f"unknown recovery panel {name!r}; expected one of {PANEL_NAMES}")

    target = panel_spec.target_column
    template = panel[panel_spec.primary_axis]
    rng = make_rng(derive_seed(seed, "recovery", name))
    out = _blank_panel(panel)
    extra: dict[str, float] = {}

    if name in ("null", "b1_only"):
        primary = _deal_zones(
            panel,
            target,
            _uniform_shares(sorted(panel[target].unique()), spec.n_test / len(panel)),
            _uniform_shares(sorted(panel[target].unique()), spec.n_train / len(panel)),
            rng,
        )
        secondary = primary
    elif name == "a1_only":
        primary = _deal_a1(panel, panel_spec, spec, rng)
        secondary = primary
    elif name == "a2_only":
        primary, novel = _deal_a2(panel, panel_spec, spec, rng)
        secondary = primary
        extra["n_held_out_families"] = float(len(novel))
    elif name == "b2_only":
        primary = _deal_zones(
            panel,
            target,
            _uniform_shares(sorted(panel[target].unique()), spec.n_test / len(panel)),
            _uniform_shares(sorted(panel[target].unique()), spec.n_train / len(panel)),
            rng,
        )
        secondary = _deal_b2(panel, panel_spec, spec, rng)
    else:  # a2_b2
        primary, novel = _deal_a2(panel, panel_spec, spec, rng)
        secondary = _deal_b2(panel, panel_spec, spec, rng)
        extra["n_held_out_families"] = float(len(novel))

    _apply_axis(out, panel_spec.primary_axis, primary, spec, template, rng)
    if secondary is primary:
        out[panel_spec.secondary_axis] = out[panel_spec.primary_axis].to_numpy()
    else:
        _apply_axis(out, panel_spec.secondary_axis, secondary, spec, template, rng)

    if name == "b1_only":
        out = pandas.concat([out, _add_decoys(out, panel_spec, spec, rng)], axis=1)

    magnitude, diagnostics = confound_diagnostics(out, panel_spec, spec)
    diagnostics.update(extra)
    return RecoveryPanel(
        name=name,
        panel=out,
        injected=PANEL_INJECTED[name],
        magnitude=magnitude,
        diagnostics=diagnostics,
    )


# --- Measuring what is in a panel -------------------------------------------


def _halves(
    panel: pandas.DataFrame, spec: InjectionSpec, column: str
) -> tuple[pandas.Index, pandas.Index]:
    """Return the temporal halves this panel would produce on one axis."""
    return temporal_split(panel, spec.split_spec("temporal"), column)


def _total_variation(left: pandas.Series, right: pandas.Series) -> float:
    """Total-variation distance between two class distributions."""
    index = left.index.union(right.index)
    a = left.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    b = right.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    return float(0.5 * numpy.abs(a - b).sum())


def confound_diagnostics(
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure all four confounds on a realised panel.

    This is what makes the ground truth a measurement rather than an assumption.
    Each confound is read off the panel by the same definition the corresponding
    control acts on, and every panel is measured for all four, so the three that
    were not injected are visible as the construction's own selectivity evidence.

    Parameters
    ----------
    panel : pandas.DataFrame
        A recovery panel.
    panel_spec : sift.config.PanelSpec
        Target column and axis names.
    spec : InjectionSpec
        Cut point and window width.

    Returns
    -------
    magnitude : dict of {str: float}
        One entry per control name in :data:`sift.config.CONTROL_NAMES`.
    diagnostics : dict of {str: float}
        Half sizes, class counts and the secondary-axis readings.
    """
    target = panel_spec.target_column
    primary, secondary = panel_spec.primary_axis, panel_spec.secondary_axis

    train_idx, test_idx = _halves(panel, spec, primary)
    corpus = panel[target].value_counts(normalize=True)
    window = panel.loc[test_idx, target].value_counts(normalize=True)
    seen = set(panel.loc[train_idx, target].unique())
    unseen_rows = int((~panel.loc[test_idx, target].isin(seen)).sum())

    decoys = [c for c in panel.columns if str(c).startswith(DECOY_PREFIX)]

    train_2, test_2 = _halves(panel, spec, secondary)
    zone_primary = _zone_of(panel, spec, primary)
    zone_secondary = _zone_of(panel, spec, secondary)

    magnitude = {
        "a1_prior": _total_variation(window, corpus),
        "a2_labels": unseen_rows / max(len(test_idx), 1),
        "b1_fs": len(decoys) / float(spec.n_features),
        "b2_axis": float((zone_primary != zone_secondary).mean()),
    }
    diagnostics = {
        "n_train": float(len(train_idx)),
        "n_test": float(len(test_idx)),
        "k_train": float(panel.loc[train_idx, target].nunique()),
        "k_test": float(panel.loc[test_idx, target].nunique()),
        "n_decoys": float(len(decoys)),
        "train_prior_tv_secondary": _total_variation(
            panel.loc[train_2, target].value_counts(normalize=True), corpus
        ),
        "train_prior_tv_primary": _total_variation(
            panel.loc[train_idx, target].value_counts(normalize=True), corpus
        ),
        "window_prior_tv_secondary": _total_variation(
            panel.loc[test_2, target].value_counts(normalize=True), corpus
        ),
        "unseen_share_secondary": float(
            (
                ~panel.loc[test_2, target].isin(set(panel.loc[train_2, target].unique()))
            ).sum()
        )
        / max(len(test_2), 1),
    }
    return magnitude, diagnostics


def _zone_of(
    panel: pandas.DataFrame, spec: InjectionSpec, column: str
) -> numpy.ndarray:
    """Return the zone label of every row on one axis."""
    years = panel[column].to_numpy()
    out = numpy.full(years.shape, "post", dtype=object)
    out[years < spec.cut] = "train"
    out[(years >= spec.cut) & (years <= spec.last_test_year)] = "test"
    return out


def selected_decoy_share(
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
    seed: int = 0,
) -> dict[str, float]:
    """Return the share of the feature budget the decoys win under each B1 setting.

    Diagnostic for the ``b1_only`` panel. It answers the question the injection
    rests on: does whole-panel selection actually buy the decoys, and does
    training-half selection actually reject them.

    Returns
    -------
    dict of {str: float}
        ``leaky`` is the decoy share of the budget when the selector is fitted on
        the whole panel, ``honest`` when it is fitted on the training half.
    """
    target = panel_spec.target_column
    columns = feature_columns(panel)
    train_idx, _ = _halves(panel, spec, panel_spec.primary_axis)
    selection_seed = derive_seed(seed, "features", spec.cut)

    leaky = select_features(panel, columns, target, spec.n_features, selection_seed)
    honest = select_features(
        panel.loc[train_idx], columns, target, spec.n_features, selection_seed
    )
    return {
        "leaky": sum(str(c).startswith(DECOY_PREFIX) for c in leaky) / float(len(leaky)),
        "honest": sum(str(c).startswith(DECOY_PREFIX) for c in honest) / float(len(honest)),
    }


def arm_prior_gap(
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
    seed: int | None = None,
) -> dict[str, float]:
    """Measure the class-prior distance between the two arms of the game.

    ``confound_diagnostics`` reports A1's magnitude as the distance between the
    temporal window and the corpus, which is the definition the control acts on.
    It is not, however, the quantity the *game* sees. ``Delta(S)`` is a
    difference between two arms, so A1 can only contribute to it through a
    disagreement between the temporal window's class proportions and those the
    reference arm actually draws. The reference arm is ``random_fully_matched``,
    which matches the temporal window's class *set* and size; if it also
    inherited its class *proportions*, A1 would be correcting both arms by the
    same amount and the contrast would cancel.

    Parameters
    ----------
    panel : pandas.DataFrame
        A recovery panel.
    panel_spec : sift.config.PanelSpec
        Target column and axis names.
    spec : InjectionSpec
        Cut point, window width and seed list.
    seed : int, optional
        Single seed to measure. Omitted, every seed in ``spec.seeds`` is measured
        and the mean is returned; the per-seed spread is reported alongside.

    Returns
    -------
    dict of {str: float}
        ``tv_window_vs_corpus``, ``tv_reference_vs_corpus``,
        ``tv_arm_to_arm`` and ``tv_arm_to_arm_sd``.
    """
    target = panel_spec.target_column
    temporal_design, reference_design = LATTICE_DESIGNS
    _, test_idx = _halves(panel, spec, panel_spec.primary_axis)
    corpus = panel[target].value_counts(normalize=True)
    window = panel.loc[test_idx, target].value_counts(normalize=True)

    seeds = spec.seeds if seed is None else (int(seed),)
    arm_to_arm: list[float] = []
    reference_to_corpus: list[float] = []
    split = spec.split_spec(reference_design)
    for one in seeds:
        split_seed = derive_seed(one, "split", reference_design, spec.cut)
        _, reference_idx = make_split(
            panel,
            split,
            panel_spec.primary_axis,
            split_seed,
            stratify_labels=panel[target],
            label_column=target,
            restrict_labels=False,
        )
        drawn = panel.loc[reference_idx, target].value_counts(normalize=True)
        arm_to_arm.append(_total_variation(window, drawn))
        reference_to_corpus.append(_total_variation(drawn, corpus))
    return {
        "tv_window_vs_corpus": _total_variation(window, corpus),
        "tv_reference_vs_corpus": float(numpy.mean(reference_to_corpus)),
        "tv_arm_to_arm": float(numpy.mean(arm_to_arm)),
        "tv_arm_to_arm_sd": float(numpy.std(arm_to_arm)),
    }


def per_class_dispersion(
    y_true: numpy.ndarray, y_pred: numpy.ndarray
) -> dict[str, float]:
    """Measure how far per-class performance varies within one fit.

    Reweighting classes can only move a class-averaged metric to the extent that
    the metric differs between classes, so a decomposition run on a panel whose
    classes are all equally learnable would report ``phi_a1 = 0`` for a reason
    that has nothing to do with the estimator. This makes that possibility
    checkable rather than assumed.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Stored predictions of one fit.

    Returns
    -------
    dict of {str: float}
        Standard deviation, minimum, maximum and mean of per-class F1 and of
        per-class recall, over the classes present in ``y_true``, plus
        ``k_present``.
    """
    from sklearn.metrics import f1_score, recall_score

    present = numpy.unique(numpy.asarray(y_true))
    f1 = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    out: dict[str, float] = {"k_present": float(present.size)}
    for name, values in (("f1", f1), ("recall", recall)):
        out[f"{name}_sd"] = float(numpy.std(values))
        out[f"{name}_min"] = float(numpy.min(values))
        out[f"{name}_max"] = float(numpy.max(values))
        out[f"{name}_mean"] = float(numpy.mean(values))
    return out


# --- Running the lattice on one panel ---------------------------------------


@contextlib.contextmanager
def recovery_cache(directory: Path = RECOVERY_CACHE_DIR) -> Iterator[Path]:
    """Point :mod:`sift.cache` at a recovery-only directory for the duration.

    The fit key already carries the panel fingerprint, so a recovery fit could
    not collide with a production one even in a shared directory. Keeping them
    apart is about the directory listing, not about correctness: nobody
    inspecting ``results/fit_cache`` should have to work out which of its entries
    belong to a synthetic panel.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    previous = cache.CACHE_DIR
    cache.CACHE_DIR = directory
    try:
        yield directory
    finally:
        cache.CACHE_DIR = previous


def _config(
    panel_spec: PanelSpec, spec: InjectionSpec, design: str, config_id: int, seed: int
) -> ExperimentConfig:
    return ExperimentConfig(
        panel=panel_spec,
        split=spec.split_spec(design),
        flags=ControlFlags.from_index(config_id),
        model_name=spec.model_name,
        seed=int(seed),
        target=panel_spec.target_column,
        n_features=spec.n_features,
    )


def lattice_metrics(
    recovery_panel: RecoveryPanel,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
) -> tuple[pandas.DataFrame, dict[tuple[str, int, int], dict[str, numpy.ndarray]]]:
    """Run all sixteen coalitions on both arms, for every seed.

    Parameters
    ----------
    recovery_panel : RecoveryPanel
        The panel to evaluate.
    panel_spec : sift.config.PanelSpec
        Panel specification handed to every cell.
    spec : InjectionSpec
        Model, seeds, cut point and feature budget.

    Returns
    -------
    metrics : pandas.DataFrame
        One row per fit, carrying the columns
        :func:`sift.shapley.gap_values` requires.
    predictions : dict
        Stored predictions keyed by ``(design, config_id, seed)``, so the
        bootstrap can resample the same test windows the point estimates used.
    """
    panel = recovery_panel.panel
    digest = panel_fingerprint(panel)
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int, int], dict[str, numpy.ndarray]] = {}

    for design in LATTICE_DESIGNS:
        for config_id in range(N_COALITIONS):
            for seed in spec.seeds:
                cfg = _config(panel_spec, spec, design, config_id, seed)
                record = run_cell(panel, cfg, panel_digest=digest)
                predictions[(design, config_id, int(seed))] = record["predictions"]
                rows.append(
                    {
                        "panel": recovery_panel.name,
                        "design": design,
                        "role": LATTICE_ROLE,
                        "config_id": int(config_id),
                        "model": spec.model_name,
                        "cut": int(spec.cut),
                        "seed": int(seed),
                        "n_train": int(record["n_train"]),
                        "n_test": int(record["n_test"]),
                        "k_test": int(record["k_test"]),
                        "macro_f1": float(record["macro_f1"]),
                        "accuracy": float(record["accuracy"]),
                        **{name: bool(record[name]) for name in CONTROL_NAMES},
                    }
                )
    return pandas.DataFrame(rows), predictions


# --- Bootstrap --------------------------------------------------------------


def rebuild_test_weights(
    y_true: numpy.ndarray,
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    a1_enabled: bool,
) -> numpy.ndarray | None:
    """Rebuild control A1's test weights from stored predictions.

    ``run_cell`` stores predictions but not the weight vector that scored them.
    Bootstrapping an A1-enabled cell without those weights would resample a
    different estimand from the one the point estimate reports, so the vector is
    rebuilt here from the encoded labels and the corpus prior, which is exactly
    the pair :func:`sift.controls.match_class_prior` consumes.

    Parameters
    ----------
    y_true : numpy.ndarray
        Encoded true labels of one fit, positions in
        :func:`sift.experiment.label_categories`.
    panel : pandas.DataFrame
        The panel the fit ran on, supplying the reference distribution.
    panel_spec : sift.config.PanelSpec
        Target column and task.
    a1_enabled : bool
        Whether control A1 was on.

    Returns
    -------
    numpy.ndarray or None
        Weights averaging to one, or ``None`` when A1 was off.
    """
    if not a1_enabled:
        return None
    target = panel_spec.target_column
    categories = label_categories(panel, target)
    labels = pandas.Series(categories[numpy.asarray(y_true)], name=target)
    frame = pandas.DataFrame({target: labels})
    reference = panel[target].value_counts(normalize=True)
    _, weights = match_class_prior(frame, frame, target, True, panel_spec, reference)
    return weights


def _bootstrap_draws(
    predictions: Mapping[tuple[str, int, int], Mapping[str, numpy.ndarray]],
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
    metric: str = "macro_f1",
) -> dict[tuple[str, int, int], numpy.ndarray]:
    """Resample every fit's test window ``n_boot`` times.

    All fits are handed the same resampler seed, so two cells over an identical
    test window draw identical indices and their difference is paired. Cells over
    different windows cannot be paired at all; nothing here pretends otherwise.
    """
    draws: dict[tuple[str, int, int], numpy.ndarray] = {}
    for key, stored in predictions.items():
        _, config_id, _ = key
        flags = ControlFlags.from_index(config_id)
        weights = rebuild_test_weights(
            stored["y_true"], panel, panel_spec, flags.a1_prior
        )
        draws[key] = bootstrap_distribution(
            stored["y_true"],
            stored["y_pred"],
            metric,
            spec.n_boot,
            spec.bootstrap_seed,
            sample_weight=weights,
        )
    return draws


def _values_from_scores(
    scores: Mapping[tuple[str, int], float],
) -> tuple[dict[frozenset[str], float], float]:
    """Turn a ``(design, config_id) -> metric`` map into ``v(S)`` and ``Delta(empty)``.

    Mirrors :func:`sift.shapley.gap_values` exactly, but takes plain floats so a
    bootstrap replicate does not have to be materialised as a DataFrame first.
    """
    temporal, reference = LATTICE_DESIGNS
    gaps: dict[frozenset[str], float] = {}
    for config_id in range(N_COALITIONS):
        coalition = ControlFlags.from_index(config_id).active()
        gaps[coalition] = scores[(reference, config_id)] - scores[(temporal, config_id)]
    baseline = gaps[frozenset()]
    return {key: baseline - gap for key, gap in gaps.items()}, baseline


# --- Recovery ---------------------------------------------------------------


def recover(
    recovery_panel: RecoveryPanel,
    panel_spec: PanelSpec,
    spec: InjectionSpec,
    metrics: pandas.DataFrame,
    predictions: Mapping[tuple[str, int, int], Mapping[str, numpy.ndarray]],
    metric: str = "macro_f1",
) -> dict[str, Any]:
    """Decompose one recovery panel and attach bootstrap intervals.

    Parameters
    ----------
    recovery_panel : RecoveryPanel
        The panel, carrying the ground truth.
    panel_spec : sift.config.PanelSpec
        Panel specification.
    spec : InjectionSpec
        Bootstrap settings.
    metrics : pandas.DataFrame
        Output of :func:`lattice_metrics`.
    predictions : Mapping
        Stored predictions from the same call.
    metric : str, default 'macro_f1'
        Metric the gap is measured in.

    Returns
    -------
    dict
        ``phi``, ``phi_ci``, ``dividends``, ``dividend_ci``, ``delta_empty``,
        ``v_full``, ``residual`` and ``checks_passed``.
    """
    point: dict[tuple[str, int], float] = {}
    for (design, config_id), block in metrics.groupby(["design", "config_id"], sort=True):
        point[(str(design), int(config_id))] = float(block[metric].mean())

    values, delta_empty = _values_from_scores(point)
    decomposition = decompose(
        delta_empty, values, groups=TIER_GROUPS, group_names=("data", "protocol")
    )

    draws = _bootstrap_draws(predictions, recovery_panel.panel, panel_spec, spec, metric)
    n_seeds = len(spec.seeds)
    phi_draws: dict[str, list[float]] = {name: [] for name in CONTROL_NAMES}
    dividend_draws: dict[frozenset[str], list[float]] = {}
    v_full_draws: list[float] = []

    for replicate in range(spec.n_boot):
        scores: dict[tuple[str, int], float] = {}
        for design in LATTICE_DESIGNS:
            for config_id in range(N_COALITIONS):
                total = 0.0
                for seed in spec.seeds:
                    total += float(draws[(design, config_id, int(seed))][replicate])
                scores[(design, config_id)] = total / n_seeds
        replicate_values, _ = _values_from_scores(scores)
        for name, value in exact_shapley(replicate_values).items():
            phi_draws[name].append(value)
        for coalition, value in harsanyi_dividends(replicate_values).items():
            dividend_draws.setdefault(coalition, []).append(value)
        v_full_draws.append(replicate_values[frozenset(CONTROL_NAMES)])

    low, high = 100.0 * spec.alpha / 2.0, 100.0 * (1.0 - spec.alpha / 2.0)

    def interval(sample: Sequence[float]) -> tuple[float, float]:
        bounds = numpy.percentile(numpy.asarray(sample, dtype=float), [low, high])
        return float(bounds[0]), float(bounds[1])

    return {
        "panel": recovery_panel.name,
        "phi": decomposition.shapley,
        "phi_ci": {name: interval(values) for name, values in phi_draws.items()},
        "dividends": decomposition.dividends,
        "dividend_ci": {
            coalition: interval(sample) for coalition, sample in dividend_draws.items()
        },
        "delta_empty": float(decomposition.delta),
        "v_full": float(values[frozenset(CONTROL_NAMES)]),
        "v_full_ci": interval(v_full_draws),
        "residual": float(decomposition.residual),
        "checks_passed": bool(decomposition.all_checks_passed),
        "n_checks": len(decomposition.checks),
    }


# --- Orchestration ----------------------------------------------------------


def run_recovery(
    panel: pandas.DataFrame,
    panel_spec: PanelSpec,
    spec: InjectionSpec | None = None,
    panel_names: Sequence[str] = PANEL_NAMES,
    seed: int = 0,
    metric: str = "macro_f1",
    progress: bool = False,
) -> dict[str, pandas.DataFrame]:
    """Build every recovery panel, decompose it, and tabulate the result.

    Parameters
    ----------
    panel : pandas.DataFrame
        The real MLRan analysis panel.
    panel_spec : sift.config.PanelSpec
        Panel specification.
    spec : InjectionSpec, optional
        Injection magnitudes; the defaults are the ones the report uses.
    panel_names : sequence of str, optional
        Panels to run, defaulting to all of :data:`PANEL_NAMES`.
    seed : int, default 0
        Seed for the deals.
    metric : str, default 'macro_f1'
        Metric the gap is measured in.
    progress : bool, default False
        Print one line per panel as it finishes. Off by default so the function
        stays usable from a test.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        ``recovery`` with one row per (panel, control), ``dividends`` with one
        row per (panel, coalition), and ``metrics`` with every fit.
    """
    spec = spec or InjectionSpec()
    recovery_rows: list[dict[str, Any]] = []
    dividend_rows: list[dict[str, Any]] = []
    metric_frames: list[pandas.DataFrame] = []

    with recovery_cache():
        for name in panel_names:
            started = time.perf_counter()
            built = build_recovery_panel(panel, panel_spec, name, spec, seed)
            frame, predictions = lattice_metrics(built, panel_spec, spec)
            result = recover(built, panel_spec, spec, frame, predictions, metric)
            elapsed = time.perf_counter() - started

            metric_frames.append(frame)
            for control in CONTROL_NAMES:
                lo, hi = result["phi_ci"][control]
                phi = float(result["phi"][control])
                recovery_rows.append(
                    {
                        "panel": name,
                        "control": control,
                        "injected": control in built.injected,
                        "injected_magnitude": float(built.magnitude[control]),
                        "magnitude_kind": MAGNITUDE_KIND[control],
                        "phi": phi,
                        "phi_lo": lo,
                        "phi_hi": hi,
                        "covers_zero": bool(lo <= 0.0 <= hi),
                        "phi_share_of_v_full": (
                            phi / result["v_full"] if result["v_full"] != 0.0 else float("nan")
                        ),
                        "delta_empty": result["delta_empty"],
                        "v_full": result["v_full"],
                        "v_full_lo": result["v_full_ci"][0],
                        "v_full_hi": result["v_full_ci"][1],
                        "residual": result["residual"],
                        "checks_passed": result["checks_passed"],
                        "n_train": float(built.diagnostics["n_train"]),
                        "n_test": float(built.diagnostics["n_test"]),
                        "k_test": float(built.diagnostics["k_test"]),
                        "model": spec.model_name,
                        "cut": int(spec.cut),
                        "n_seeds": len(spec.seeds),
                        "n_boot": int(spec.n_boot),
                        "seconds": float(elapsed),
                    }
                )
            for coalition, value in result["dividends"].items():
                lo, hi = result["dividend_ci"][coalition]
                dividend_rows.append(
                    {
                        "panel": name,
                        "coalition": "+".join(sorted(coalition)) or "empty",
                        "size": len(coalition),
                        "dividend": float(value),
                        "dividend_lo": lo,
                        "dividend_hi": hi,
                        "covers_zero": bool(lo <= 0.0 <= hi),
                    }
                )
            if progress:
                print(
                    f"{name}: v(N)={result['v_full']:+.4f} "
                    f"Delta0={result['delta_empty']:+.4f} "
                    f"R={result['residual']:+.4f} "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

    return {
        "recovery": pandas.DataFrame(recovery_rows),
        "dividends": pandas.DataFrame(dividend_rows),
        "metrics": pandas.concat(metric_frames, ignore_index=True),
    }


# --- Output -----------------------------------------------------------------

COMPRESSION: str = "zstd"
COMPRESSION_LEVEL: int = 3


def write_recovery(
    frames: Mapping[str, pandas.DataFrame], directory: Path = RECOVERY_DIR
) -> dict[str, Path]:
    """Write the recovery tables as parquet.

    Parameters
    ----------
    frames : Mapping of {str: pandas.DataFrame}
        Output of :func:`run_recovery`.
    directory : pathlib.Path, optional
        Destination, defaulting to :data:`RECOVERY_DIR`.

    Returns
    -------
    dict of {str: pathlib.Path}
        Written paths, keyed as the input.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame in frames.items():
        path = directory / f"{name}.parquet"
        table = pyarrow.Table.from_pandas(frame, preserve_index=False)
        pyarrow.parquet.write_table(
            table, path, compression=COMPRESSION, compression_level=COMPRESSION_LEVEL
        )
        written[name] = path
    return written
