"""The four evaluation controls and the order in which they are applied.

The controls do not commute, so the Shapley value function ``v(S)`` is only well
defined once an application order is fixed. That order is part of the method and
must be stated in the paper.

The rationale for :data:`CONTROL_ORDER`: ``b2_axis`` decides which sample falls
in which half, so it runs first; ``a2_labels`` then filters samples; ``a1_prior``
assigns weights without changing sample composition; ``b1_fs`` selects features
on the training half, which by then is settled.

Each control function takes an ``enabled`` flag and is a no-op when it is
``False``. The 16 coalitions are therefore total: every subset of the four
players is a valid, runnable configuration.

An opt-in fifth control, C1 deduplication, is defined at the foot of this
module. It does not touch anything above it: :data:`CONTROL_ORDER` and
:func:`apply_controls` stay four-player and stay the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
import pandas

from sift.config import (
    C1_NAME,
    ExperimentConfig,
    ExtendedControlFlags,
    PanelSpec,
    config_for_cell_c1,
)
from sift.data import PanelVariants
from sift.features import feature_columns, select_features
from sift.seeding import derive_seed
from sift.splits import make_split

__all__ = [
    "CONTROL_ORDER",
    "ControlledData",
    "resolve_time_axis",
    "restrict_label_space",
    "match_class_prior",
    "resolve_feature_set",
    "apply_controls",
    # Opt-in five-control extension; see the C1 section at the foot of the file.
    "CONTROL_ORDER_C1",
    "apply_controls_c1",
]

#: Mandatory application order of the four controls.
CONTROL_ORDER: tuple[str, ...] = ("b2_axis", "a2_labels", "a1_prior", "b1_fs")


@dataclass(frozen=True)
class ControlledData:
    """A train/test split with all four controls applied.

    Attributes
    ----------
    train, test : pandas.DataFrame
        The two halves after the controls that filter samples.
    columns : tuple of str
        Feature columns after selection. Identical for both halves.
    time_column : str
        Temporal axis the split was drawn on, as resolved by control B2.
    test_weights : numpy.ndarray or None
        Sample weights from control A1, or ``None`` when A1 is disabled. Weights
        average to one, so they rescale the class prior without changing the
        effective sample size. These are applied when scoring the test window.
    train_weights : numpy.ndarray or None
        Computed but deliberately never used at fit time. A1 is an intervention
        on the measurement, not on the learner: the question it answers is what
        the reported score would have been had the test window carried the
        reference class prior, which is a property of how the score is computed.
        Reweighting the training set as well would additionally change what the
        model learned, so the resulting number would no longer isolate the prior
        shift from the model's response to it. The field is retained because it
        makes the symmetry of the two windows visible and because
        :mod:`sift.experiment` requires the attribute to be present; do not wire
        it into ``fit``.
    train_idx, test_idx : pandas.Index
        Panel index labels of the two halves, retained so predictions can be
        joined back to the panel.
    """

    train: pandas.DataFrame
    test: pandas.DataFrame
    columns: tuple[str, ...]
    time_column: str
    train_weights: numpy.ndarray | None
    test_weights: numpy.ndarray | None
    train_idx: pandas.Index
    test_idx: pandas.Index


def resolve_time_axis(spec: PanelSpec, enabled: bool) -> str:
    """Resolve the temporal axis selected by control B2.

    Parameters
    ----------
    spec : sift.config.PanelSpec
        Panel specification carrying both axis names.
    enabled : bool
        ``False`` selects the compile timestamp, the erroneous axis that a naive
        evaluation would reach for. ``True`` selects the first-submission date.

    Returns
    -------
    str
        Name of the year column to split on.

    Notes
    -----
    With the control disabled the two halves differ in sample composition and not
    merely in ordering, because a large group of samples carries a compile
    timestamp pushed back before the year 2000. This must be stated when the
    result is presented.
    """
    return spec.primary_axis if enabled else spec.secondary_axis


def restrict_label_space(
    train: pandas.DataFrame,
    test: pandas.DataFrame,
    label_column: str,
    enabled: bool,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Apply control A2, restricting the test label space to families seen in training.

    Parameters
    ----------
    train, test : pandas.DataFrame
        The two halves of the split.
    label_column : str
        Column defining the label space, normally ``'ransomware_family'``. The
        family column is used even on the binary task, where the meaningful
        restriction is still to drop ransomware of an unseen family.
    enabled : bool
        When ``False`` the halves are returned unchanged and the test window
        keeps families never seen in training.

    Returns
    -------
    train, test : pandas.DataFrame
        The training half is never altered; only the test half can shrink.
    """
    if not enabled:
        return train, test
    seen = set(train[label_column].unique())
    return train, test[test[label_column].isin(seen)]


def match_class_prior(
    train: pandas.DataFrame,
    test: pandas.DataFrame,
    target: str,
    enabled: bool,
    spec: PanelSpec,
    reference: pandas.Series | None = None,
) -> tuple[numpy.ndarray | None, numpy.ndarray | None]:
    """Apply control A1, weighting samples towards a reference class prior.

    Weighting is used rather than discarding samples. Reaching the reference rate
    by discarding would remove the majority of ransomware from the later windows,
    and the measured difference would then reflect the lost samples rather than
    the class prior.

    Parameters
    ----------
    train, test : pandas.DataFrame
        The two halves of the split.
    target : str
        Label column.
    enabled : bool
        When ``False`` both weight vectors are ``None`` and each window keeps its
        natural class distribution.
    spec : sift.config.PanelSpec
        Panel specification; ``prior_reference_rate`` is read for the binary task.
    reference : pandas.Series, optional
        Reference class distribution, indexed by class label and summing to one.
        Required for the multi-class task, where no scalar rate is meaningful.

    Returns
    -------
    train_weights, test_weights : numpy.ndarray or None
        Weights averaging to one, or ``None`` when the control is disabled.

    Raises
    ------
    ValueError
        If the multi-class task is requested without a reference distribution.
    """
    if not enabled:
        return None, None

    if spec.task == "binary":
        rate = spec.prior_reference_rate
        reference = pandas.Series({0: 1.0 - rate, 1: rate})
    elif reference is None:
        raise ValueError("a reference class distribution is required for the family task")

    return (
        _prior_weights(train[target], reference),
        _prior_weights(test[target], reference),
    )


def _prior_weights(labels: pandas.Series, reference: pandas.Series) -> numpy.ndarray:
    """Return weights reshaping an empirical class prior into a reference prior.

    Parameters
    ----------
    labels : pandas.Series
        Observed labels of one window.
    reference : pandas.Series
        Target class distribution, indexed by class label.

    Returns
    -------
    numpy.ndarray
        Weights aligned to ``labels``, normalised to average one so the effective
        sample size is preserved.
    """
    empirical = labels.value_counts(normalize=True)
    ratio = labels.map(lambda value: reference.get(value, 0.0) / empirical[value])
    weights = ratio.to_numpy(dtype=numpy.float64)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("prior weighting produced an all-zero weight vector")
    return weights * (len(weights) / total)


def resolve_feature_set(
    panel: pandas.DataFrame,
    train: pandas.DataFrame,
    columns: list[str],
    target: str,
    n_features: int,
    enabled: bool,
    seed: int,
) -> list[str]:
    """Apply control B1, deciding which rows the feature selector is fitted on.

    Both settings retain the same number of features, so the control varies only
    the rows the selector sees and never the width of the design matrix.

    Parameters
    ----------
    panel : pandas.DataFrame
        Full analysis panel, used when the control is disabled.
    train : pandas.DataFrame
        Training half, used when the control is enabled.
    columns : list of str
        Candidate feature columns.
    target : str
        Label column.
    n_features : int
        Number of features to retain.
    enabled : bool
        When ``False`` selection is fitted on the whole panel and therefore leaks
        test information. When ``True`` it is fitted on the training half alone.
    seed : int
        Seed for the mutual-information estimator.

    Returns
    -------
    list of str
        Selected feature columns.

    Notes
    -----
    The published feature space has already been reduced from over six million
    raw features to 483 by the dataset authors, using their own split. Selection
    here can only be re-run within those 483 columns, so the contribution
    attributed to B1 is a lower bound on the true effect.
    """
    source = train if enabled else panel
    return select_features(source, columns, target, n_features, seed)


def apply_controls(panel: pandas.DataFrame, cfg: ExperimentConfig) -> ControlledData:
    """Draw the split and apply all four controls in :data:`CONTROL_ORDER`.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel from :func:`sift.data.build_panel`.
    cfg : sift.config.ExperimentConfig
        Configuration for this lattice cell.

    Returns
    -------
    ControlledData
        The controlled split, ready to be fitted.

    Raises
    ------
    ValueError
        If either half is empty, or the training half holds a single class. The
        failure is raised rather than skipped: exact Shapley needs all 16 cells,
        so a missing cell must be loud.
    """
    columns = feature_columns(panel)
    label_column = "ransomware_family" if "ransomware_family" in panel.columns else cfg.target

    time_column = resolve_time_axis(cfg.panel, cfg.flags.b2_axis)

    split_seed = derive_seed(cfg.seed, "split", cfg.split.design, cfg.split.cut_year or -1)
    train_idx, test_idx = make_split(
        panel,
        cfg.split,
        time_column,
        split_seed,
        stratify_labels=panel[cfg.target],
        label_column=label_column,
        restrict_labels=cfg.flags.a2_labels,
    )
    train, test = panel.loc[train_idx], panel.loc[test_idx]

    # A no-op for the matched design, which applied the same filter while drawing
    # its test half in order to hit the temporal size exactly.
    train, test = restrict_label_space(train, test, label_column, cfg.flags.a2_labels)

    if train.empty or test.empty:
        raise ValueError(
            f"configuration {cfg.flags.to_index()} at cut {cfg.split.cut_year} left "
            f"{len(train)} training and {len(test)} test samples"
        )
    if train[cfg.target].nunique() < 2:
        raise ValueError(
            f"configuration {cfg.flags.to_index()} at cut {cfg.split.cut_year} left a "
            "single class in the training half"
        )

    reference = panel[cfg.target].value_counts(normalize=True)
    train_weights, test_weights = match_class_prior(
        train, test, cfg.target, cfg.flags.a1_prior, cfg.panel, reference
    )

    selection_seed = derive_seed(cfg.seed, "features", cfg.split.cut_year or -1)
    selected = resolve_feature_set(
        panel, train, columns, cfg.target, cfg.n_features, cfg.flags.b1_fs, selection_seed
    )

    return ControlledData(
        train=train,
        test=test,
        columns=tuple(selected),
        time_column=time_column,
        train_weights=train_weights,
        test_weights=test_weights,
        train_idx=train.index,
        test_idx=test.index,
    )


# ===========================================================================
# C1: deduplication as an opt-in fifth control
# ===========================================================================

#: Mandatory application order of the five controls.
#:
#: ``c1_dedup`` comes first, and the reason is not stylistic. The other four
#: controls rearrange or reweight a fixed set of samples: B2 decides which
#: window a sample falls in, A2 filters the test window, A1 assigns weights, B1
#: chooses the rows the selector sees. C1 decides *which samples exist at all*.
#: A group of identical feature vectors spread across several years is either
#: one sample or five, and until that is settled B2 cannot decide which window
#: they fall in, A2 cannot know which families the training half contains, A1
#: cannot count the class prior, and B1 cannot fit a selector. Every one of the
#: four therefore reads a quantity that C1 defines, so C1 precedes all of them.
#:
#: The order is pinned by ``tests/test_c1_control.py`` and is realised
#: structurally rather than by convention: C1 is applied by
#: :func:`sift.data.build_panel_variants`, upstream of the panel that
#: :func:`apply_controls` receives, so no code path can apply it later.
CONTROL_ORDER_C1: tuple[str, ...] = (C1_NAME,) + CONTROL_ORDER


def apply_controls_c1(
    panels: PanelVariants,
    base: ExperimentConfig,
    flags: ExtendedControlFlags,
) -> ControlledData:
    """Draw the split and apply all five controls in :data:`CONTROL_ORDER_C1`.

    Parameters
    ----------
    panels : sift.data.PanelVariants
        Both C1 variants of the analysis panel, from
        :func:`sift.data.build_panel_variants`.
    base : sift.config.ExperimentConfig
        Template configuration supplying split, model, seed, target and feature
        budget. Its ``flags`` and its ``panel.dedup_exact`` are overridden.
    flags : sift.config.ExtendedControlFlags
        The extended coalition for this cell.

    Returns
    -------
    ControlledData
        Exactly as :func:`apply_controls`, which does the work.

    Raises
    ------
    ValueError
        Propagated from :func:`apply_controls`.

    Notes
    -----
    C1 is applied by selecting the panel, not by a step inside this function.
    Everything downstream is the unmodified four-control path, so a cell with
    ``c1_dedup=True`` produces byte-for-byte what the four-control lattice
    produces for the same coalition.
    """
    return apply_controls(
        panels.select(flags.c1_dedup),
        config_for_cell_c1(base, flags),
    )
