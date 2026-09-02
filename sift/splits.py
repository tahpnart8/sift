"""Split construction for the three designs, with consistency assertions.

Four designs are defined, forming a chain in which each consecutive pair differs
in exactly one respect:
"""

from __future__ import annotations

import warnings

import pandas
from sklearn.model_selection import train_test_split

from sift.config import SplitSpec
from sift.seeding import derive_seed

__all__ = [
    "random_split",
    "temporal_split",
    "random_matched_split",
    "matched_sizes",
    "temporal_test_classes",
    "make_split",
    "assert_temporal_consistency",
    "assert_disjoint",
]


def random_split(
    panel: pandas.DataFrame,
    spec: SplitSpec,
    seed: int,
    stratify_labels: pandas.Series | None = None,
) -> tuple[pandas.Index, pandas.Index]:
    """Draw the naive stratified random reference split.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    spec : sift.config.SplitSpec
        Split specification; ``test_size`` and ``stratify`` are read.
    seed : int
        Seed for the draw, normally derived through
        :func:`sift.seeding.derive_seed`.
    stratify_labels : pandas.Series, optional
        Labels to stratify on. Stratification is skipped when omitted, or when
        some class holds fewer than two samples.

    Returns
    -------
    train_idx, test_idx : pandas.Index
        Index labels of the two halves.
    """
    stratify = None
    if spec.stratify and stratify_labels is not None:
        counts = stratify_labels.value_counts()
        if counts.min() >= 2:
            stratify = stratify_labels
        else:
            warnings.warn(
                f"stratification disabled: {int((counts < 2).sum())} class(es) hold "
                "fewer than two samples",
                RuntimeWarning,
                stacklevel=2,
            )
    train_idx, test_idx = train_test_split(
        panel.index,
        test_size=spec.test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return pandas.Index(train_idx), pandas.Index(test_idx)


def temporal_split(
    panel: pandas.DataFrame,
    spec: SplitSpec,
    time_column: str,
) -> tuple[pandas.Index, pandas.Index]:
    """Draw the temporal split for one cut point.

    Training uses everything strictly before the cut year; testing uses the cut
    year and the following ``test_window - 1`` years. The training window is
    expanding rather than sliding, which matches how a deployed classifier
    accumulates history.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    spec : sift.config.SplitSpec
        Split specification; ``cut_year`` and ``test_window`` are read.
    time_column : str
        Year column to split on. Which column this is is precisely what control
        B2 varies.

    Returns
    -------
    train_idx, test_idx : pandas.Index
        Index labels of the two halves.

    Raises
    ------
    ValueError
        If ``time_column`` is absent from the panel.
    """
    if time_column not in panel.columns:
        raise ValueError(f"time column {time_column!r} absent from the panel")
    years = panel[time_column]
    last_test_year = spec.cut_year + spec.test_window - 1
    train_idx = panel.index[years < spec.cut_year]
    test_idx = panel.index[years.between(spec.cut_year, last_test_year)]
    return pandas.Index(train_idx), pandas.Index(test_idx)


def matched_sizes(
    panel: pandas.DataFrame,
    spec: SplitSpec,
    time_column: str,
    label_column: str | None = None,
    restrict_labels: bool = False,
) -> tuple[int, int]:
    """Return the half sizes of the temporal split at the same cut.

    These are the sizes the ``random_matched`` design reproduces. They are read
    from the data rather than configured, because the purpose of that design is
    to inherit exactly what the temporal split happened to produce.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    spec : sift.config.SplitSpec
        Split specification; ``cut_year`` and ``test_window`` are read.
    time_column : str
        Year column, as resolved by control B2.
    label_column : str, optional
        Column defining the label space for the seen-classes filter.
    restrict_labels : bool, default False
        Whether control A2 is enabled.

    Returns
    -------
    n_train, n_test : int
        Sizes of the temporal split at this cut.
    """
    train_idx, test_idx = temporal_split(panel, spec, time_column)
    if restrict_labels and label_column is not None:
        seen = set(panel.loc[train_idx, label_column].unique())
        test_idx = test_idx[panel.loc[test_idx, label_column].isin(seen).to_numpy()]
    return len(train_idx), len(test_idx)


def temporal_test_classes(
    panel: pandas.DataFrame,
    spec: SplitSpec,
    time_column: str,
    label_column: str,
    restrict_labels: bool = False,
) -> frozenset:
    """Return the classes actually present in the temporal test window.

    This is the pool the ``random_fully_matched`` design draws its test half
    from, so that its macro average is taken over the same number of classes as
    the temporal design's.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    spec : sift.config.SplitSpec
        Split specification; ``cut_year`` and ``test_window`` are read.
    time_column : str
        Year column, as resolved by control B2.
    label_column : str
        Column defining the label space.
    restrict_labels : bool, default False
        Whether control A2 is enabled. When it is, families unseen in training
        have already been removed from the temporal window and must not
        reappear here.

    Returns
    -------
    frozenset
        Class labels present in the temporal test window.
    """
    train_idx, test_idx = temporal_split(panel, spec, time_column)
    if restrict_labels:
        seen = set(panel.loc[train_idx, label_column].unique())
        test_idx = test_idx[panel.loc[test_idx, label_column].isin(seen).to_numpy()]
    return frozenset(panel.loc[test_idx, label_column].unique())


def _draw_exactly(
    candidates: pandas.Index,
    n_draw: int,
    seed: int,
    stratify_labels: pandas.Series | None,
    role: str,
) -> tuple[pandas.Index, pandas.Index]:
    """Draw exactly ``n_draw`` labels from ``candidates``, returning drawn and rest.

    Stratification is attempted first and abandoned with a warning when the
    subset cannot support it, which happens when a class holds a single sample or
    when either side of the draw is smaller than the number of classes.

    Parameters
    ----------
    candidates : pandas.Index
        Index labels available to draw from.
    n_draw : int
        Exact number to draw.
    seed : int
        Seed for the draw.
    stratify_labels : pandas.Series or None
        Labels aligned to ``candidates``, or ``None`` to draw uniformly.
    role : str
        Name of the half being drawn, used in error and warning messages.

    Returns
    -------
    drawn, rest : pandas.Index
        The drawn labels and everything not drawn.

    Raises
    ------
    ValueError
        If ``candidates`` cannot supply ``n_draw`` labels.
    """
    if n_draw <= 0:
        raise ValueError(f"the temporal split left no {role} samples to match")
    if len(candidates) < n_draw:
        raise ValueError(
            f"cannot draw a matched {role} half of {n_draw} samples: only "
            f"{len(candidates)} candidates remain in the panel"
        )
    if len(candidates) == n_draw:
        return pandas.Index(candidates), pandas.Index([])

    stratify = None
    if stratify_labels is not None:
        counts = stratify_labels.value_counts()
        n_classes = int(counts.size)
        feasible = (
            int(counts.min()) >= 2
            and n_draw >= n_classes
            and len(candidates) - n_draw >= n_classes
        )
        if feasible:
            stratify = stratify_labels
        else:
            warnings.warn(
                f"stratification disabled for the matched {role} half: drawing "
                f"{n_draw} of {len(candidates)} samples over {n_classes} classes",
                RuntimeWarning,
                stacklevel=3,
            )
    drawn, rest = train_test_split(
        candidates,
        train_size=n_draw,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return pandas.Index(drawn), pandas.Index(rest)


def random_matched_split(
    panel: pandas.DataFrame,
    spec: SplitSpec,
    time_column: str,
    seed: int,
    label_column: str | None = None,
    restrict_labels: bool = False,
    stratify_labels: pandas.Series | None = None,
) -> tuple[pandas.Index, pandas.Index]:
    """Draw a random split holding exactly the temporal split's half sizes.

    Time order is then the only difference remaining between this design and the
    temporal one, so their gap isolates the effect of ordering from the effect of
    training-set size.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    spec : sift.config.SplitSpec
        Split specification; ``cut_year`` and ``test_window`` fix the sizes.
    time_column : str
        Year column, as resolved by control B2. Used only to read the sizes; the
        draw itself ignores time.
    seed : int
        Seed for the draw. Each stage derives its own seed from it, so the split
        is reproducible from the run seed alone.
    label_column : str, optional
        Column defining the label space for the seen-classes filter.
    restrict_labels : bool, default False
        Whether control A2 is enabled.
    stratify_labels : pandas.Series, optional
        Labels to stratify both draws on.

    Returns
    -------
    train_idx, test_idx : pandas.Index
        Index labels of the two halves, of exactly the temporal sizes, and under
        ``'random_fully_matched'`` over exactly the temporal class set.

    Raises
    ------
    ValueError
        If the panel cannot supply the requested sizes. The failure is raised
        rather than absorbed by a smaller draw: a silently shrunk matched design
        would no longer be matched, and the number it produced would be
        misleading rather than merely imprecise.
    """
    n_train, n_test = matched_sizes(panel, spec, time_column, label_column, restrict_labels)
    if n_train + n_test > len(panel):
        raise ValueError(
            f"the temporal split at cut {spec.cut_year} needs {n_train} training and "
            f"{n_test} test samples, but the panel holds only {len(panel)}"
        )

    labels = stratify_labels if spec.stratify else None

    # The test half is drawn first because it is the constrained one. Drawing the
    # training half first would consume the scarce families that the fully matched
    # design needs, and at the later cuts it leaves too few of them behind to fill
    # a test half of the temporal size.
    pool = panel.index
    if spec.design == "random_fully_matched":
        if label_column is None:
            raise ValueError("random_fully_matched requires a label_column")
        wanted = temporal_test_classes(panel, spec, time_column, label_column, restrict_labels)
        pool = pool[panel.loc[pool, label_column].isin(wanted).to_numpy()]
        if len(pool) < n_test:
            raise ValueError(
                f"the {len(wanted)} families present in the temporal test window at cut "
                f"{spec.cut_year} supply only {len(pool)} samples in total, fewer than "
                f"the matched test half of {n_test}"
            )

    test_idx, _ = _draw_exactly(
        pool,
        n_test,
        derive_seed(seed, "matched", "test"),
        None if labels is None else labels.loc[pool],
        "test",
    )

    remainder = panel.index.difference(test_idx, sort=False)
    train_idx, _ = _draw_exactly(
        remainder,
        n_train,
        derive_seed(seed, "matched", "train"),
        None if labels is None else labels.loc[remainder],
        "training",
    )

    if restrict_labels and label_column is not None:
        unseen = set(panel.loc[test_idx, label_column].unique()) - set(
            panel.loc[train_idx, label_column].unique()
        )
        if unseen:
            raise ValueError(
                f"control A2 requires every test family to appear in training, but the "
                f"matched draw at cut {spec.cut_year} left {len(unseen)} unseen: "
                f"{sorted(unseen)[:5]}"
            )

    return train_idx, test_idx


def make_split(
    panel: pandas.DataFrame,
    spec: SplitSpec,
    time_column: str,
    seed: int,
    stratify_labels: pandas.Series | None = None,
    label_column: str | None = None,
    restrict_labels: bool = False,
) -> tuple[pandas.Index, pandas.Index]:
    """Dispatch to the split implementation named by ``spec.design``.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    spec : sift.config.SplitSpec
        Split specification.
    time_column : str
        Year column for the temporal design, and for reading the sizes of the matched
        design.
    seed : int
        Seed for the two random designs. Ignored by the temporal design, which is
        deterministic.
    stratify_labels : pandas.Series, optional
        Labels to stratify the random designs on.
    label_column : str, optional
        Column defining the label space, used by the matched designs to reproduce the
        seen-classes filter at draw time.
    restrict_labels : bool, default False
        Whether control A2 is enabled. Read by the matched designs only; for the
        temporal and naive random designs A2 is applied after the split.

    Returns
    -------
    train_idx, test_idx : pandas.Index
        Index labels of the two halves.
    """
    if spec.design == "temporal":
        return temporal_split(panel, spec, time_column)
    if spec.design in SplitSpec.MATCHED_DESIGNS:
        return random_matched_split(
            panel, spec, time_column, seed, label_column, restrict_labels, stratify_labels
        )
    return random_split(panel, spec, seed, stratify_labels)


def assert_disjoint(
    panel: pandas.DataFrame,
    train_idx: pandas.Index,
    test_idx: pandas.Index,
) -> None:
    """Assert that no sample appears in both halves of a split.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    train_idx, test_idx : pandas.Index
        The two halves.

    Raises
    ------
    AssertionError
        If any ``sample_id`` occurs in both halves.
    """
    shared = set(panel.loc[train_idx, "sample_id"]) & set(panel.loc[test_idx, "sample_id"])
    if shared:
        raise AssertionError(
            f"{len(shared)} sample_id(s) appear in both halves of the split, "
            f"for example {sorted(shared)[:5]}"
        )


def assert_temporal_consistency(
    panel: pandas.DataFrame,
    train_idx: pandas.Index,
    test_idx: pandas.Index,
    time_column: str,
) -> None:
    """Assert that no training sample is dated at or after any test sample.

    Applies to the temporal design only. Both random designs violate this by
    construction, which is the bias the experiment is built to quantify.

    Parameters
    ----------
    panel : pandas.DataFrame
        Analysis panel.
    train_idx, test_idx : pandas.Index
        The two halves.
    time_column : str
        Year column the split was drawn on.

    Raises
    ------
    AssertionError
        If the halves overlap in time or share a sample.
    """
    assert_disjoint(panel, train_idx, test_idx)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    latest_train = panel.loc[train_idx, time_column].max()
    earliest_test = panel.loc[test_idx, time_column].min()
    if latest_train >= earliest_test:
        raise AssertionError(
            f"temporal inconsistency on {time_column!r}: latest training year "
            f"{latest_train} is not before earliest test year {earliest_test}"
        )
