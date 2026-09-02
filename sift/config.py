"""Frozen configuration objects for the SIFT ablation lattice.

This module holds dataclasses only. It imports nothing from :mod:`sift`, which
keeps the dependency graph acyclic and lets any other module accept a
configuration object without risking a circular import.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, ClassVar

__all__ = [
    "CONTROL_NAMES",
    "ControlFlags",
    "LatticeCell",
    "LATTICE_CELLS",
    "lattice_cell",
    "PanelSpec",
    "SplitSpec",
    "ExperimentConfig",
    "SCHEMA_VERSION",
    # Opt-in five-control extension. None of these names is reachable from the
    # four-control path; see the C1 section at the foot of this module.
    "C1_NAME",
    "CONTROL_NAMES_C1",
    "TIER_GROUPS_C1",
    "ExtendedControlFlags",
    "LATTICE_CELLS_C1",
    "lattice_cell_c1",
    "panel_spec_for_c1",
    "config_for_cell_c1",
    "extended_flags_of",
]

SCHEMA_VERSION: str = "1.0"

#: The four players of the ablation lattice. Preprocessing steps such as
#: ``drop_empty``, ``dedup_exact`` and ``min_class_size`` are deliberately absent:
#: they are fixed preprocessing applied identically to all 16 configurations and
#: are never lattice players.
CONTROL_NAMES: tuple[str, ...] = ("a1_prior", "a2_labels", "b1_fs", "b2_axis")


@dataclass(frozen=True)
class ControlFlags:
    """State of the four evaluation controls for one lattice cell.

    Each flag is a lattice player. ``False`` reproduces the naive evaluation
    practice that SIFT measures; ``True`` applies the correction.

    Parameters
    ----------
    a1_prior : bool, default False
        When ``False`` the class distribution of each window is left as observed.
    a2_labels : bool, default False
        When ``False`` the test window keeps families never seen in training.
        When ``True`` the label space is restricted to families present in the
        training half of the same cut.
    b1_fs : bool, default False
        When ``False`` feature selection is fitted on the whole panel, which leaks test
        information.
    b2_axis : bool, default False
        When ``False`` the split uses the compile timestamp, the erroneous axis.
    """

    a1_prior: bool = False
    a2_labels: bool = False
    b1_fs: bool = False
    b2_axis: bool = False

    @classmethod
    def from_index(cls, index: int) -> "ControlFlags":
        """Build the flags for a lattice cell index.

        Bit ``i`` of ``index`` corresponds to ``CONTROL_NAMES[i]``, so index 0 is
        all controls off and index 15 is all controls on.

        Parameters
        ----------
        index : int
            Lattice cell index in ``range(16)``.

        Returns
        -------
        ControlFlags
            The flag combination addressed by ``index``.

        Raises
        ------
        ValueError
            If ``index`` falls outside ``range(16)``.
        """
        limit = 2 ** len(CONTROL_NAMES)
        if not 0 <= index < limit:
            raise ValueError(f"index must lie in range(0, {limit}), got {index}")
        bits = {name: bool(index >> pos & 1) for pos, name in enumerate(CONTROL_NAMES)}
        return cls(**bits)

    def to_index(self) -> int:
        """Return the lattice cell index of this flag combination.

        Returns
        -------
        int
            Index in ``range(16)``, the inverse of :meth:`from_index`.
        """
        return sum(int(getattr(self, name)) << pos for pos, name in enumerate(CONTROL_NAMES))

    def active(self) -> frozenset[str]:
        """Return the names of the enabled controls.

        Returns
        -------
        frozenset of str
            The coalition ``S`` used as the argument of the Shapley value
            function.
        """
        return frozenset(name for name in CONTROL_NAMES if getattr(self, name))


@dataclass(frozen=True)
class LatticeCell:
    """One of the sixteen configurations, with the reading it must be given.

    The sixteen cells are written
    out rather than left implicit in a bit pattern, because several combinations
    are ambiguous until someone states what they mean. The clearest example is
    cell 4, feature selection done correctly on a time axis that is wrong.

    Attributes
    ----------
    config_id : int
        Lattice index in ``range(16)``, the ``config_id`` column of the results.
    flags : ControlFlags
        The coalition this cell applies.
    description : str
        What the cell means, in one line.
    """

    config_id: int
    flags: ControlFlags
    description: str

    @property
    def active(self) -> frozenset[str]:
        """Return the names of the controls enabled in this cell."""
        return self.flags.active()


#: Every cell of the ablation lattice, indexed by ``config_id``.
#:
#: Bit ``i`` of the index is ``CONTROL_NAMES[i]``, so cell 0 enables nothing and
#: cell 15 enables everything. Cell 0 is the naive practice the study measures;
#: cell 15 is the fully corrected evaluation.
LATTICE_CELLS: tuple[LatticeCell, ...] = (
    LatticeCell(
        0,
        ControlFlags(),
        "Naive practice, and the baseline of the value function: features "
        "selected on the whole panel, split on the compile timestamp, unseen "
        "families left in the test window, class prior as observed.",
    ),
    LatticeCell(
        1,
        ControlFlags(a1_prior=True),
        "Prior shift alone: the test window is reweighted to the reference class "
        "prior at metric time, while selection, axis and label space stay naive.",
    ),
    LatticeCell(
        2,
        ControlFlags(a2_labels=True),
        "Novelty alone: the test window keeps only families seen in training, "
        "while selection remains leaky and the axis remains the compile stamp.",
    ),
    LatticeCell(
        3,
        ControlFlags(a1_prior=True, a2_labels=True),
        "Both data-side controls, neither protocol-side control: the data "
        "artefacts are removed but the protocol is still wrong in both respects.",
    ),
    LatticeCell(
        4,
        ControlFlags(b1_fs=True),
        "Correct feature selection on a wrong time axis: the selector sees only "
        "the training half, but that half was cut on the compile timestamp, so "
        "the leak is closed with respect to an ordering that never held.",
    ),
    LatticeCell(
        5,
        ControlFlags(a1_prior=True, b1_fs=True),
        "Selection leak and prior shift both removed, on the wrong time axis and "
        "with unseen families still scored.",
    ),
    LatticeCell(
        6,
        ControlFlags(a2_labels=True, b1_fs=True),
        "Selection leak and novelty both removed, still on the compile timestamp.",
    ),
    LatticeCell(
        7,
        ControlFlags(a1_prior=True, a2_labels=True, b1_fs=True),
        "Everything corrected except the temporal axis: isolates what the choice "
        "of time field alone is worth.",
    ),
    LatticeCell(
        8,
        ControlFlags(b2_axis=True),
        "Correct temporal axis on its own: the split follows first submission, "
        "but the selector still reads the whole panel including the future.",
    ),
    LatticeCell(
        9,
        ControlFlags(a1_prior=True, b2_axis=True),
        "Correct axis with the prior matched, while selection still leaks and "
        "unseen families are still scored.",
    ),
    LatticeCell(
        10,
        ControlFlags(a2_labels=True, b2_axis=True),
        "Correct axis with the label space restricted, while selection leaks.",
    ),
    LatticeCell(
        11,
        ControlFlags(a1_prior=True, a2_labels=True, b2_axis=True),
        "Everything corrected except the feature-selection leak: isolates what "
        "the leak alone is worth on an otherwise sound protocol.",
    ),
    LatticeCell(
        12,
        ControlFlags(b1_fs=True, b2_axis=True),
        "Both protocol-side controls, neither data-side control: the evaluation "
        "protocol is sound but the window keeps its natural prior and its "
        "unseen families.",
    ),
    LatticeCell(
        13,
        ControlFlags(a1_prior=True, b1_fs=True, b2_axis=True),
        "Everything corrected except novelty: isolates what scoring families the "
        "model has never seen is worth.",
    ),
    LatticeCell(
        14,
        ControlFlags(a2_labels=True, b1_fs=True, b2_axis=True),
        "Everything corrected except the prior: isolates what the class "
        "distribution of the window alone is worth.",
    ),
    LatticeCell(
        15,
        ControlFlags(a1_prior=True, a2_labels=True, b1_fs=True, b2_axis=True),
        "Fully corrected evaluation. The gap that remains here is the residual "
        "the lattice cannot explain.",
    ),
)


def lattice_cell(config_id: int) -> LatticeCell:
    """Return the description of one lattice cell.

    Parameters
    ----------
    config_id : int
        Lattice index in ``range(16)``.

    Returns
    -------
    LatticeCell
        The cell, including its one-line reading.

    Raises
    ------
    ValueError
        If ``config_id`` falls outside ``range(16)``.
    """
    if not 0 <= config_id < len(LATTICE_CELLS):
        raise ValueError(
            f"config_id must lie in range(0, {len(LATTICE_CELLS)}), got {config_id}"
        )
    return LATTICE_CELLS[config_id]


@dataclass(frozen=True)
class PanelSpec:
    """Fixed preprocessing that produces the analysis panel.

    These steps are applied identically to all 16 lattice configurations. None of
    them is a control.

    Parameters
    ----------
    task : {'family', 'binary'}, default 'family'
        ``'family'`` keeps ransomware only and restricts to families with at least
        ``min_class_size`` samples.
    year_min, year_max : int, default 2012 and 2024
        Inclusive bounds of the analysis window on the primary temporal axis. The two
        bounds are set for different reasons.
    drop_empty : bool, default True
        Drop samples activating no feature. Such rows carry no behavioural
        information and would otherwise inflate goodware performance.
    dedup_exact : bool, default True
        Collapse groups sharing an identical feature vector, keeping the earliest
        member.
    min_class_size : int, default 20
        Minimum family size for the family task.
    primary_axis : str, default 'first_submission_date_year'
        Year of first submission to the sample aggregator. This is the defensible
        temporal axis and defines the analysis window.
    secondary_axis : str, default 'Year'
        Year taken from the compile timestamp. Retained deliberately as the
        material of control B2; it is attacker-controlled and unreliable.
    prior_reference_rate : float, default 0.40
        Reference positive rate targeted by control A1 on the binary task. On the
        family task A1 targets the panel-wide class distribution instead, for
        which no scalar rate is meaningful.
    """

    task: str = "family"
    year_min: int = 2012
    year_max: int = 2024
    drop_empty: bool = True
    dedup_exact: bool = True
    min_class_size: int = 20
    primary_axis: str = "first_submission_date_year"
    secondary_axis: str = "Year"
    prior_reference_rate: float = 0.40

    def __post_init__(self) -> None:
        if self.task not in ("family", "binary"):
            raise ValueError(f"task must be 'family' or 'binary', got {self.task!r}")
        if self.year_min > self.year_max:
            raise ValueError(f"year_min {self.year_min} exceeds year_max {self.year_max}")
        if not 0.0 < self.prior_reference_rate < 1.0:
            raise ValueError(
                f"prior_reference_rate must lie in (0, 1), got {self.prior_reference_rate}"
            )

    @property
    def target_column(self) -> str:
        """Return the label column implied by :attr:`task`.

        Returns
        -------
        str
            ``'ransomware_family'`` for the family task, ``'sample_type'`` for
            the binary task.
        """
        return "ransomware_family" if self.task == "family" else "sample_type"


@dataclass(frozen=True)
class SplitSpec:
    """Specification of one train/test split.

    The two designs differ in exactly one respect, how the split is drawn. Data,
    features, models and hyperparameters are identical, so the measured gap is
    attributable to the split alone.

    Parameters
    ----------
    design : {'temporal', 'random', 'random_matched', 'random_fully_matched'}
        ``'temporal'`` trains on everything before ``cut_year`` and tests on the
        following ``test_window`` years.
    cut_year : int or None, default None
        First year of the test window. Required for ``'temporal'`` and for both
        matched designs, which read the temporal split at the same cut.
    test_window : int, default 3
        Length of the test window in years, clipped at the end of the analysis window.
    test_size : float, default 0.25
        Test fraction for the ``'random'`` design. Ignored by the other two,
        which take their sizes from the data.
    stratify : bool, default True
        Stratify the random draws by the target label. Disabled automatically
        when some class is too small to stratify.
    """

    design: str = "temporal"
    cut_year: int | None = None
    test_window: int = 3
    test_size: float = 0.25
    stratify: bool = True

    #: Designs requiring a cut year, because their sizes or their boundary are
    #: defined by one.
    CUT_DEPENDENT_DESIGNS: ClassVar[tuple[str, ...]] = (
        "temporal",
        "random_matched",
        "random_fully_matched",
    )

    #: Designs that reproduce the temporal split's half sizes.
    MATCHED_DESIGNS: ClassVar[tuple[str, ...]] = ("random_matched", "random_fully_matched")

    #: Every recognised design.
    DESIGNS: ClassVar[tuple[str, ...]] = (
        "temporal",
        "random",
        "random_matched",
        "random_fully_matched",
    )

    def __post_init__(self) -> None:
        if self.design not in self.DESIGNS:
            raise ValueError(f"design must be one of {self.DESIGNS}, got {self.design!r}")
        if self.design in self.CUT_DEPENDENT_DESIGNS and self.cut_year is None:
            raise ValueError(f"cut_year is required when design={self.design!r}")
        if self.test_window < 1:
            raise ValueError(f"test_window must be at least 1, got {self.test_window}")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must lie in (0, 1), got {self.test_size}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete description of a single lattice cell.

    Parameters
    ----------
    panel : PanelSpec
        Fixed preprocessing, identical across the lattice.
    split : SplitSpec
        Split design and cut point.
    flags : ControlFlags
        The coalition of controls applied in this cell.
    model_name : str
        Key accepted by :func:`sift.models.build_model`.
    seed : int
        Base seed. Every stochastic component derives its own seed from this one
        through :func:`sift.seeding.derive_seed`; nothing consumes it directly.
    target : str, default 'ransomware_family'
        Label column. Should agree with ``panel.target_column``.
    n_features : int, default 200
        Number of features retained by the selection stage. The count is held
        constant across the lattice so control B1 varies only the set the
        selector is fitted on, never the width of the design matrix.
    """

    panel: PanelSpec
    split: SplitSpec
    flags: ControlFlags
    model_name: str
    seed: int
    target: str = "ransomware_family"
    n_features: int = 200

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be at least 1, got {self.n_features}")

    def as_dict(self) -> dict[str, Any]:
        """Return a plain nested dictionary of every configuration field.

        Returns
        -------
        dict
            JSON-serialisable representation, suitable for a result row or a
            cache manifest.
        """
        return asdict(self)

    def fingerprint(self) -> str:
        """Return a stable content hash of the configuration.

        The hash is taken over canonical JSON with sorted keys, so it depends on
        field values alone and never on dictionary ordering or process state. It
        is therefore comparable across machines and across runs.

        Returns
        -------
        str
            Sixteen hexadecimal characters.
        """
        payload = json.dumps(
            {"schema": SCHEMA_VERSION, "config": self.as_dict()},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


# --- C1: deduplication as an opt-in fifth control ---------------------------
#
# Everything above this line is the four-control lattice and is frozen.
# Depending on the cut, 0.4 to 28.7 per cent of test-window samples have an
# exact duplicate in the training window, and turning the collapse off moves
# the reported gap by 0.069, the same order as phi_A2 at 0.123. A quantity that
# large cannot sit inside fixed preprocessing and be left out of the
# attribution.
#
# The extension is strictly additive. ``ControlFlags``, ``CONTROL_NAMES`` and
# ``LATTICE_CELLS`` are untouched, so ``config_id`` in
# ``results/metrics.parquet`` keeps its four-flag meaning. A five-control run
# is reached only by asking for it by name.

#: Name of the fifth control.
C1_NAME: str = "c1_dedup"

#: The five players of the extended lattice. ``CONTROL_NAMES`` is the prefix,
#: so bit ``i`` keeps its four-control meaning for ``i < 4`` and the extended
#: index of a cell is ``four_control_index + 16 * c1_dedup``.
CONTROL_NAMES_C1: tuple[str, ...] = CONTROL_NAMES + (C1_NAME,)

#: Two-tier taxonomy for the Owen value once C1 is a player.
#:
#: C1 is protocol-side: exact duplicates of training samples in the test window
#: are an artefact of how the corpus was assembled, not a property of the
#: malware stream a deployed detector would face. That places it with B1 and
#: B2. The data-side tier keeps its two members, so the tiers become 2 and 3
#: players wide.
TIER_GROUPS_C1: tuple[tuple[str, ...], ...] = (
    ("a1_prior", "a2_labels"),
    ("b1_fs", "b2_axis", C1_NAME),
)


@dataclass(frozen=True)
class ExtendedControlFlags:
    """State of the five evaluation controls, C1 included.

    A separate type from :class:`ControlFlags` on purpose. The four-flag
    ``config_id`` encoding is already written into ``results/metrics.parquet``
    and read by three modules and the notebooks, so widening ``ControlFlags``
    would silently renumber a published table. This class is opt-in: nothing in
    the four-control path constructs one.

    Parameters
    ----------
    a1_prior, a2_labels, b1_fs, b2_axis : bool, default False
        Exactly as in :class:`ControlFlags`.
    c1_dedup : bool, default False
        When ``False`` groups of samples sharing an identical feature vector are left in
        place, so a test-window sample can be an exact copy of a training sample.
    """

    a1_prior: bool = False
    a2_labels: bool = False
    b1_fs: bool = False
    b2_axis: bool = False
    c1_dedup: bool = False

    @classmethod
    def from_index(cls, index: int) -> "ExtendedControlFlags":
        """Build the flags for an extended lattice cell index.

        Parameters
        ----------
        index : int
            Extended lattice index in ``range(32)``.

        Returns
        -------
        ExtendedControlFlags
            The flag combination addressed by ``index``. Because
            :data:`CONTROL_NAMES` is a prefix of :data:`CONTROL_NAMES_C1`,
            ``index % 16`` is the four-control index of the same cell.

        Raises
        ------
        ValueError
            If ``index`` falls outside ``range(32)``.
        """
        limit = 2 ** len(CONTROL_NAMES_C1)
        if not 0 <= index < limit:
            raise ValueError(f"index must lie in range(0, {limit}), got {index}")
        bits = {name: bool(index >> pos & 1) for pos, name in enumerate(CONTROL_NAMES_C1)}
        return cls(**bits)

    @classmethod
    def from_flags(
        cls, flags: ControlFlags, c1_dedup: bool = False
    ) -> "ExtendedControlFlags":
        """Lift a four-control cell into the extended lattice.

        Parameters
        ----------
        flags : ControlFlags
            The four-control coalition.
        c1_dedup : bool, default False
            State of the fifth control. Pass ``True`` to describe a cell of the
            existing production run, whose panel was deduplicated.

        Returns
        -------
        ExtendedControlFlags
        """
        return cls(
            a1_prior=bool(flags.a1_prior),
            a2_labels=bool(flags.a2_labels),
            b1_fs=bool(flags.b1_fs),
            b2_axis=bool(flags.b2_axis),
            c1_dedup=bool(c1_dedup),
        )

    def to_index(self) -> int:
        """Return the extended lattice index of this flag combination.

        Returns
        -------
        int
            Index in ``range(32)``, the inverse of :meth:`from_index`.
        """
        return sum(
            int(getattr(self, name)) << pos for pos, name in enumerate(CONTROL_NAMES_C1)
        )

    def base(self) -> ControlFlags:
        """Return the four-control projection of this cell.

        Returns
        -------
        ControlFlags
            The same coalition with C1 dropped. ``base().to_index()`` equals
            ``to_index() % 16``, which is what pins the four-control encoding
            against renumbering.
        """
        return ControlFlags(
            a1_prior=self.a1_prior,
            a2_labels=self.a2_labels,
            b1_fs=self.b1_fs,
            b2_axis=self.b2_axis,
        )

    def active(self) -> frozenset[str]:
        """Return the names of the enabled controls, C1 included."""
        return frozenset(name for name in CONTROL_NAMES_C1 if getattr(self, name))


def _c1_clause(enabled: bool) -> str:
    """Return the one-line reading of the C1 half of an extended cell."""
    if enabled:
        return "Exact duplicate feature vectors are collapsed to their earliest member."
    return (
        "Exact duplicate feature vectors are left in place, so a test row may be "
        "a byte-identical copy of a training row."
    )


#: Every cell of the extended lattice, indexed by its extended ``config_id``.
#:
#: Derived from :data:`LATTICE_CELLS` rather than written out again, so the two
#: enumerations cannot drift apart: cell ``i`` and cell ``i + 16`` carry the
#: same four-control reading and differ only in C1.
LATTICE_CELLS_C1: tuple[LatticeCell, ...] = tuple(
    LatticeCell(
        index,
        ExtendedControlFlags.from_index(index),  # type: ignore[arg-type]
        LATTICE_CELLS[index % len(LATTICE_CELLS)].description
        + " "
        + _c1_clause(bool(index >> len(CONTROL_NAMES) & 1)),
    )
    for index in range(2 ** len(CONTROL_NAMES_C1))
)


def lattice_cell_c1(config_id: int) -> LatticeCell:
    """Return the description of one extended lattice cell.

    Parameters
    ----------
    config_id : int
        Extended lattice index in ``range(32)``.

    Returns
    -------
    LatticeCell
        The cell, whose ``flags`` is an :class:`ExtendedControlFlags`.

    Raises
    ------
    ValueError
        If ``config_id`` falls outside ``range(32)``.
    """
    if not 0 <= config_id < len(LATTICE_CELLS_C1):
        raise ValueError(
            f"config_id must lie in range(0, {len(LATTICE_CELLS_C1)}), got {config_id}"
        )
    return LATTICE_CELLS_C1[config_id]


def panel_spec_for_c1(spec: PanelSpec, c1_dedup: bool) -> PanelSpec:
    """Return the panel specification implied by the state of C1.

    Parameters
    ----------
    spec : PanelSpec
        Template specification.
    c1_dedup : bool
        State of the fifth control.

    Returns
    -------
    PanelSpec
        ``spec`` with ``dedup_exact`` set to ``c1_dedup``.

    Notes
    -----
    C1 is expressed through ``PanelSpec.dedup_exact`` rather than as a new step
    inside the pipeline, which is what keeps the fit-cache key correct:
    ``cache.build_payload`` already hashes the panel specification *and* a
    fingerprint of the realised panel, so the two C1 states differ in the key
    twice over and neither can be served a record fitted under the other.
    """
    return replace(spec, dedup_exact=bool(c1_dedup))


def config_for_cell_c1(
    base: ExperimentConfig, flags: ExtendedControlFlags
) -> ExperimentConfig:
    """Project an extended cell onto an ordinary :class:`ExperimentConfig`.

    Parameters
    ----------
    base : ExperimentConfig
        Template supplying split, model, seed, target and feature budget.
    flags : ExtendedControlFlags
        The extended coalition.

    Returns
    -------
    ExperimentConfig
        A four-control configuration whose ``flags`` is ``flags.base()`` and
        whose ``panel.dedup_exact`` carries the state of C1.
    """
    return replace(
        base,
        flags=flags.base(),
        panel=panel_spec_for_c1(base.panel, flags.c1_dedup),
    )


def extended_flags_of(cfg: ExperimentConfig) -> ExtendedControlFlags:
    """Recover the extended coalition of a projected configuration.

    Parameters
    ----------
    cfg : ExperimentConfig
        A configuration, typically one produced by :func:`config_for_cell_c1`.

    Returns
    -------
    ExtendedControlFlags
        ``cfg.flags`` widened with ``cfg.panel.dedup_exact``. Applied to a
        configuration from the four-control production run this returns the
        cell with ``c1_dedup=True``, which is the correct reading: that run
        deduplicated.
    """
    return ExtendedControlFlags.from_flags(cfg.flags, cfg.panel.dedup_exact)
