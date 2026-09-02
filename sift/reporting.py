"""Table builders for the SIFT paper.

Two tables carry the result. The main table gives, per model, the evaluation
gap and its Shapley-Shorrocks decomposition into the four controls plus the
residual. The appendix table gives all sixteen lattice cells, which is the
primary evidence and depends on no aggregation assumption whatsoever; design
section 8.1 requires it to be printed in full.

Each builder returns a :class:`pandas.DataFrame` for inspection and notebook
use, and :func:`to_booktabs` renders any of them as a LaTeX ``booktabs``
tabular. The separation matters because the numbers must be checkable in the
notebook before they reach the manuscript.

This module carries no dependency on the SIFT data layer. It reads the column
names of the frame it is given, so the metrics schema of ``CONTRACT.md``
section 6 is honoured by convention rather than by import, and the builders can
be unit-tested on a synthetic frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from sift.shapley import Decomposition, VerificationResult

__all__ = [
    "BASES",
    "BASELINE_LABELS_BASIS",
    "CONTROL_LABELS",
    "GAP_GROUP_KEYS",
    "MIN_COMMON_SAMPLES",
    "MIN_DELTA_FOR_SHARE",
    "ModelDecomposition",
    "SCORING_GROUP_KEYS",
    "build_agreement_report",
    "class_prior_reference",
    "build_coalition_values",
    "build_decomposition_table",
    "build_gap_decomposition",
    "build_gap_table",
    "build_interaction_table",
    "build_lattice_table",
    "build_model_summary",
    "build_verification_table",
    "coalition_mapping",
    "compare_metric_decompositions",
    "compute_metrics_on_labels",
    "decompose_groups",
    "baseline_label_sets",
    "panel_matches_run",
    "recompute_a1_weights",
    "rescore_predictions",
    "verify_a1_recomputation",
    "to_booktabs",
]

CONTROL_LABELS: dict[str, str] = {
    "a1_prior": "A1",
    "a2_labels": "A2",
    "b1_fs": "B1",
    "b2_axis": "B2",
}
"""Short display labels for the four controls.

Presentation only. The authoritative control names and their application order
live in ``sift.config``; nothing here may be used to infer them.
"""

_BASELINE_CONFIG_ID: int = 0
"""Lattice index of the temporal configuration with every control switched off."""

GAP_GROUP_KEYS: tuple[str, ...] = ("cut", "model", "seed")
"""Keys identifying one instance of the game: one cut, one model, one seed."""

SCORING_GROUP_KEYS: tuple[str, ...] = ("design", "cut", "model", "seed")
"""Keys identifying one arm of one game instance, across which cells are compared."""

MIN_DELTA_FOR_SHARE: float = 0.02
"""Below this gap magnitude a share of the gap is not reported.

The floor is the noise floor named in design section 7bis.2. Dividing by a gap
smaller than the noise on that gap produces a ratio like ``R/Delta = 39.1``,
which is arithmetic noise dressed up as a result; such rows report ``NaN``
instead and the reader is sent to the signed quantity itself.
"""

MIN_COMMON_SAMPLES: int = 20
"""Below this many scored samples a frozen-basis cell is not usable.

The threshold matches ``sift.experiment.MIN_TEST_SAMPLES``: a macro average over
32 families computed on fewer samples than this is dominated by empty classes
and says more about the metric than about the model. It gates both frozen bases:
``common``, where the count is the size of the shared window, and
``baseline_labels``, where it is the number of the cell's own test samples whose
family survives the frozen support.
"""

BASELINE_LABELS_BASIS: str = "baseline_labels"
"""Name of the a priori frozen basis; see :data:`BASES`."""

BASES: tuple[str, ...] = ("as_reported", "common", BASELINE_LABELS_BASIS)
"""The three scoring bases :func:`rescore_predictions` accepts.

``as_reported``
    Each cell keeps its own test window and its own label set. This is the
    number an experimenter would publish and it stays the primary view.

``common``
    Every cell of a group is scored on the intersection of the test windows of
    all sixteen coalitions. Design section 8.5 introduced it as the
    comparability check, and it is kept here unchanged so the published
    behaviour stays reproducible, but it is not a valid check for the two
    controls that move the test window. The intersection is a subset of the
    A2-on window, because A2 works precisely by dropping test samples of unseen
    families, so the intersection has already had A2 applied to it. Measuring
    phi_A2 on that basis returns roughly zero by construction rather than as
    evidence. The same holds for B2, which chooses the time axis and so decides
    which samples fall in the window at all; the intersection is a subset of
    both axes' windows. The two controls the check most needs to validate are
    the two it cannot. The intersection also falls below
    :data:`MIN_COMMON_SAMPLES` for most groups of the matched random arm, which
    redraws its window at every coalition, so those groups are lost entirely.

``baseline_labels``
    Every cell of a group is scored over one frozen family support: the families
    present in the test window of the baseline coalition, the cell with all four
    controls switched off. Each cell keeps its own test samples but drops those
    whose family lies outside the frozen support, and macro-F1 is then averaged
    over the frozen support itself, so a family the cell cannot reach
    contributes a genuine zero instead of shrinking the denominator.

    The defining property is that the frozen support is not a function of which
    controls are on. It is fixed by the group's baseline window alone, which
    exists before any coalition is formed, so it is identical for all sixteen
    coalitions and in particular cannot shrink when A2 is switched on. That is
    what makes phi_A2 measurable on it: A2 can still remove samples, and the
    families it removes then score zero, which is exactly the cost the control
    is claimed to hide.

    Why the baseline window and not another a priori rule. The whole panel's
    thirty-two families would be independent even of the cut, but at the late
    cuts a window holds four to nineteen of them, so averaging over thirty-two
    would divide every score by a constant made mostly of families no coalition
    could ever predict, compressing the differences the decomposition is trying
    to resolve. The families present in the panel over the test years would be
    the natural population rule, but which years those are depends on the time
    axis, and the choice of axis is control B2 itself, so that rule cannot be
    stated without letting a control back into the definition. The union over
    the sixteen windows is a function of the realised windows and fails the
    defining property outright. The baseline window is the one support that is
    both reachable at that cut and fixed by something the game already treats as
    given: the reference point v(empty) is measured on it, and Delta and every
    Shapley value are differences from that reference.
"""



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    """Raise :class:`KeyError` naming every column the frame is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{context}: missing column(s) {missing!r}; frame has "
            f"{list(frame.columns)!r}"
        )


def _infer_control_columns(
    frame: pd.DataFrame, controls: Sequence[str] | None
) -> tuple[str, ...]:
    """Return the control flag columns, either as given or inferred from the frame."""
    if controls is not None:
        _require_columns(frame, controls, "control flags")
        return tuple(controls)
    inferred = tuple(name for name in CONTROL_LABELS if name in frame.columns)
    if not inferred:
        raise KeyError(
            "no control flag columns found; pass `controls` explicitly or supply "
            f"a frame carrying {list(CONTROL_LABELS)!r}"
        )
    return inferred


def _flag_string(row: pd.Series, controls: Sequence[str]) -> str:
    """Render one lattice cell as a fixed-width on/off string, for instance ``1010``."""
    return "".join("1" if bool(row[control]) else "0" for control in controls)


def _escape_latex(text: str) -> str:
    """Escape the LaTeX specials that occur in model and control names."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Main decomposition table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelDecomposition:
    """One row of the main table: a model, its gap and its decomposition.

    Attributes
    ----------
    model : str
        Model name as it should appear in the table.
    f1_random : float
        macro-F1 under the random reference design, averaged over seeds.
    f1_temporal : float
        macro-F1 under the temporal design with no control applied.
    decomposition : sift.shapley.Decomposition
        The attribution, residual, dividends and arithmetic checks.
    intervals : Mapping of {str: tuple of (float, float)}, optional
        Bootstrap confidence interval per control name, and optionally for the
        key ``"residual"``. Design section 8.7 requires every reported
        attribution to carry one, and requires any interval containing zero to
        be called out rather than silently printed.
    """

    model: str
    f1_random: float
    f1_temporal: float
    decomposition: Decomposition
    intervals: Mapping[str, tuple[float, float]] | None = None


def build_decomposition_table(
    rows: Sequence[ModelDecomposition],
    controls: Sequence[str] | None = None,
    include_intervals: bool = True,
    include_owen: bool = False,
) -> pd.DataFrame:
    """Build the per-model decomposition table, the centrepiece of the paper.

    Parameters
    ----------
    rows : sequence of ModelDecomposition
        One entry per model. Order is preserved.
    controls : sequence of str, optional
        Control names, fixing the column order. Inferred from the first
        decomposition when omitted.
    include_intervals : bool, optional
        Emit ``<name>_lo`` and ``<name>_hi`` columns and a
        ``<name>_covers_zero`` flag wherever an interval was supplied.
    include_owen : bool, optional
        Emit the Owen value alongside the Shapley value, for the tier-level
        reading of the same lattice.

    Returns
    -------
    pandas.DataFrame
        Columns ``model``, ``f1_random``, ``f1_temporal``, ``delta``, one
        ``phi_<control>`` per control, ``residual``, ``residual_share``, plus
        the optional interval and Owen columns, and ``checks_passed``.

    Raises
    ------
    ValueError
        If ``rows`` is empty, or if the decompositions disagree about the
        player set.

    Notes
    -----
    ``residual_share`` is ``residual / delta`` and is left as ``NaN`` when the
    gap is zero, since a share of a vanishing gap is not defined. It is a
    convenience for the reader and never the basis of a claim.

    Results are reported per model and not averaged over the twenty cells,
    because a control that dominates for one model and not for another is
    itself the finding, and an average would hide it; see design section 8.7.
    """
    if not rows:
        raise ValueError("no decompositions to tabulate")

    if controls is None:
        controls = tuple(sorted(rows[0].decomposition.shapley))
    controls = tuple(controls)
    for row in rows:
        if set(row.decomposition.shapley) != set(controls):
            raise ValueError(
                f"model {row.model!r} has players "
                f"{sorted(row.decomposition.shapley)!r}, expected {sorted(controls)!r}"
            )

    records: list[dict[str, object]] = []
    for row in rows:
        decomposition = row.decomposition
        delta = decomposition.delta
        record: dict[str, object] = {
            "model": row.model,
            "f1_random": float(row.f1_random),
            "f1_temporal": float(row.f1_temporal),
            "delta": delta,
        }
        for control in controls:
            record[f"phi_{control}"] = decomposition.shapley[control]
        record["residual"] = decomposition.residual
        record["residual_share"] = (
            decomposition.residual / delta if delta != 0.0 else float("nan")
        )

        if include_owen:
            if decomposition.owen is None:
                raise ValueError(
                    f"model {row.model!r} has no Owen value; call `decompose` with "
                    "`groups` to request one"
                )
            for control in controls:
                record[f"owen_{control}"] = decomposition.owen[control]
            if decomposition.tier_shapley is not None:
                for tier, value in decomposition.tier_shapley.items():
                    record[f"tier_{tier}"] = value

        if include_intervals and row.intervals is not None:
            for name, bounds in row.intervals.items():
                low, high = float(bounds[0]), float(bounds[1])
                record[f"{name}_lo"] = low
                record[f"{name}_hi"] = high
                record[f"{name}_covers_zero"] = bool(low <= 0.0 <= high)

        record["checks_passed"] = decomposition.all_checks_passed
        records.append(record)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Lattice appendix table
# ---------------------------------------------------------------------------


def build_lattice_table(
    values: pd.DataFrame,
    model: str,
    cut: int,
    controls: Sequence[str] | None = None,
    group_keys: Sequence[str] = GAP_GROUP_KEYS,
) -> pd.DataFrame:
    """Build the sixteen-row lattice appendix table for one model at one cut.

    Parameters
    ----------
    values : pandas.DataFrame
        Output of :func:`build_coalition_values`, carrying both arms and ``v``.
    model : str
        Model to select.
    cut : int
        Cut-point to select.
    controls : sequence of str, optional
        Control flag columns, fixing the bit order of the flag string.
    group_keys : sequence of str, optional
        Keys identifying one instance of the game. Seeds inside the selection
        are averaged, and their spread is reported as ``v_std``.

    Returns
    -------
    pandas.DataFrame
        Sixteen rows ordered by ``config_id``, with ``config_id``, ``flags``,
        ``active``, ``n_seeds``, ``m_temporal``, ``m_reference``, ``delta``,
        ``v``, ``v_std`` and ``share_of_delta``.

    Raises
    ------
    ValueError
        If the selection does not carry exactly the sixteen coalitions.

    Notes
    -----
    This table is the primary evidence of the study. It depends on no
    aggregation assumption at all, so the design requires it to be printed in
    full rather than summarised, and a missing cell raises instead of being
    dropped.
    """
    _require_columns(values, ["model", "cut", "config_id", "v", "delta"], "lattice table")
    control_columns = _infer_control_columns(values, controls)

    selected = values[(values["model"] == model) & (values["cut"] == cut)]
    if selected.empty:
        raise ValueError(f"no lattice rows for model={model!r} cut={cut!r}")

    found = sorted(int(item) for item in selected["config_id"].unique())
    if found != list(range(16)):
        raise ValueError(
            f"expected config_id 0..15 for model={model!r} cut={cut!r}, got {found!r}; "
            "the lattice must be complete for the exact Shapley value to be defined"
        )

    table = (
        selected.groupby("config_id", as_index=False)
        .agg(
            n_seeds=("v", "size"),
            m_temporal=("m_temporal", "mean"),
            m_reference=("m_reference", "mean"),
            delta=("delta", "mean"),
            v=("v", "mean"),
            v_std=("v", "std"),
            share_of_delta=("share_of_delta", "mean"),
            **{name: (name, "first") for name in control_columns},
        )
        .sort_values("config_id")
        .reset_index(drop=True)
    )
    table["flags"] = table.apply(lambda row: _flag_string(row, control_columns), axis=1)
    table["active"] = table.apply(
        lambda row: ", ".join(
            CONTROL_LABELS.get(name, name) for name in control_columns if bool(row[name])
        )
        or "none",
        axis=1,
    )
    ordered = [
        "config_id",
        "flags",
        "active",
        "n_seeds",
        "m_temporal",
        "m_reference",
        "delta",
        "v",
        "v_std",
        "share_of_delta",
    ]
    return table[ordered + list(control_columns)]


# ---------------------------------------------------------------------------
# Interaction and verification tables
# ---------------------------------------------------------------------------


def build_interaction_table(
    dividends: Mapping[frozenset[str], float],
    controls: Sequence[str] | None = None,
    max_order: int = 2,
) -> pd.DataFrame:
    """Tabulate Harsanyi dividends up to a given interaction order.

    Parameters
    ----------
    dividends : Mapping of {frozenset of str: float}
        Output of ``sift.shapley.harsanyi_dividends``.
    controls : sequence of str, optional
        Display order of the controls. Sorted player set when omitted.
    max_order : int, optional
        Highest coalition size to include. Two gives main effects and pairwise
        interactions, which is what design section 8.4 asks the paper to show.

    Returns
    -------
    pandas.DataFrame
        Columns ``order``, ``coalition`` and ``dividend``, sorted by order and
        then by descending magnitude within each order.

    Notes
    -----
    A pairwise dividend that is large relative to the main effects is the
    quantitative answer to the reviewer question of whether the controls
    interact, and therefore to why switching them off one at a time would be
    order-dependent.
    """
    if controls is None:
        players: set[str] = set()
        for coalition in dividends:
            players.update(coalition)
        controls = tuple(sorted(players))
    controls = tuple(controls)

    records: list[dict[str, object]] = []
    for size in range(1, max_order + 1):
        for subset in combinations(controls, size):
            key = frozenset(subset)
            if key not in dividends:
                continue
            records.append(
                {
                    "order": size,
                    "coalition": " + ".join(
                        CONTROL_LABELS.get(control, control) for control in subset
                    ),
                    "dividend": float(dividends[key]),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["_magnitude"] = frame["dividend"].abs()
    frame = (
        frame.sort_values(["order", "_magnitude"], ascending=[True, False])
        .drop(columns="_magnitude")
        .reset_index(drop=True)
    )
    return frame


def build_verification_table(
    checks: Sequence[VerificationResult],
) -> pd.DataFrame:
    """Tabulate arithmetic checks for the reproducibility appendix.

    Parameters
    ----------
    checks : sequence of sift.shapley.VerificationResult
        Results from ``sift.shapley.run_all_checks``.

    Returns
    -------
    pandas.DataFrame
        Columns ``check``, ``passed``, ``deviation``, ``tolerance``, ``detail``.

    Notes
    -----
    Printing the achieved deviations, rather than only the fact that the checks
    passed, lets a reader confirm that the core equation holds to machine
    precision instead of merely to the declared tolerance.
    """
    return pd.DataFrame.from_records(
        [
            {
                "check": result.name,
                "passed": result.passed,
                "deviation": result.deviation,
                "tolerance": result.tolerance,
                "detail": result.detail,
            }
            for result in checks
        ]
    )


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def to_booktabs(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    columns: Sequence[str] | None = None,
    headers: Mapping[str, str] | None = None,
    float_format: str = "{:.3f}",
    column_format: str | None = None,
    note: str | None = None,
    environment: str = "table",
    placement: str = "t",
    escape_headers: bool = True,
    raw_columns: Sequence[str] = (),
    column_formats: Mapping[str, str] | None = None,
) -> str:
    """Render a frame as a LaTeX ``booktabs`` table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to render.
    caption : str
        Caption text. Passed through unescaped so that it may contain maths.
    label : str
        Label, used as ``\\label{<label>}``.
    columns : sequence of str, optional
        Columns to emit, in order. All columns when omitted.
    headers : Mapping of {str: str}, optional
        Column name to header text. Headers given here are emitted verbatim
        when ``escape_headers`` is false, which is how ``$\\phi_{A1}$`` and
        ``$\\Delta$`` reach the output.
    float_format : str, optional
        Format applied to floating point cells. ``NaN`` renders as ``--``.
    column_format : str, optional
        LaTeX column specification. Inferred as ``l`` for object columns and
        ``r`` for numeric ones when omitted.
    note : str, optional
        Text placed below the tabular in ``\\footnotesize``. The natural place
        for the statement that the row total is an identity rather than a
        finding.
    environment : str, optional
        Outer environment, ``"table"`` for one column or ``"table*"`` for two.
    placement : str, optional
        Float placement specifier.
    escape_headers : bool, optional
        Escape LaTeX specials in headers. Set false when supplying maths.
    raw_columns : sequence of str, optional
        Columns whose string cells are already LaTeX and must be emitted
        verbatim. Without this a cell such as ``$+0.016^{\dagger}$`` would be
        escaped into visible backslashes.
    column_formats : Mapping of {str: str}, optional
        Per-column float format overriding ``float_format``. A standard
        deviation or a percentage should not carry the leading ``+`` that suits
        a signed attribution.

    Returns
    -------
    str
        A complete float environment, ready to paste into the manuscript.

    Notes
    -----
    Rendering is done here rather than through ``DataFrame.to_latex`` so that
    header maths, the ``--`` for missing values and the trailing note behave
    predictably across pandas versions.
    """
    selected = list(frame.columns) if columns is None else list(columns)
    _require_columns(frame, selected, "booktabs rendering")
    view = frame[selected]

    if column_format is None:
        column_format = "".join(
            "r" if (pd.api.types.is_numeric_dtype(view[name]) or name in set(raw_columns))
            else "l"
            for name in selected
        )

    header_map = dict(headers or {})
    rendered_headers = []
    for name in selected:
        text = header_map.get(name, name)
        rendered_headers.append(_escape_latex(text) if escape_headers else text)

    raw = set(raw_columns)
    formats = dict(column_formats or {})

    def render(value: object, column: str = "") -> str:
        if column in raw and isinstance(value, str):
            return value
        if value is None:
            return "--"
        if isinstance(value, bool):
            return r"\checkmark" if value else r"$\times$"
        if isinstance(value, float):
            if pd.isna(value):
                return "--"
            return formats.get(column, float_format).format(value)
        if isinstance(value, (int,)):
            return str(value)
        if pd.isna(value):
            return "--"
        return _escape_latex(str(value))

    lines = [
        f"\\begin{{{environment}}}[{placement}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        " & ".join(rendered_headers) + r" \\",
        "\\midrule",
    ]
    for _, row in view.iterrows():
        lines.append(
            " & ".join(render(row[name], name) for name in selected) + r" \\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append(r"\\[2pt]")
        lines.append(r"\footnotesize " + note)
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rescoring from the stored predictions
# ---------------------------------------------------------------------------

def compute_metrics_on_labels(
    y_true,
    y_pred,
    labels,
    sample_weight=None,
) -> dict[str, float]:
    """Compute the reported metrics over a label set the caller fixes.

    ``sift.metrics.compute_metrics`` reads the label set off ``y_true``, which
    makes the denominator of the macro average a property of the window being
    scored. That is the right default and the wrong thing for a frozen basis,
    where the point is that the denominator must not move with the coalition.

    Parameters
    ----------
    y_true, y_pred : array-like
        Labels of the scored subset, aligned.
    labels : array-like
        The frozen label set. Duplicates are collapsed and the set is sorted, so
        the same set in any order gives the same numbers. A label absent from
        ``y_true`` still occupies a slot in the macro average and contributes
        zero, which is the whole point: a family the coalition dropped is
        counted as unclassified rather than counted out of the average.
    sample_weight : array-like, optional
        Control A1 weights for the scored subset.

    Returns
    -------
    dict
        The four ``sift.metrics.METRIC_NAMES`` plus ``k_test``, which is here
        the size of the frozen set rather than of the window.

    Notes
    -----
    Only macro-F1 takes a label set. Balanced accuracy, MCC and accuracy are
    computed exactly as ``compute_metrics`` computes them and remain functions
    of the classes present in the scored subset; scikit-learn offers no label
    pinning for them. The frozen basis therefore fixes the denominator of the
    primary metric and of that metric only, and a decomposition of balanced
    accuracy on this basis carries the same caveat it carries on the common one.
    The arithmetic is delegated to ``sift.metrics`` so that this module cannot
    drift from the one that produced the stored numbers.
    """
    import numpy  # noqa: PLC0415

    from sift.metrics import METRIC_NAMES, _metric_from_labels  # noqa: PLC0415

    y_true = numpy.asarray(y_true)
    y_pred = numpy.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true holds {y_true.shape[0]} labels but y_pred holds {y_pred.shape[0]}"
        )
    if y_true.shape[0] == 0:
        raise ValueError("cannot compute metrics on an empty scored subset")
    pinned = numpy.unique(numpy.asarray(list(labels)))
    if pinned.size == 0:
        raise ValueError("the frozen label set is empty")

    computed = {
        name: _metric_from_labels(y_true, y_pred, name, pinned, sample_weight)
        for name in METRIC_NAMES
    }
    computed["k_test"] = int(pinned.size)
    return computed


def baseline_label_sets(
    joined: pd.DataFrame,
    group_keys: Sequence[str] = SCORING_GROUP_KEYS,
    baseline_config_id: int = _BASELINE_CONFIG_ID,
) -> dict[tuple, tuple]:
    """Frozen family support of every group, read off the baseline coalition.

    Parameters
    ----------
    joined : pandas.DataFrame
        Per-sample predictions carrying ``y_true``, ``config_id`` and the group
        keys, as :func:`rescore_predictions` assembles them.
    group_keys : sequence of str, optional
        Columns identifying one arm of one game instance.
    baseline_config_id : int, optional
        Lattice index of the coalition with every control switched off.

    Returns
    -------
    dict
        Group key tuple to the sorted tuple of families present in that group's
        baseline test window.

    Raises
    ------
    ValueError
        If a group carries no baseline cell, or if its baseline window is empty.
        Both are refused rather than filled in: a frozen support inferred from
        whichever cells happen to be present would be a function of which
        coalitions ran, which is the property this basis exists to deny.

    Notes
    -----
    One lookup per group, and the same tuple is then handed to all sixteen
    coalitions of that group. Reading it once is not an optimisation, it is the
    guarantee: there is no code path on which a cell's own flags can influence
    the support it is scored against.
    """
    keys = list(group_keys)
    _require_columns(joined, ["y_true", "config_id", *keys], "baseline label sets")
    frozen: dict[tuple, tuple] = {}
    for group_key, block in joined.groupby(keys, sort=False):
        normalised = group_key if isinstance(group_key, tuple) else (group_key,)
        baseline = block[block["config_id"] == baseline_config_id]
        if baseline.empty:
            found = sorted(int(value) for value in block["config_id"].unique())
            raise ValueError(
                f"group {normalised!r} carries no configuration "
                f"{baseline_config_id}, only {found!r}; the {BASELINE_LABELS_BASIS!r} "
                "basis is defined by the window of the coalition with no controls "
                "on, and no other cell may stand in for it"
            )
        labels = tuple(sorted(set(baseline["y_true"].tolist())))
        if not labels:
            raise ValueError(
                f"group {normalised!r} has an empty baseline test window; the "
                f"{BASELINE_LABELS_BASIS!r} basis has no support to freeze"
            )
        frozen[normalised] = labels
    return frozen


def rescore_predictions(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    basis: str = "as_reported",
    metric_function=None,
    group_keys: Sequence[str] = SCORING_GROUP_KEYS,
    prior_reference: pd.Series | None = None,
    lattice_config_count: int = 16,
    baseline_config_id: int = _BASELINE_CONFIG_ID,
    label_metric_function=None,
) -> pd.DataFrame:
    """Recompute every metric from the stored per-sample predictions.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Contents of ``results/predictions.parquet``: ``fit_id``, ``sample_id``,
        ``grp_id``, ``y_true``, ``y_pred``, ``test_year``.
    metrics : pandas.DataFrame
        Contents of ``results/metrics.parquet``, used only for the fit metadata
        that ``predictions`` does not carry. Its metric columns are not read
        here; they are the drift check, not the source of truth.
    basis : {"as_reported", "common", "baseline_labels"}
        ``"as_reported"`` scores each fit on its own test window, which is the
        number an experimenter would publish. ``"common"`` scores every cell of
        a group on the intersection of the test windows of all its cells, which
        is the comparability check of design section 8.5. ``"baseline_labels"``
        scores every cell over the family support of the group's baseline
        window, which is the same check with a frozen set that does not depend
        on which controls are on. See :data:`BASES` for the full statement of
        all three and for why the common basis cannot check A2 or B2.
    metric_function : callable, optional
        Function of ``(y_true, y_pred)`` returning a mapping of metric names to
        values. Defaults to ``sift.metrics.compute_metrics``. Used by the
        ``as_reported`` and ``common`` bases only.
    group_keys : sequence of str, optional
        Columns identifying one arm of one game instance. Cells are intersected
        within a group and never across groups.
    prior_reference : pandas.Series, optional
        Reference class distribution from :func:`class_prior_reference`. When
        given, control A1's sample weights are recomputed for every ``a1_prior``
        cell so that those cells are scored exactly as the production run scored
        them. When omitted, A1 cells are scored unweighted and flagged.
    lattice_config_count : int, optional
        Number of configurations a lattice group must carry before the common
        basis will intersect over it. A group holding fewer raises, because the
        intersection would then be one cell's own window. The
        ``baseline_labels`` basis imposes no such requirement: its frozen set
        comes from one named cell, so a partial lattice is scored correctly
        rather than silently rebased.
    baseline_config_id : int, optional
        Lattice index of the coalition with every control switched off, whose
        window defines the frozen support of the ``baseline_labels`` basis.
    label_metric_function : callable, optional
        Function of ``(y_true, y_pred, labels, sample_weight)`` used by the
        ``baseline_labels`` basis. Defaults to
        :func:`compute_metrics_on_labels`.

    Returns
    -------
    pandas.DataFrame
        One row per fit, carrying ``fit_id``, the group keys, ``config_id``,
        the control flags, every recomputed metric, ``n_scored``,
        ``k_scored``, ``basis``, ``reconstructable`` and, for the common basis,
        ``n_common`` and ``common_feasible``.

        The ``baseline_labels`` basis adds ``frozen_labels``, the support
        itself, ``n_frozen_labels``, its size, ``n_window``, the size of the
        cell's own test window before the support was applied, ``k_present``,
        how many frozen families the scored samples actually contain, and
        ``baseline_feasible``. ``n_scored`` is then the number of samples the
        cell was really scored on, which differs from cell to cell: coverage is
        a reported quantity on this basis rather than something the reader has
        to infer from the design. ``k_scored`` is the size of the frozen support
        and is constant across the group by construction.

    Notes
    -----
    Control A1 is applied as a ``sample_weight`` at metric time and that weight
    vector is not stored in ``predictions.parquet``. It does not need to be:
    the weight of a sample is ``p_reference(c) / p_window(c)`` for its class,
    the reference is the class distribution of the panel, and the window's label
    vector is stored. Passing ``prior_reference`` therefore recovers those cells
    exactly; :func:`verify_a1_recomputation` proves the recovery reproduces the
    stored metric before any of it is believed.

    Without ``prior_reference`` the A1 cells are scored unweighted and marked
    ``reconstructable=False``. That is reported rather than hidden, because a
    silent unweighted recomputation would understate exactly the control it
    concerns.

    On the common basis the weights are recomputed from the labels of the shared
    window rather than of the full window, since the class proportions of a
    subset differ and the weight is a function of those proportions. The same
    holds on the ``baseline_labels`` basis, where the subset is the cell's own
    window with the out-of-support families removed.
    """
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {list(BASES)}")
    if metric_function is None:
        from sift.metrics import compute_metrics as metric_function  # noqa: PLC0415
    if label_metric_function is None:
        label_metric_function = compute_metrics_on_labels

    keys = list(group_keys)
    meta_columns = ["fit_id", "config_id", *keys]
    control_columns = _infer_control_columns(metrics, None)
    _require_columns(metrics, meta_columns, "rescoring metadata")
    _require_columns(predictions, ["fit_id", "sample_id", "y_true", "y_pred"], "predictions")

    role_column = ["role"] if "role" in metrics.columns else []
    meta = metrics[
        meta_columns + list(control_columns) + role_column
    ].drop_duplicates("fit_id")
    joined = predictions.merge(meta, on="fit_id", how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("no prediction rows matched the metrics table on fit_id")

    common_by_group: dict[tuple, frozenset[int]] = {}
    configs_by_group: dict[tuple, int] = {}
    if basis == "common":
        for group_key, block in joined.groupby(keys, sort=False):
            normalised = group_key if isinstance(group_key, tuple) else (group_key,)
            per_cell = [
                frozenset(cell["sample_id"].to_numpy().tolist())
                for _, cell in block.groupby("config_id", sort=False)
            ]
            # Intersecting fewer cells than the group is supposed to carry
            # succeeds silently and hands back one cell's own window, which is a
            # vacuous freeze that looks exactly like a working one downstream.
            is_lattice = (
                "role" not in block.columns
                or bool((block["role"] == "lattice").any())
            )
            if is_lattice and len(per_cell) != lattice_config_count:
                found = sorted(int(value) for value in block["config_id"].unique())
                raise ValueError(
                    f"group {normalised!r} carries {len(per_cell)} configuration(s) "
                    f"{found!r}, not the {lattice_config_count} the common basis "
                    "must intersect over; freezing across a partial lattice would "
                    "return one cell's own window and silently defeat the check"
                )
            shared = frozenset.intersection(*per_cell) if per_cell else frozenset()
            common_by_group[normalised] = shared
            configs_by_group[normalised] = len(per_cell)

    frozen_by_group: dict[tuple, tuple] = {}
    if basis == BASELINE_LABELS_BASIS:
        frozen_by_group = baseline_label_sets(joined, keys, baseline_config_id)

    records: list[dict[str, object]] = []
    for fit_id, block in joined.groupby("fit_id", sort=False):
        head = block.iloc[0]
        group_key = tuple(head[key] for key in keys)
        scored = block
        shared = common_by_group.get(group_key)
        frozen = frozen_by_group.get(group_key)
        feasible = True
        if basis == "common":
            shared = shared or frozenset()
            feasible = len(shared) >= MIN_COMMON_SAMPLES
            scored = block[block["sample_id"].isin(shared)]
        elif basis == BASELINE_LABELS_BASIS:
            scored = block[block["y_true"].isin(frozen)]
            feasible = len(scored) >= MIN_COMMON_SAMPLES

        record: dict[str, object] = {
            "fit_id": fit_id,
            "config_id": int(head["config_id"]),
            **{key: head[key] for key in keys},
            **{name: bool(head[name]) for name in control_columns},
            "basis": basis,
            "reconstructable": (
                prior_reference is not None or not bool(head.get("a1_prior", False))
            ),
            "a1_weighted": bool(head.get("a1_prior", False)),
        }
        if basis == "common":
            record["n_common"] = int(len(shared or ()))
            record["n_configs_in_group"] = int(configs_by_group.get(group_key, 0))
            record["common_feasible"] = feasible
        if basis == BASELINE_LABELS_BASIS:
            record["frozen_labels"] = frozen
            record["n_frozen_labels"] = int(len(frozen))
            record["n_window"] = int(len(block))
            record["k_present"] = int(scored["y_true"].nunique())
            record["baseline_feasible"] = feasible

        if scored.empty or not feasible:
            record.update({"n_scored": int(len(scored)), "k_scored": 0})
            records.append(record)
            continue

        y_true = scored["y_true"].to_numpy()
        weights = None
        if prior_reference is not None and bool(head.get("a1_prior", False)):
            weights = recompute_a1_weights(y_true, prior_reference)
        if basis == BASELINE_LABELS_BASIS:
            computed = label_metric_function(
                y_true,
                scored["y_pred"].to_numpy(),
                labels=frozen,
                sample_weight=weights,
            )
        else:
            computed = metric_function(
                y_true, scored["y_pred"].to_numpy(), sample_weight=weights
            )
        record.update({name: value for name, value in computed.items() if name != "k_test"})
        record["n_scored"] = int(len(scored))
        record["k_scored"] = int(computed.get("k_test", 0))
        records.append(record)

    return pd.DataFrame.from_records(records)


def build_agreement_report(
    recomputed: pd.DataFrame,
    metrics: pd.DataFrame,
    metric_columns: Sequence[str] = ("macro_f1", "balanced_accuracy", "mcc", "accuracy"),
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Compare metrics recomputed from predictions against the stored columns.

    Parameters
    ----------
    recomputed : pandas.DataFrame
        Output of :func:`rescore_predictions` on the ``"as_reported"`` basis.
    metrics : pandas.DataFrame
        Contents of ``results/metrics.parquet``.
    metric_columns : sequence of str, optional
        Metric columns present in both frames.
    tolerance : float, optional
        Absolute tolerance for calling a pair equal.

    Returns
    -------
    pandas.DataFrame
        One row per metric and reconstructability class, with ``n_fits``,
        ``n_mismatch``, ``max_abs_diff`` and ``worst_fit_id``.

    Notes
    -----
    The two files are written by different code paths, so a disagreement means
    one of them is wrong and no number downstream can be trusted. Cells that
    carry the A1 weighting are reported separately, because the weight vector
    is absent from ``predictions.parquet`` and those cells are expected to
    differ; a disagreement among the reconstructable cells is a genuine fault.
    """
    columns = [name for name in metric_columns if name in recomputed.columns]
    _require_columns(metrics, ["fit_id", *columns], "agreement report")
    merged = recomputed.merge(
        metrics[["fit_id", *columns]], on="fit_id", how="inner", suffixes=("_new", "_old")
    )
    if merged.empty:
        raise ValueError("no fits in common between the recomputed and stored metrics")

    records: list[dict[str, object]] = []
    for reconstructable, block in merged.groupby("reconstructable", sort=True):
        for name in columns:
            difference = (block[f"{name}_new"] - block[f"{name}_old"]).abs()
            worst_position = int(difference.to_numpy().argmax())
            records.append(
                {
                    "metric": name,
                    "reconstructable": bool(reconstructable),
                    "n_fits": int(len(block)),
                    "n_mismatch": int((difference > tolerance).sum()),
                    "max_abs_diff": float(difference.max()),
                    "worst_fit_id": str(block.iloc[worst_position]["fit_id"]),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["reconstructable", "metric"], ascending=[False, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The gap-based characteristic function
# ---------------------------------------------------------------------------


def build_gap_table(
    scored: pd.DataFrame,
    metric_column: str = "macro_f1",
    temporal_design: str = "temporal",
    reference_design: str = "random_fully_matched",
    controls: Sequence[str] | None = None,
    group_keys: Sequence[str] = GAP_GROUP_KEYS,
) -> pd.DataFrame:
    """Pair the two lattice arms and form ``Delta(S)`` for every coalition.

    Parameters
    ----------
    scored : pandas.DataFrame
        Lattice cells only, already restricted with ``lattice_cells``. Must
        carry ``design``, ``config_id``, the group keys, the control flags and
        ``metric_column``.
    metric_column : str, optional
        Metric forming the gap. macro-F1 is the primary metric of the study.
    temporal_design, reference_design : str, optional
        Names of the two arms. The reference arm moves with the coalition,
        which is the whole point: ``b2_axis`` changes which families the
        temporal window holds and the reference restricts its test pool to that
        same family set.
    controls : sequence of str, optional
        Control flag columns. Inferred from the frame when omitted.
    group_keys : sequence of str, optional
        Keys identifying one instance of the game.

    Returns
    -------
    pandas.DataFrame
        One row per group and coalition, with ``m_temporal``, ``m_reference``
        and ``delta``, plus the control flags and a ``flags`` string.

    Raises
    ------
    ValueError
        If either arm is missing, or if a group does not carry all sixteen
        coalitions on both arms.

    Notes
    -----
    ``Delta(S) = M_reference(S) - M_temporal(S)``. Both arms are recomputed at
    every coalition, so the subtraction is between two numbers measured on the
    same family set, which is what makes ``v(S)`` comparable across the lattice.
    """
    keys = list(group_keys)
    _require_columns(scored, ["design", "config_id", metric_column, *keys], "gap table")
    control_columns = _infer_control_columns(scored, controls)

    present = set(scored["design"].unique())
    for design in (temporal_design, reference_design):
        if design not in present:
            raise ValueError(
                f"design {design!r} is absent from the lattice cells; present: "
                f"{sorted(present)!r}"
            )

    aggregated = (
        scored.groupby(["design", *keys, "config_id"], as_index=False)
        .agg(
            metric=(metric_column, "mean"),
            n_rows=(metric_column, "size"),
            **{name: (name, "first") for name in control_columns},
        )
    )

    temporal = aggregated[aggregated["design"] == temporal_design]
    reference = aggregated[aggregated["design"] == reference_design]
    merged = temporal.merge(
        reference[[*keys, "config_id", "metric"]],
        on=[*keys, "config_id"],
        how="inner",
        suffixes=("_temporal", "_reference"),
        validate="one_to_one",
    )

    for group_key, block in merged.groupby(keys, sort=True):
        found = sorted(int(value) for value in block["config_id"].unique())
        if found != list(range(16)):
            raise ValueError(
                f"group {group_key!r} carries config_id {found!r}; all sixteen "
                "coalitions are required on both arms for the exact Shapley value"
            )

    merged = merged.rename(
        columns={"metric_temporal": "m_temporal", "metric_reference": "m_reference"}
    )
    merged["delta"] = merged["m_reference"] - merged["m_temporal"]
    merged["flags"] = merged.apply(lambda row: _flag_string(row, control_columns), axis=1)
    ordered = [*keys, "config_id", "flags", "m_temporal", "m_reference", "delta"]
    return merged[ordered + list(control_columns)].sort_values(
        [*keys, "config_id"]
    ).reset_index(drop=True)


def build_coalition_values(
    gap_table: pd.DataFrame,
    controls: Sequence[str] | None = None,
    group_keys: Sequence[str] = GAP_GROUP_KEYS,
) -> pd.DataFrame:
    """Turn the gap table into the characteristic function ``v(S)``.

    Parameters
    ----------
    gap_table : pandas.DataFrame
        Output of :func:`build_gap_table`.
    controls : sequence of str, optional
        Control flag columns. Inferred when omitted.
    group_keys : sequence of str, optional
        Keys identifying one instance of the game.

    Returns
    -------
    pandas.DataFrame
        The gap table with ``delta_empty``, ``v`` and ``share_of_delta`` added.

    Notes
    -----
    ``v(S) = Delta(empty) - Delta(S)``: how much of the baseline gap is removed
    once the controls in ``S`` are applied. ``v(empty) = 0`` follows
    immediately, and the residual ``R = Delta(N)`` is what the controls leave
    behind.

    ``v`` is signed. A control that widens the gap gives a negative ``v`` and
    the design forbids clipping it, since the sign is the finding.
    """
    keys = list(group_keys)
    control_columns = _infer_control_columns(gap_table, controls)
    _require_columns(gap_table, ["config_id", "delta", *keys], "coalition values")

    baseline = (
        gap_table[gap_table["config_id"] == _BASELINE_CONFIG_ID]
        .set_index(keys)["delta"]
        .rename("delta_empty")
    )
    values = gap_table.join(baseline, on=keys)
    present = set(map(tuple, baseline.index.to_frame().to_numpy().tolist()))
    wanted = set(map(tuple, values[keys].drop_duplicates().to_numpy().tolist()))
    absent = wanted - present
    if absent:
        raise ValueError(
            f"the baseline coalition row is missing for {len(absent)} group(s), "
            f"for example {sorted(absent)[:3]!r}; v(S) is defined relative to "
            "Delta(empty) and cannot be formed without it"
        )
    values["v"] = values["delta_empty"] - values["delta"]
    denominator = values["delta_empty"].where(
        values["delta_empty"].abs() >= MIN_DELTA_FOR_SHARE
    )
    values["share_of_delta"] = values["v"] / denominator
    values["coalition"] = values.apply(
        lambda row: frozenset(name for name in control_columns if bool(row[name])), axis=1
    )
    return values


def coalition_mapping(
    values: pd.DataFrame, group_keys: Sequence[str] = GAP_GROUP_KEYS
) -> dict[tuple, dict[frozenset[str], float]]:
    """Index the characteristic function by game instance.

    Parameters
    ----------
    values : pandas.DataFrame
        Output of :func:`build_coalition_values`.
    group_keys : sequence of str, optional
        Keys identifying one instance of the game.

    Returns
    -------
    dict
        Mapping from the group key tuple to a ``{frozenset: float}`` suitable
        for :func:`sift.shapley.exact_shapley`.
    """
    keys = list(group_keys)
    mapping: dict[tuple, dict[frozenset[str], float]] = {}
    for group_key, block in values.groupby(keys, sort=True):
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        mapping[key] = {
            coalition: float(value)
            for coalition, value in zip(block["coalition"], block["v"], strict=True)
        }
    return mapping


def decompose_groups(
    values: pd.DataFrame,
    groups: Sequence[Sequence[str]] | None = None,
    group_names: Sequence[str] | None = None,
    group_keys: Sequence[str] = GAP_GROUP_KEYS,
    on_incomplete: str = "raise",
) -> tuple[dict[tuple, Decomposition], pd.DataFrame, pd.DataFrame]:
    """Run the full decomposition on every game instance.

    Parameters
    ----------
    values : pandas.DataFrame
        Output of :func:`build_coalition_values`.
    groups : sequence of sequence of str, optional
        A priori unions for the Owen value, for SIFT the two tiers.
    group_names : sequence of str, optional
        Tier labels.
    group_keys : sequence of str, optional
        Keys identifying one instance of the game.

    on_incomplete : {"raise", "skip"}, optional
        What to do with a group whose value function contains a non-finite
        entry, which happens when a cell could not be scored on the requested
        basis. ``"raise"`` is the default so nothing is dropped silently;
        ``"skip"`` excludes the group and returns it in the third element.

    Returns
    -------
    tuple
        A mapping from group key to :class:`sift.shapley.Decomposition`; a tidy
        frame with one row per decomposed instance carrying ``delta_empty``, one
        ``phi_<control>`` per control, ``residual``, the Owen and tier columns
        when requested, and ``checks_passed``; and a frame of the groups that
        were skipped, with the reason.

    Notes
    -----
    Instances are kept separate all the way through. The design forbids
    averaging the twenty cut-by-model cells into a single decomposition,
    because a control that dominates for one model and not another is itself
    the result.
    """
    from sift.shapley import decompose  # noqa: PLC0415

    keys = list(group_keys)
    mapping = coalition_mapping(values, keys)
    deltas = (
        values.drop_duplicates(keys).set_index(keys)["delta_empty"].to_dict()
    )

    if on_incomplete not in {"raise", "skip"}:
        raise ValueError(f"unknown on_incomplete {on_incomplete!r}")

    decompositions: dict[tuple, Decomposition] = {}
    records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for key, characteristic in mapping.items():
        delta_empty = float(deltas[key if len(key) > 1 else key[0]])
        unscored = sorted(
            sorted(coalition) for coalition, value in characteristic.items()
            if value != value
        )
        if unscored or delta_empty != delta_empty:
            detail = (
                f"{len(unscored)} of {len(characteristic)} coalitions unscored"
                if unscored
                else "Delta(empty) unscored"
            )
            if on_incomplete == "raise":
                raise ValueError(
                    f"group {key!r} cannot be decomposed: {detail}. The exact "
                    "Shapley value needs every coalition measured; pass "
                    'on_incomplete="skip" to exclude such groups and report them.'
                )
            record = dict(zip(keys, key, strict=True))
            record["n_unscored"] = len(unscored)
            record["reason"] = detail
            skipped.append(record)
            continue
        result = decompose(delta_empty, characteristic, groups, group_names)
        decompositions[key] = result
        record: dict[str, object] = dict(zip(keys, key, strict=True))
        record["delta_empty"] = delta_empty
        for control, phi in result.shapley.items():
            record[f"phi_{control}"] = phi
        record["residual"] = result.residual
        record["residual_share"] = (
            result.residual / delta_empty
            if abs(delta_empty) >= MIN_DELTA_FOR_SHARE
            else float("nan")
        )
        if result.owen is not None:
            for control, value in result.owen.items():
                record[f"owen_{control}"] = value
        if result.tier_shapley is not None:
            for tier, value in result.tier_shapley.items():
                record[f"tier_{tier}"] = value
        record["checks_passed"] = result.all_checks_passed
        records.append(record)

    return (
        decompositions,
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(skipped),
    )


def build_model_summary(
    decomposition_frame: pd.DataFrame,
    cut: int | None = None,
    controls: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarise the decomposition per model, averaging over seeds only.

    Parameters
    ----------
    decomposition_frame : pandas.DataFrame
        Tidy frame from :func:`decompose_groups`, one row per instance.
    cut : int, optional
        Restrict to one cut-point. All cuts are kept when omitted, and the
        result carries one row per cut and model.
    controls : sequence of str, optional
        Control names, fixing the column order.
    confidence : float, optional
        Nominal coverage of the reported interval.

    Returns
    -------
    pandas.DataFrame
        One row per (cut, model), with the mean of every attribution, the
        half-width of its interval across seeds, and a ``covers_zero`` flag per
        control and for the residual.

    Notes
    -----
    Seeds are the only axis averaged over. Cuts and models are kept apart
    because the design treats variation between them as a result rather than as
    noise to be smoothed away.

    The interval here is the spread across the five model seeds. It is not the
    bootstrap interval over test samples that the design asks for in the final
    table; two of the four models are deterministic given a seed, so their
    seed spread is exactly zero and only the bootstrap can supply an interval
    for them. ``covers_zero`` computed from this column will therefore be
    optimistic and must be recomputed from the bootstrap before publication.
    """
    from scipy import stats  # noqa: PLC0415

    frame = decomposition_frame if cut is None else decomposition_frame[
        decomposition_frame["cut"] == cut
    ]
    if frame.empty:
        raise ValueError(f"no decompositions for cut={cut!r}")

    if controls is None:
        controls = tuple(
            name[len("phi_"):] for name in frame.columns if name.startswith("phi_")
        )
    quantities = (
        ["delta_empty"]
        + [f"phi_{name}" for name in controls]
        + ["residual", "residual_share"]
        + [f"owen_{name}" for name in controls if f"owen_{name}" in frame.columns]
        + [name for name in frame.columns if name.startswith("tier_")]
    )
    quantities = [name for name in quantities if name in frame.columns]

    records: list[dict[str, object]] = []
    for (cut_value, model), block in frame.groupby(["cut", "model"], sort=True):
        record: dict[str, object] = {"cut": cut_value, "model": model, "n_seeds": len(block)}
        for name in quantities:
            series = block[name].astype(float)
            mean = float(series.mean())
            record[name] = mean
            if len(series) > 1 and series.std(ddof=1) > 0.0:
                half = float(
                    stats.t.ppf(0.5 + confidence / 2, len(series) - 1)
                    * series.std(ddof=1)
                    / len(series) ** 0.5
                )
            else:
                half = 0.0
            record[f"{name}_half"] = half
            if name.startswith("phi_") or name == "residual":
                record[f"{name}_covers_zero"] = bool(abs(mean) <= half)
        record["checks_passed"] = bool(block["checks_passed"].all())
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_gap_decomposition(
    metrics: pd.DataFrame,
    model: str,
    cut: int,
    metric_column: str = "macro_f1",
    temporal_design: str = "temporal",
    naive_design: str = "random",
    size_matched_design: str = "random_matched",
    fully_matched_design: str = "random_fully_matched",
    baseline_config_id: int = _BASELINE_CONFIG_ID,
) -> dict[str, float]:
    """Split the reported gap into a size, a class-composition and a remainder term.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Contents of ``results/metrics.parquet``. All four designs must be
        present at the baseline coalition.
    model : str
        Model to select.
    cut : int
        Cut-point to select.
    metric_column : str, optional
        Metric the gap is measured in.
    temporal_design : str, optional
        The arm every gap is measured against.
    naive_design, size_matched_design, fully_matched_design : str, optional
        The three random references, in increasing order of matching.
    baseline_config_id : int, optional
        The coalition the reference designs are run at.

    Returns
    -------
    dict of {str: float}
        Keys ``size``, ``class_composition``, ``remainder`` and
        ``delta_reported``.

    Raises
    ------
    ValueError
        If any of the four designs is missing at the baseline coalition.

    Notes
    -----
    With ``Delta(d) = M_d - M_temporal`` at the baseline coalition,

    .. code-block:: text

        size effect              = Delta(random) - Delta(random_matched)
        class-composition effect = Delta(random_matched) - Delta(random_fully_matched)
        remainder                = Delta(random_fully_matched)

    The three telescope onto ``Delta(random)``, the naively reported gap, so the
    identity is exact arithmetic rather than an empirical finding.

    Both correction terms are signed and are never clipped. The class term
    measured -0.033 at cut 2023 and +0.237 at cut 2015, and the size term goes
    negative wherever the temporal arm trains on more data than the naive random
    reference, which at cut 2023 it does. A clamp anywhere here would report a
    real effect as absent.

    Dropping ``random_matched`` raises rather than folding the size effect
    silently into the class term, which would misattribute one confound to
    another.
    """
    required = ["design", "config_id", "model", "cut", metric_column]
    _require_columns(metrics, required, "gap decomposition")

    selected = metrics[
        (metrics["model"] == model)
        & (metrics["cut"] == cut)
        & (metrics["config_id"] == baseline_config_id)
    ]
    if selected.empty:
        raise ValueError(
            f"no baseline rows for model={model!r} cut={cut!r} "
            f"config_id={baseline_config_id!r}"
        )

    scores: dict[str, float] = {}
    for design in (
        temporal_design,
        naive_design,
        size_matched_design,
        fully_matched_design,
    ):
        block = selected[selected["design"] == design][metric_column]
        if block.empty:
            raise ValueError(
                f"design {design!r} is missing at the baseline coalition for "
                f"model={model!r} cut={cut!r}; all four designs are required, and "
                "dropping one would fold its confound into a neighbouring term"
            )
        scores[design] = float(block.mean())

    temporal = scores[temporal_design]
    delta_naive = scores[naive_design] - temporal
    delta_size_matched = scores[size_matched_design] - temporal
    delta_fully_matched = scores[fully_matched_design] - temporal

    return {
        "size": delta_naive - delta_size_matched,
        "class_composition": delta_size_matched - delta_fully_matched,
        "remainder": delta_fully_matched,
        "delta_reported": delta_naive,
    }


def class_prior_reference(panel: pd.DataFrame, target: str = "ransomware_family") -> pd.Series:
    """Return the reference class distribution that control A1 weights towards.

    Parameters
    ----------
    panel : pandas.DataFrame
        The analysis panel, as built by ``sift.data.build_panel``.
    target : str, optional
        Label column.

    Returns
    -------
    pandas.Series
        Class distribution over the whole panel, summing to one, indexed by the
        integer label code used in ``predictions.parquet``.

    Notes
    -----
    This mirrors the single line in ``sift.controls.apply_controls`` that forms
    the reference. It depends on the panel alone and not on the coalition, the
    cut or the seed, which is what makes the A1 weight vector recomputable from
    the stored predictions instead of needing to be written to disk.

    The index is deliberately the *code*, not the family name. ``y_true`` in
    ``predictions.parquet`` holds positions in the sorted panel vocabulary, so a
    prior indexed by name would silently mismatch every lookup and yield weights
    that are wrong rather than absent. The vocabulary is taken from
    ``sift.experiment.label_categories`` so there is one definition of it.
    """
    from sift.experiment import label_categories  # noqa: PLC0415

    _require_columns(panel, [target], "class prior reference")
    categories = label_categories(panel, target)
    shares = panel[target].value_counts(normalize=True)
    return pd.Series(
        [float(shares[label]) for label in categories],
        index=pd.RangeIndex(len(categories)),
        name="share",
    )


def recompute_a1_weights(y_true, reference: pd.Series):
    """Recompute the control A1 sample weights for one scored window.

    Parameters
    ----------
    y_true : array-like
        True labels of the window, in the order the metrics were computed.
    reference : pandas.Series
        Reference class distribution from :func:`class_prior_reference`.

    Returns
    -------
    numpy.ndarray
        Weights averaging to one, aligned to ``y_true``.

    Notes
    -----
    The weight of a sample is ``p_reference(c) / p_window(c)`` for its class
    ``c``, renormalised so the effective sample size is preserved. It is a
    deterministic function of the window's label vector and the panel, and the
    label vector is stored in ``predictions.parquet``, so nothing about A1 needs
    to be persisted.

    The arithmetic is delegated to ``sift.controls`` rather than reimplemented.
    A second implementation here could drift from the one that produced the
    stored numbers and would then reproduce them only by luck;
    :func:`verify_a1_recomputation` exists to prove that it does not.
    """
    from sift.controls import _prior_weights  # noqa: PLC0415

    return _prior_weights(pd.Series(list(y_true)), reference)


def verify_a1_recomputation(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    reference: pd.Series,
    metric_function=None,
    metric_column: str = "macro_f1",
    sample_size: int | None = 40,
    tolerance: float = 1e-9,
    seed: int = 0,
) -> pd.DataFrame:
    """Check that recomputed A1 weights reproduce the stored metric.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Contents of ``results/predictions.parquet``.
    metrics : pandas.DataFrame
        Contents of ``results/metrics.parquet``.
    reference : pandas.Series
        Reference class distribution from :func:`class_prior_reference`.
    metric_function : callable, optional
        Defaults to ``sift.metrics.compute_metrics``.
    metric_column : str, optional
        Stored metric column to reproduce.
    sample_size : int, optional
        Number of A1-on fits to check. ``None`` checks every one.
    tolerance : float, optional
        Absolute tolerance for calling a pair equal.
    seed : int, optional
        Seed for choosing the sample of fits.

    Returns
    -------
    pandas.DataFrame
        One row per checked fit: ``fit_id``, ``stored``, ``recomputed``,
        ``abs_diff``, ``n_test`` and ``agrees``.

    Raises
    ------
    ValueError
        If no A1-on fits are present.

    Notes
    -----
    Recomputing a weight vector is convenient; reproducing the stored metric
    with it is what makes the recomputation trustworthy. If any row here
    disagrees, the weights are not the ones that produced the published numbers
    and the A1 attribution must not be reported until that is resolved.

    The first thing to suspect on a disagreement is the panel: the reference
    prior is the panel's class distribution, so a panel that differs from the
    one the run used gives every A1 cell a wrong weight. Run
    :func:`panel_matches_run` before this.
    """
    import numpy as np  # noqa: PLC0415

    if metric_function is None:
        from sift.metrics import compute_metrics as metric_function  # noqa: PLC0415

    _require_columns(metrics, ["fit_id", "a1_prior", metric_column], "A1 verification")
    weighted = metrics[metrics["a1_prior"]]
    if weighted.empty:
        raise ValueError("no A1-on fits present; nothing to verify")

    chosen = weighted
    if sample_size is not None and len(weighted) > sample_size:
        chosen = weighted.sample(n=sample_size, random_state=seed)

    wanted = set(chosen["fit_id"])
    blocks = predictions[predictions["fit_id"].isin(wanted)].groupby("fit_id", sort=False)
    stored_by_fit = chosen.set_index("fit_id")[metric_column].to_dict()

    records: list[dict[str, object]] = []
    for fit_id, block in blocks:
        y_true = block["y_true"].to_numpy()
        y_pred = block["y_pred"].to_numpy()
        weights = recompute_a1_weights(y_true, reference)
        computed = metric_function(y_true, y_pred, sample_weight=np.asarray(weights))
        stored = float(stored_by_fit[fit_id])
        difference = abs(float(computed[metric_column]) - stored)
        records.append(
            {
                "fit_id": fit_id,
                "stored": stored,
                "recomputed": float(computed[metric_column]),
                "abs_diff": difference,
                "n_test": int(len(block)),
                "agrees": difference <= tolerance,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("abs_diff", ascending=False)


def panel_matches_run(panel: pd.DataFrame, manifest: dict) -> dict[str, object]:
    """Check that a panel is the one the recorded run was produced from.

    Parameters
    ----------
    panel : pandas.DataFrame
        The analysis panel just built in this session.
    manifest : dict
        Contents of ``results/run_manifest.json``.

    Returns
    -------
    dict
        ``panel_fingerprint``, ``run_fingerprint``, ``matches`` and ``detail``.

    Notes
    -----
    Everything recomputed from the predictions depends on the panel: the A1
    weights are a function of its class distribution, and the common basis is a
    function of its sample identifiers. Rebuilding a different panel than the
    run used produces numbers that are internally consistent and quietly wrong,
    so the fingerprints are compared before any of it is believed.

    A smoke run on a synthetic panel writing to the same directory as a
    production run is the concrete way this goes wrong, and it is exactly what
    this check caught during development.
    """
    from sift import cache  # noqa: PLC0415

    fingerprint = str(cache.frame_fingerprint(panel))
    recorded = manifest.get("lattice", {}).get("panel_fingerprint")
    matches = recorded is not None and str(recorded) == fingerprint
    if recorded is None:
        detail = "the manifest records no panel fingerprint; cannot verify"
    elif matches:
        detail = f"panel matches the recorded run ({fingerprint[:16]})"
    else:
        detail = (
            f"panel fingerprint {fingerprint[:16]} does not match the run's "
            f"{str(recorded)[:16]}; the results in results/ were produced from a "
            "different panel, so recomputed weights and common-basis windows "
            "will not correspond to them"
        )
    return {
        "panel_fingerprint": fingerprint,
        "run_fingerprint": recorded,
        "matches": matches,
        "detail": detail,
    }


def compare_metric_decompositions(
    frames: Mapping[str, pd.DataFrame],
    controls: Sequence[str] | None = None,
    group_keys: Sequence[str] = GAP_GROUP_KEYS,
) -> pd.DataFrame:
    """Put the decomposition under two or more metrics side by side.

    Parameters
    ----------
    frames : Mapping of {str: pandas.DataFrame}
        Tidy decomposition frames from :func:`decompose_groups`, keyed by the
        metric they were built from.
    controls : sequence of str, optional
        Control names. Inferred from the first frame when omitted.
    group_keys : sequence of str, optional
        Keys identifying one instance of the game.

    Returns
    -------
    pandas.DataFrame
        One row per quantity and metric, with the mean attribution, its mean
        magnitude, and the ratio of that magnitude to the same quantity under
        the first metric given.

    Notes
    -----
    The choice of metric is itself an evaluation decision, and it can hide a
    confound. macro-F1 averages over classes and can neutralise a change in the
    class prior almost exactly, which is the one thing control A1 manipulates;
    accuracy is prior-sensitive and so reveals it. Reporting the decomposition
    under one metric alone would therefore let the metric decide which confounds
    appear to exist.

    The ``ratio_to_first`` column is the diagnostic: a control whose attribution
    is near zero under the first metric and materially larger under another is
    being masked by the first, and that masking is a result in its own right.
    """
    if not frames:
        raise ValueError("no decomposition frames to compare")

    names = list(frames)
    first = frames[names[0]]
    if controls is None:
        controls = tuple(
            name[len("phi_"):] for name in first.columns if name.startswith("phi_")
        )
    quantities = [f"phi_{name}" for name in controls] + ["residual", "delta_empty"]

    baseline = {
        quantity: float(first[quantity].abs().mean())
        for quantity in quantities
        if quantity in first.columns
    }

    records: list[dict[str, object]] = []
    for metric_name in names:
        frame = frames[metric_name]
        for quantity in quantities:
            if quantity not in frame.columns:
                continue
            magnitude = float(frame[quantity].abs().mean())
            reference = baseline.get(quantity, float("nan"))
            records.append(
                {
                    "quantity": quantity,
                    "metric": metric_name,
                    "mean": float(frame[quantity].mean()),
                    "mean_abs": magnitude,
                    "ratio_to_first": (
                        magnitude / reference if reference > 1e-12 else float("nan")
                    ),
                    "n_instances": int(len(frame)),
                }
            )
    return pd.DataFrame.from_records(records)
