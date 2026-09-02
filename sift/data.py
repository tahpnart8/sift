"""Loading the MLRan release and building the analysis panel.

The pipeline implemented here is fixed preprocessing. It is applied identically
to all 16 lattice configurations, so none of its steps is a control and none of
them appears in :class:`sift.config.ControlFlags`.

The steps, with the row counts they are asserted against:

1. Join the feature matrix to the metadata on ``sample_id``: 4880 rows.
2. Carry both temporal axes into the panel.
3. Drop samples activating no feature: 4742 rows.
4. Collapse exact duplicate feature vectors, keeping the earliest: 3605 rows.
5. Restrict to the analysis window on the primary axis (2012-2024): 3311 rows.
6. Apply the task filter: the family task keeps 1425 rows over 34 families.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy
import pandas

from sift.config import PanelSpec, panel_spec_for_c1
from sift.features import feature_columns
from sift.paths import MLRAN_DIR

__all__ = [
    "load_mlran",
    "build_panel",
    "assign_duplicate_groups",
    "N_RAW_SAMPLES",
    "N_RFE_FEATURES",
    # Opt-in five-control extension; see the C1 section at the foot of the file.
    "PanelVariants",
    "build_panel_variants",
]

#: Row count of the MLRan release, asserted after the join.
N_RAW_SAMPLES: int = 4880

#: Width of the published feature space after the authors' own RFE reduction.
N_RFE_FEATURES: int = 483

#: Metadata columns duplicated between the feature matrix and the metadata file.
#: They are dropped from the metadata side so the join produces no suffixes.
_DUPLICATED_ON_JOIN: tuple[str, ...] = ("sample_type", "family_label", "type_label")


def load_mlran(data_dir: Path = MLRAN_DIR) -> pandas.DataFrame:
    """Load the MLRan feature matrix and join it to the sample metadata.

    The published split into train and test files reflects the original authors'
    own temporal split. It is concatenated back into a single frame, because SIFT
    draws its own splits and inheriting theirs would fix the very design choice
    under study.

    Parameters
    ----------
    data_dir : pathlib.Path, default :data:`sift.paths.MLRAN_DIR`
        Directory holding the three MLRan CSV files.

    Returns
    -------
    pandas.DataFrame
        One row per sample, carrying the feature columns, the label columns and
        every metadata column including both temporal axes.

    Raises
    ------
    FileNotFoundError
        If any of the three input files is missing.
    ValueError
        If the join is not one to one, or the row or feature count departs from
        the published release.
    """
    required = (
        data_dir / "MLRan_X_train_RFE.csv",
        data_dir / "MLRan_X_test_RFE.csv",
        data_dir / "mlran_dataset_metadata.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("MLRan input files not found: " + ", ".join(missing))

    features = pandas.concat(
        [
            pandas.read_csv(required[0], low_memory=False),
            pandas.read_csv(required[1], low_memory=False),
        ],
        ignore_index=True,
    )
    metadata = pandas.read_csv(required[2], low_memory=False)
    metadata = metadata.drop(columns=[c for c in _DUPLICATED_ON_JOIN if c in metadata.columns])

    raw = features.merge(metadata, on="sample_id", how="left", validate="one_to_one")

    if len(raw) != N_RAW_SAMPLES:
        raise ValueError(f"expected {N_RAW_SAMPLES} rows after the join, got {len(raw)}")
    n_features = len(feature_columns(raw))
    if n_features != N_RFE_FEATURES:
        raise ValueError(f"expected {N_RFE_FEATURES} feature columns, got {n_features}")
    return raw


def assign_duplicate_groups(frame: pandas.DataFrame, columns: list[str]) -> pandas.Series:
    """Label each row with the identifier of its exact feature-vector group.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame holding binary feature columns.
    columns : list of str
        Feature columns defining the vector.

    Returns
    -------
    pandas.Series
        Integer group identifier aligned to ``frame.index``. Identifiers follow
        order of first appearance, which is stable for a given input ordering.
    """
    matrix = frame[columns].to_numpy(dtype=numpy.uint8)
    # Bit-packing makes the row key eight times smaller than one byte per feature
    # and keeps the comparison exact, since the features are binary.
    packed = numpy.packbits(matrix, axis=1)
    keys = numpy.array([row.tobytes() for row in packed], dtype=object)
    return pandas.Series(pandas.factorize(keys)[0], index=frame.index, name="grp_id")


def build_panel(
    raw: pandas.DataFrame,
    spec: PanelSpec,
) -> tuple[pandas.DataFrame, dict[str, int]]:
    """Apply the fixed preprocessing pipeline and return the analysis panel.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`load_mlran`, or a synthetic frame with the same columns.
    spec : sift.config.PanelSpec
        Preprocessing specification, including the task.

    Returns
    -------
    panel : pandas.DataFrame
        One row per retained sample. Carries ``grp_id`` and ``n_active`` in
        addition to the input columns, and a fresh :class:`~pandas.RangeIndex`.
    provenance : dict of str to int
        Row counts after each step, plus the composition of the result. Returned
        rather than printed, so a caller can assert on it.

    Raises
    ------
    ValueError
        If a required column is absent, or if the pipeline empties the panel.

    Notes
    -----
    Deduplication sorts on ``['first_submission_date', 'sample_id']`` with a
    stable algorithm. Two duplicate groups contain three samples each sharing the
    same earliest timestamp; under the default quicksort the surviving member of
    those groups depends on the library's internal ordering, which changes the
    panel and every split drawn from it. The secondary key and the stable
    algorithm together close that hole.
    """
    for column in ("sample_id", "sample_type", spec.primary_axis, spec.secondary_axis):
        if column not in raw.columns:
            raise ValueError(f"required column {column!r} absent from the input frame")
    if spec.task == "family" and "ransomware_family" not in raw.columns:
        raise ValueError("the family task requires a 'ransomware_family' column")

    columns = feature_columns(raw)
    panel = raw.copy()
    provenance: dict[str, int] = {"joined": len(panel)}

    panel["n_active"] = panel[columns].to_numpy(dtype=numpy.int32).sum(axis=1)
    if spec.drop_empty:
        panel = panel[panel["n_active"] > 0]
    provenance["after_drop_empty"] = len(panel)

    panel = panel.assign(grp_id=assign_duplicate_groups(panel, columns))
    provenance["n_unique_vectors"] = int(panel["grp_id"].nunique())
    if spec.dedup_exact:
        panel = panel.sort_values(
            ["first_submission_date", "sample_id"], kind="stable"
        ).drop_duplicates("grp_id", keep="first")
    provenance["after_dedup"] = len(panel)

    within_window = panel[spec.primary_axis].between(spec.year_min, spec.year_max)
    panel = panel[within_window]
    provenance["after_year_filter"] = len(panel)
    provenance["n_ransomware"] = int((panel["sample_type"] == 1).sum())
    provenance["n_goodware"] = int((panel["sample_type"] == 0).sum())
    provenance["n_families"] = (
        int(panel.loc[panel["sample_type"] == 1, "ransomware_family"].nunique())
        if "ransomware_family" in panel.columns
        else 0
    )

    if spec.task == "family":
        panel = panel[panel["sample_type"] == 1]
        sizes = panel["ransomware_family"].value_counts()
        kept = sizes.index[sizes >= spec.min_class_size]
        panel = panel[panel["ransomware_family"].isin(kept)]
    provenance["after_task_filter"] = len(panel)

    target = spec.target_column
    provenance["n_classes"] = int(panel[target].nunique()) if target in panel.columns else 0
    provenance["n_features"] = len(columns)

    if panel.empty:
        raise ValueError(f"preprocessing produced an empty panel; provenance={provenance}")

    panel = panel.sort_values("sample_id", kind="stable").reset_index(drop=True)

    # pandas propagates DataFrame.attrs across copy and slicing, so anything the
    # caller attached to the raw frame would ride into the panel and be read as a
    # property of the panel itself. A memoised content digest is the dangerous
    # case: it would key every fit on the identity of a different frame. A panel
    # is a new object, so it starts with no inherited metadata.
    panel.attrs.clear()
    return panel, provenance


# ===========================================================================
# C1: the two panel variants a five-control run needs
# ===========================================================================


@dataclass(frozen=True)
class PanelVariants:
    """The deduplicated and non-deduplicated panels, built from one raw frame.

    Two panels are built rather than one panel collapsed per configuration, and
    the choice is load bearing for two separate reasons.

    *Correctness of the ON state.* ``build_panel`` collapses duplicate groups
    **before** the analysis-window filter and before the task filter. A group
    whose earliest member falls outside the window is therefore collapsed onto
    that member and then dropped entirely. Collapsing per configuration, after
    the panel was built, would instead keep a later member of the same group,
    which is a different panel from the one every published result was measured
    on. Rebuilding through ``build_panel`` with ``dedup_exact=True`` reproduces
    the production panel exactly; ``test_c1_control.py`` pins that.

    *Correctness of the cache key.* ``cache.build_payload`` hashes both the
    ``PanelSpec`` and a content fingerprint of the realised panel. Two panels
    built here differ in both, so the two C1 states can never collide on one
    ``fit_id``. Had C1 been implemented as a per-configuration collapse of a
    single panel, both states would have shared one panel fingerprint and one
    ``dedup_exact`` value, and the key would have separated them only if a new
    field had been threaded into the payload -- which is exactly the failure R2
    found and E2 measured at 0.114.

    Attributes
    ----------
    spec : sift.config.PanelSpec
        Template specification. Its own ``dedup_exact`` is ignored; the two
        variants carry ``True`` and ``False`` respectively.
    with_dedup, without_dedup : pandas.DataFrame
        The two analysis panels. ``with_dedup`` is the panel of the four-control
        production run.
    provenance_with_dedup, provenance_without_dedup : dict of str to int
        Row counts after each preprocessing step, one per variant.
    """

    spec: PanelSpec
    with_dedup: pandas.DataFrame
    without_dedup: pandas.DataFrame
    provenance_with_dedup: dict[str, int]
    provenance_without_dedup: dict[str, int]

    def select(self, c1_dedup: bool) -> pandas.DataFrame:
        """Return the panel implied by the state of control C1.

        Parameters
        ----------
        c1_dedup : bool
            ``True`` selects the deduplicated panel, ``False`` the panel that
            keeps every duplicate.

        Returns
        -------
        pandas.DataFrame
        """
        return self.with_dedup if c1_dedup else self.without_dedup

    def spec_for(self, c1_dedup: bool) -> PanelSpec:
        """Return the :class:`~sift.config.PanelSpec` of one variant."""
        return panel_spec_for_c1(self.spec, c1_dedup)

    def provenance_for(self, c1_dedup: bool) -> dict[str, int]:
        """Return the provenance counters of one variant."""
        return (
            self.provenance_with_dedup if c1_dedup else self.provenance_without_dedup
        )


def build_panel_variants(raw: pandas.DataFrame, spec: PanelSpec) -> PanelVariants:
    """Build both C1 variants of the analysis panel.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`load_mlran`, or a synthetic frame with the same columns.
    spec : sift.config.PanelSpec
        Preprocessing specification. ``spec.dedup_exact`` is overridden for each
        variant, so its value in ``spec`` does not matter.

    Returns
    -------
    PanelVariants
        The two panels and their provenance.

    Raises
    ------
    ValueError
        Propagated from :func:`build_panel`.

    Notes
    -----
    This is the only entry point that a five-control run needs from this module.
    A four-control run calls :func:`build_panel` as before and never reaches
    here.
    """
    with_dedup, provenance_on = build_panel(raw, panel_spec_for_c1(spec, True))
    without_dedup, provenance_off = build_panel(raw, panel_spec_for_c1(spec, False))
    return PanelVariants(
        spec=spec,
        with_dedup=with_dedup,
        without_dedup=without_dedup,
        provenance_with_dedup=provenance_on,
        provenance_without_dedup=provenance_off,
    )
