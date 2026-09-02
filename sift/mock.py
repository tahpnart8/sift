"""Synthetic panels honouring the real column contract.

Tests and notebook smoke runs must not require the MLRan release to be present,
and must not take the several seconds a real load costs. The generators here
produce frames with the same columns, dtypes and invariants as the real ones, so
code exercised against them is exercised against the real contract.

The synthetic data carries real structure rather than noise: family membership
drives feature activation probabilities, so a classifier reaches a score
meaningfully above chance and a degenerate pipeline is detectable. It also
carries a deliberate discrepancy between the two temporal axes, so control B2 has
something to act on.
"""

from __future__ import annotations

import numpy
import pandas

from sift.seeding import make_rng

__all__ = ["PANEL_COLUMNS", "synthetic_panel", "synthetic_raw"]

#: Non-feature columns every panel carries, whether real or synthetic.
PANEL_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "sample_type",
    "family_label",
    "type_label",
    "ransomware_family",
    "ransomware_type",
    "source",
    "first_submission_date",
    "first_submission_date_year",
    "Year",
    "n_active",
    "grp_id",
)

_SECONDS_PER_YEAR: int = 31_556_952
_EPOCH_YEAR: int = 1970


def _year_to_epoch(years: numpy.ndarray, rng: numpy.random.Generator) -> numpy.ndarray:
    """Convert years to plausible Unix timestamps scattered within each year."""
    offset = rng.integers(0, _SECONDS_PER_YEAR, size=years.shape)
    return (years - _EPOCH_YEAR) * _SECONDS_PER_YEAR + offset


def synthetic_panel(
    n_samples: int = 800,
    n_features: int = 50,
    n_families: int = 8,
    seed: int = 0,
) -> pandas.DataFrame:
    """Generate a ready-to-use analysis panel.

    The result satisfies the invariants :func:`sift.data.build_panel` guarantees:
    no sample activates zero features, no feature vector repeats, and both
    temporal axes lie inside the analysis window.

    Parameters
    ----------
    n_samples : int, default 800
        Total rows, split between goodware and ransomware.
    n_features : int, default 50
        Number of binary feature columns, named by numeric identifier as in the
        real release.
    n_families : int, default 8
        Number of ransomware families. Each receives enough samples to survive
        the default ``min_class_size`` filter.
    seed : int, default 0
        Seed for the generator.

    Returns
    -------
    pandas.DataFrame
        A panel carrying :data:`PANEL_COLUMNS` and ``n_features`` feature
        columns, indexed by a fresh :class:`~pandas.RangeIndex`.

    Raises
    ------
    ValueError
        If the requested shape cannot yield families above the size filter.
    """
    if n_families < 2:
        raise ValueError(f"n_families must be at least 2, got {n_families}")
    if n_samples < 4 * n_families:
        raise ValueError(
            f"n_samples={n_samples} is too small to populate {n_families} families"
        )
    if n_features < 8:
        raise ValueError(f"n_features must be at least 8, got {n_features}")

    rng = make_rng(seed)
    n_goodware = n_samples // 3
    n_ransomware = n_samples - n_goodware

    family_index = numpy.sort(rng.integers(0, n_families, size=n_ransomware))
    classes = numpy.concatenate([numpy.full(n_goodware, -1), family_index])

    # Each class activates its own overlapping block of features, so the classes
    # are separable but not trivially so.
    profiles = rng.uniform(0.05, 0.25, size=(n_families + 1, n_features))
    for class_position in range(n_families + 1):
        block = rng.choice(n_features, size=max(3, n_features // 6), replace=False)
        profiles[class_position, block] = rng.uniform(0.65, 0.95, size=block.size)

    probabilities = profiles[classes + 1]
    matrix = (rng.random((n_samples, n_features)) < probabilities).astype(numpy.int8)
    matrix = _repair_invariants(matrix, rng)

    primary_year = rng.integers(2012, 2024, size=n_samples)
    # The compile timestamp disagrees with the submission date for roughly a third
    # of samples, some of it pushed implausibly far back, exactly as in MLRan.
    drift = rng.choice([0, 0, 0, -1, -2, 1, -40], size=n_samples)
    secondary_year = numpy.clip(primary_year + drift, 1970, 2024)

    features = pandas.DataFrame(matrix, columns=[str(i) for i in range(n_features)])
    panel = pandas.DataFrame(
        {
            "sample_id": numpy.arange(10_001, 10_001 + n_samples, dtype=numpy.int64),
            "sample_type": (classes >= 0).astype(numpy.int64),
            "family_label": (classes + 1).astype(numpy.int64),
            "type_label": numpy.where(classes < 0, 0, classes % 4 + 1).astype(numpy.int64),
            "ransomware_family": [
                "goodware" if value < 0 else f"family_{value:02d}" for value in classes
            ],
            "ransomware_type": numpy.where(classes < 0, "goodware", "crypto"),
            "source": numpy.where(classes < 0, "software_informer", "curated"),
            "first_submission_date": _year_to_epoch(primary_year, rng),
            "first_submission_date_year": primary_year.astype(numpy.int64),
            "Year": secondary_year.astype(numpy.int64),
            "n_active": matrix.sum(axis=1).astype(numpy.int64),
            "grp_id": numpy.arange(n_samples, dtype=numpy.int64),
        }
    )
    return pandas.concat([panel, features], axis=1)


def synthetic_raw(
    n_samples: int = 800,
    n_features: int = 50,
    n_families: int = 8,
    seed: int = 0,
    n_empty: int = 12,
    n_duplicates: int = 20,
) -> pandas.DataFrame:
    """Generate a raw frame that still needs preprocessing.

    Unlike :func:`synthetic_panel` this deliberately violates the panel
    invariants, so :func:`sift.data.build_panel` has empty rows to drop and
    duplicate vectors to collapse. It also plants a timestamp tie inside one
    duplicate group, reproducing the condition that makes an unstable sort change
    the panel.

    Parameters
    ----------
    n_samples : int, default 800
        Rows before the empty and duplicate rows are appended.
    n_features : int, default 50
        Number of binary feature columns.
    n_families : int, default 8
        Number of ransomware families.
    seed : int, default 0
        Seed for the generator.
    n_empty : int, default 12
        Number of all-zero rows to append.
    n_duplicates : int, default 20
        Number of duplicated feature vectors to append.

    Returns
    -------
    pandas.DataFrame
        A frame shaped like the output of :func:`sift.data.load_mlran`.
    """
    rng = make_rng(seed + 1)
    base = synthetic_panel(n_samples, n_features, n_families, seed)
    columns = [str(i) for i in range(n_features)]

    empties = base.iloc[:n_empty].copy()
    empties[columns] = 0
    empties["sample_type"] = 0
    empties["ransomware_family"] = "goodware"

    duplicates = base.iloc[rng.integers(0, len(base), size=n_duplicates)].copy()
    # Later resubmissions of an identical binary: the vector repeats, the date moves.
    duplicates["first_submission_date_year"] = numpy.clip(
        duplicates["first_submission_date_year"].to_numpy() + 3, 2012, 2023
    )
    duplicates["first_submission_date"] = _year_to_epoch(
        duplicates["first_submission_date_year"].to_numpy(), rng
    )

    tied = base.iloc[[0, 1]].copy()
    tied[columns] = base.iloc[[2, 2]][columns].to_numpy()
    tied["first_submission_date"] = int(base.iloc[2]["first_submission_date"])
    tied["first_submission_date_year"] = int(base.iloc[2]["first_submission_date_year"])

    raw = pandas.concat([base, empties, duplicates, tied], ignore_index=True)
    raw["sample_id"] = numpy.arange(10_001, 10_001 + len(raw), dtype=numpy.int64)
    return raw.drop(columns=["n_active", "grp_id"])


def _repair_invariants(
    matrix: numpy.ndarray,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Force every row to be non-empty and every row to be unique.

    Parameters
    ----------
    matrix : numpy.ndarray
        Binary matrix of shape ``(n_samples, n_features)``.
    rng : numpy.random.Generator
        Generator used to place the repair bits.

    Returns
    -------
    numpy.ndarray
        The repaired matrix, modified in place and returned for convenience.
    """
    n_samples, n_features = matrix.shape

    empty_rows = numpy.flatnonzero(matrix.sum(axis=1) == 0)
    if empty_rows.size:
        matrix[empty_rows, rng.integers(0, n_features, size=empty_rows.size)] = 1

    # A distinct low-order bit pattern per row guarantees uniqueness without
    # disturbing the class-dependent structure carried by the remaining columns.
    width = max(1, int(numpy.ceil(numpy.log2(max(n_samples, 2)))))
    width = min(width, n_features)
    identifiers = numpy.arange(n_samples, dtype=numpy.int64)
    for bit in range(width):
        matrix[:, bit] = (identifiers >> bit & 1).astype(numpy.int8)

    still_empty = numpy.flatnonzero(matrix.sum(axis=1) == 0)
    if still_empty.size:
        matrix[still_empty, width % n_features] = 1
    return matrix
