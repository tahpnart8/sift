"""Feature columns, identifier-to-name mapping and mutual-information selection.

The MLRan release names its feature columns by numeric identifier, so a panel
column is called ``'5'`` rather than anything readable. The readable names live
in a separate JSON side file. Any interpretable output must join the two, which
is why :func:`load_feature_name_map` belongs in the library rather than in a
notebook cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas
from sklearn.feature_selection import mutual_info_classif

from sift.paths import MLRAN_DIR
from sift.seeding import derive_seed

__all__ = [
    "METADATA_COLUMNS",
    "feature_columns",
    "load_feature_name_map",
    "rank_features",
    "select_features",
]

#: Columns carried by the panel that are not features. Everything else in a panel
#: is a behavioural indicator.
METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "sample_id",
        "sample_type",
        "family_label",
        "type_label",
        "ransomware_family",
        "ransomware_type",
        "source",
        "sha256",
        "sha1",
        "md5",
        "extension",
        "detections",
        "timestamp",
        "new_timestamp",
        "old_timestamp",
        "Year",
        "first_submission_date",
        "first_submission_date_converted",
        "first_submission_date_year",
        "grp_id",
        "n_active",
    }
)


def feature_columns(panel: pandas.DataFrame) -> list[str]:
    """Return the feature columns of a panel, in their stored order.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel or raw joined frame.

    Returns
    -------
    list of str
        Column names that are not metadata. Order follows the frame, which is
        stable for a given input file, so downstream matrices are reproducible.
    """
    return [column for column in panel.columns if column not in METADATA_COLUMNS]


def load_feature_name_map(data_dir: Path = MLRAN_DIR) -> dict[str, str]:
    """Load the mapping from feature identifier to readable feature name.

    Parameters
    ----------
    data_dir : pathlib.Path, default :data:`sift.paths.MLRAN_DIR`
        Directory holding ``RFE_selected_feature_names_dic.json``.

    Returns
    -------
    dict of str to str
        Identifier as it appears in the panel columns, mapped to the readable
        name. Identifiers absent from the file are simply absent from the map.

    Raises
    ------
    FileNotFoundError
        If the side file is missing.
    """
    path = data_dir / "RFE_selected_feature_names_dic.json"
    if not path.exists():
        raise FileNotFoundError(f"feature name map not found at {path}")
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key): str(value) for key, value in raw.items()}


def rank_features(
    frame: pandas.DataFrame,
    columns: list[str],
    target: str,
    seed: int,
) -> pandas.DataFrame:
    """Rank features by mutual information with the target.

    Ties are broken by column name so that the ranking is a total order. Without
    a deterministic tie-break the selected set would depend on the sort algorithm,
    and control B1 would be measuring sort behaviour alongside leakage.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows the selector is fitted on. Which rows these are is precisely what
        control B1 varies.
    columns : list of str
        Candidate feature columns.
    target : str
        Label column.
    seed : int
        Seed forwarded to :func:`sklearn.feature_selection.mutual_info_classif`.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature`` and ``mutual_information``, sorted by descending
        score then ascending name.
    """
    scores = mutual_info_classif(
        frame[columns].to_numpy(),
        frame[target].to_numpy(),
        discrete_features=True,
        random_state=derive_seed(seed, "mutual_information"),
    )
    ranked = pandas.DataFrame({"feature": columns, "mutual_information": scores})
    return ranked.sort_values(
        ["mutual_information", "feature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def select_features(
    frame: pandas.DataFrame,
    columns: list[str],
    target: str,
    n_features: int,
    seed: int,
) -> list[str]:
    """Select the highest mutual-information features.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows the selector is fitted on.
    columns : list of str
        Candidate feature columns.
    target : str
        Label column.
    n_features : int
        Number retained. Values at or above ``len(columns)`` retain everything.
    seed : int
        Seed forwarded to the mutual-information estimator.

    Returns
    -------
    list of str
        Selected column names, returned in the panel column order rather than in
        score order so that the design matrix layout does not vary between
        lattice cells.
    """
    if n_features >= len(columns):
        return list(columns)
    ranked = rank_features(frame, columns, target, seed)
    chosen = set(ranked["feature"].iloc[:n_features])
    return [column for column in columns if column in chosen]
