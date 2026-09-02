"""Filesystem anchors for the SIFT project.

This module contains no logic beyond path resolution. It exists so that no other
module has to guess where the project root is, and so that no module ever needs
``sys.path`` manipulation: the package is installed editable and locates its
sibling data directories relative to its own file location.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "MLRAN_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "NOTEBOOKS_DIR",
    "ensure_dirs",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

MLRAN_DIR: Path = PROJECT_ROOT / "mlran" / "6_experiments" / "FS_MLRan_Datasets"

DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

MLRAN_FILES: tuple[str, ...] = (
    "MLRan_X_train_RFE.csv",
    "MLRan_X_test_RFE.csv",
    "mlran_dataset_metadata.csv",
)


def ensure_dirs() -> None:
    """Create the output directories if they do not already exist.

    Only output directories are created. A missing input directory is an error
    that must surface at load time rather than being silently masked.
    """
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
