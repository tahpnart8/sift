"""SIFT: Shift Isolation Framework for Temporal evaluation.

SIFT decomposes the performance drop between a random split and a temporal split
into evaluation confounds and genuine concept drift. Four controls form a
:math:`2^4` ablation lattice whose cells are aggregated with exact Shapley values.

Notebooks orchestrate; they do not define. Import the verbs from here and keep
the logic in the package::

    from sift import build_panel, load_mlran, PanelSpec

    raw = load_mlran()
    panel, provenance = build_panel(raw, PanelSpec(task="family"))

Modules import strictly downward: :mod:`sift.config` imports nothing from the
package, and :mod:`sift.experiment` may import anything. No module writes files
except :mod:`sift.reporting`.
"""

from __future__ import annotations

from sift.config import (
    C1_NAME,
    CONTROL_NAMES,
    CONTROL_NAMES_C1,
    LATTICE_CELLS,
    LATTICE_CELLS_C1,
    SCHEMA_VERSION,
    TIER_GROUPS_C1,
    ControlFlags,
    ExperimentConfig,
    ExtendedControlFlags,
    LatticeCell,
    PanelSpec,
    SplitSpec,
    config_for_cell_c1,
    extended_flags_of,
    lattice_cell,
    lattice_cell_c1,
    panel_spec_for_c1,
)
from sift.controls import (
    CONTROL_ORDER,
    CONTROL_ORDER_C1,
    ControlledData,
    apply_controls,
    apply_controls_c1,
    match_class_prior,
    resolve_feature_set,
    resolve_time_axis,
    restrict_label_space,
)
from sift.data import (
    N_RAW_SAMPLES,
    N_RFE_FEATURES,
    PanelVariants,
    build_panel,
    build_panel_variants,
    load_mlran,
)
from sift.features import (
    METADATA_COLUMNS,
    feature_columns,
    load_feature_name_map,
    rank_features,
    select_features,
)
from sift.metrics import (
    AUT_CUT_YEARS,
    AUT_TEST_WINDOW,
    METRIC_NAMES,
    assert_equal_disjoint_slots,
    aut,
    aut_label,
    bootstrap_ci,
    bootstrap_distribution,
    compute_metrics,
)
from sift.mock import synthetic_panel, synthetic_raw
from sift.models import MODEL_NAMES, build_model
from sift.paths import (
    DATA_DIR,
    FIGURES_DIR,
    MLRAN_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    ensure_dirs,
)
from sift.seeding import derive_seed, make_rng
from sift.splits import (
    assert_disjoint,
    assert_temporal_consistency,
    make_split,
    matched_sizes,
    random_matched_split,
    random_split,
    temporal_split,
    temporal_test_classes,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "CONTROL_NAMES",
    "LATTICE_CELLS",
    "LatticeCell",
    "lattice_cell",
    "SCHEMA_VERSION",
    "ControlFlags",
    "ExperimentConfig",
    "PanelSpec",
    "SplitSpec",
    # paths
    "PROJECT_ROOT",
    "MLRAN_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "ensure_dirs",
    # seeding
    "derive_seed",
    "make_rng",
    # data
    "load_mlran",
    "build_panel",
    "N_RAW_SAMPLES",
    "N_RFE_FEATURES",
    # features
    "METADATA_COLUMNS",
    "feature_columns",
    "load_feature_name_map",
    "rank_features",
    "select_features",
    # splits
    "make_split",
    "random_split",
    "random_matched_split",
    "matched_sizes",
    "temporal_split",
    "temporal_test_classes",
    "assert_temporal_consistency",
    "assert_disjoint",
    # controls
    "CONTROL_ORDER",
    "ControlledData",
    "apply_controls",
    "resolve_time_axis",
    "restrict_label_space",
    "match_class_prior",
    "resolve_feature_set",
    # models and metrics
    "MODEL_NAMES",
    "build_model",
    "METRIC_NAMES",
    "compute_metrics",
    "bootstrap_distribution",
    "bootstrap_ci",
    "AUT_CUT_YEARS",
    "AUT_TEST_WINDOW",
    "assert_equal_disjoint_slots",
    "aut",
    "aut_label",
    # mock
    "synthetic_panel",
    "synthetic_raw",
    # C1: opt-in five-control extension. The four-control names above are
    # unchanged and remain the default path.
    "C1_NAME",
    "CONTROL_NAMES_C1",
    "CONTROL_ORDER_C1",
    "LATTICE_CELLS_C1",
    "TIER_GROUPS_C1",
    "ExtendedControlFlags",
    "PanelVariants",
    "apply_controls_c1",
    "build_panel_variants",
    "config_for_cell_c1",
    "extended_flags_of",
    "lattice_cell_c1",
    "panel_spec_for_c1",
]
