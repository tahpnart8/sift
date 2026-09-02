"""C1, exact-duplicate removal, as an opt-in fifth control.

Three things are asserted here and they pull in opposite directions, which is
why they live in one file.

*The four-control path must not move.*  ``results/metrics.parquet`` was written
with a four-flag ``config_id`` and is read by ``sift.shapley``,
``sift.reporting`` and the notebooks.  The first section pins that encoding, the
width of ``ControlFlags``, the sixteen-cell enumeration and the metrics schema,
so any future widening of the four-control types fails here rather than in a
silently renumbered table.

*C1 must come first.*  The four existing controls rearrange a fixed set of
samples; C1 decides which samples exist at all.  The second section pins the
order and, more usefully, pins the two consequences that make the order real:
duplicates are collapsed before the split is drawn, and they are collapsed
before the analysis-window filter, which is why the ON state must be rebuilt
through ``build_panel`` rather than collapsed out of a finished panel.

*The extension must be measurable.*  The last sections check that the two C1
states cannot collide in the fit cache, that the ON state reproduces the
existing four-control fit exactly, and that the Shapley machinery carries five
players.
"""

from __future__ import annotations

import dataclasses
from itertools import combinations

import pandas as pd
import pytest

import sift.cache as cache_module
from sift.config import (
    CONTROL_NAMES,
    CONTROL_NAMES_C1,
    LATTICE_CELLS,
    LATTICE_CELLS_C1,
    TIER_GROUPS_C1,
    ControlFlags,
    ExperimentConfig,
    ExtendedControlFlags,
    PanelSpec,
    SplitSpec,
    config_for_cell_c1,
    extended_flags_of,
    lattice_cell_c1,
    panel_spec_for_c1,
)
from sift.controls import CONTROL_ORDER, CONTROL_ORDER_C1, apply_controls_c1
from sift.data import build_panel, build_panel_variants
from sift.experiment import (
    BASELINE_CONFIG_IDS_C1,
    JOBS_PER_GROUP_C1,
    METRICS_COLUMN_ORDER,
    METRICS_COLUMN_ORDER_C1,
    METRICS_SCHEMA,
    N_COALITIONS,
    N_COALITIONS_C1,
    assert_lattice_complete_c1,
    build_jobs,
    build_jobs_c1,
    run_cell_c1,
)
from sift.features import feature_columns
from sift.mock import synthetic_raw
from sift.shapley import DEFAULT_CONTROL_COLUMNS, decompose, exact_shapley, gap_values

# The four-flag encoding exactly as ``results/metrics.parquet`` carries it.
# Written out as a literal rather than recomputed from ``CONTROL_NAMES``, so a
# reordering of that tuple is caught instead of being followed.
FOUR_CONTROL_CONFIG_IDS: dict[int, frozenset[str]] = {
    0: frozenset(),
    1: frozenset({"a1_prior"}),
    2: frozenset({"a2_labels"}),
    3: frozenset({"a1_prior", "a2_labels"}),
    4: frozenset({"b1_fs"}),
    5: frozenset({"a1_prior", "b1_fs"}),
    6: frozenset({"a2_labels", "b1_fs"}),
    7: frozenset({"a1_prior", "a2_labels", "b1_fs"}),
    8: frozenset({"b2_axis"}),
    9: frozenset({"a1_prior", "b2_axis"}),
    10: frozenset({"a2_labels", "b2_axis"}),
    11: frozenset({"a1_prior", "a2_labels", "b2_axis"}),
    12: frozenset({"b1_fs", "b2_axis"}),
    13: frozenset({"a1_prior", "b1_fs", "b2_axis"}),
    14: frozenset({"a2_labels", "b1_fs", "b2_axis"}),
    15: frozenset({"a1_prior", "a2_labels", "b1_fs", "b2_axis"}),
}

PRODUCTION_MODELS = ("logreg", "random_forest", "lightgbm", "mlp")
PRODUCTION_SEEDS = (0, 1, 2, 3, 4)
PRODUCTION_CUTS = (2015, 2017, 2019, 2021, 2023)


def _base_config(cut: int = 2019, n_features: int = 20) -> ExperimentConfig:
    return ExperimentConfig(
        panel=PanelSpec(min_class_size=1),
        split=SplitSpec(design="temporal", cut_year=cut),
        flags=ControlFlags(),
        model_name="logreg",
        seed=0,
        n_features=n_features,
    )


@pytest.fixture(scope="module")
def variants():
    """Both C1 panel variants, built from one synthetic raw frame."""
    raw = synthetic_raw(n_samples=800, n_features=30, n_families=8, seed=0)
    return build_panel_variants(raw, PanelSpec(min_class_size=1))


# --------------------------------------------------------------------------
# Backward compatibility: the four-control path must be byte-for-byte intact
# --------------------------------------------------------------------------
def test_four_control_config_id_mapping_is_unchanged() -> None:
    """The pin required by the brief: ``config_id`` still means what it meant.

    ``results/metrics.parquet`` is being written with this encoding while C1 is
    developed. Both directions are checked, so neither a renumbering nor a
    non-invertible change slips through.
    """
    for config_id, active in FOUR_CONTROL_CONFIG_IDS.items():
        flags = ControlFlags.from_index(config_id)
        assert flags.active() == active, f"config_id {config_id} changed meaning"
        assert flags.to_index() == config_id
    assert CONTROL_NAMES == ("a1_prior", "a2_labels", "b1_fs", "b2_axis")


def test_control_flags_still_has_exactly_four_fields() -> None:
    """Adding ``c1_dedup`` to ``ControlFlags`` would renumber the lattice.

    A fifth field would make ``to_index`` return values up to 31 for the same
    coalitions, so the published ``config_id`` column would silently change
    meaning. The fifth control lives in ``ExtendedControlFlags`` instead.
    """
    names = tuple(field.name for field in dataclasses.fields(ControlFlags))
    assert names == ("a1_prior", "a2_labels", "b1_fs", "b2_axis")
    with pytest.raises(ValueError):
        ControlFlags.from_index(16)


def test_four_control_lattice_still_has_sixteen_cells() -> None:
    assert N_COALITIONS == 16
    assert len(LATTICE_CELLS) == 16
    for cell in LATTICE_CELLS:
        assert cell.flags == ControlFlags.from_index(cell.config_id)
        assert cell.active == FOUR_CONTROL_CONFIG_IDS[cell.config_id]


def test_four_control_metrics_schema_is_unchanged() -> None:
    """The production table gains no column and loses none."""
    assert METRICS_COLUMN_ORDER == (
        "fit_id",
        "run_id",
        "config_id",
        "a1_prior",
        "a2_labels",
        "b1_fs",
        "b2_axis",
        "design",
        "role",
        "cut",
        "seed",
        "model",
        "n_train",
        "n_test",
        "k_train",
        "k_test",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "accuracy",
        "fit_seconds",
    )
    assert "c1_dedup" not in METRICS_SCHEMA.names


def test_four_control_job_enumeration_is_unchanged() -> None:
    """34 fits per (cut, model, seed), 3400 over the production grid."""
    jobs = build_jobs(
        _base_config(), PRODUCTION_MODELS, PRODUCTION_SEEDS, PRODUCTION_CUTS
    )
    assert len(jobs) == 3400
    groups = len(PRODUCTION_CUTS) * len(PRODUCTION_MODELS) * len(PRODUCTION_SEEDS)
    assert len(jobs) == 34 * groups
    assert all(cfg.panel.dedup_exact for cfg in jobs), (
        "the four-control run must keep deduplicating; C1 is opt-in"
    )


def test_default_panel_spec_still_deduplicates() -> None:
    assert PanelSpec().dedup_exact is True


def test_build_panel_default_is_exactly_the_c1_on_variant(variants) -> None:
    """The ON state is the production panel, not a reconstruction of it."""
    spec = PanelSpec(min_class_size=1)
    raw = synthetic_raw(n_samples=800, n_features=30, n_families=8, seed=0)
    panel, _ = build_panel(raw, spec)
    pd.testing.assert_frame_equal(panel, variants.with_dedup)
    assert cache_module.frame_fingerprint(panel) == cache_module.frame_fingerprint(
        variants.with_dedup
    )


# --------------------------------------------------------------------------
# The ordering problem
# --------------------------------------------------------------------------
def test_control_order_c1_puts_deduplication_first() -> None:
    """C1 decides which samples exist, so it precedes all four."""
    assert CONTROL_ORDER_C1 == ("c1_dedup", "b2_axis", "a2_labels", "a1_prior", "b1_fs")
    assert CONTROL_ORDER_C1[0] == "c1_dedup"
    assert CONTROL_ORDER_C1[1:] == CONTROL_ORDER, (
        "the relative order of the four existing controls must not change"
    )


def _with_straddling_duplicates(raw: pd.DataFrame, n_pairs: int = 6) -> pd.DataFrame:
    """Plant duplicate pairs whose two members fall on opposite sides of 2019.

    ``synthetic_raw`` plants duplicates but does not guarantee that any group
    crosses a given cut, and a group that does not cross proves nothing about
    C1. These do: the ancestor sits in 2015 and its exact copy in 2021, on both
    temporal axes, so a 2019 cut with a three-year window puts one member in
    each half.
    """
    ransomware = raw[raw["sample_type"] == 1]
    next_id = int(raw["sample_id"].max()) + 1
    late = int(raw["first_submission_date"].max()) + 1
    planted = []
    for offset in range(n_pairs):
        ancestor = ransomware.iloc[offset].copy()
        ancestor["sample_id"] = next_id + 2 * offset
        ancestor["first_submission_date"] = offset
        ancestor["first_submission_date_year"] = 2015
        ancestor["Year"] = 2015

        clone = ancestor.copy()
        clone["sample_id"] = next_id + 2 * offset + 1
        clone["first_submission_date"] = late + offset
        clone["first_submission_date_year"] = 2021
        clone["Year"] = 2021
        planted.extend([ancestor, clone])
    return pd.concat([raw, pd.DataFrame(planted)], ignore_index=True)


def test_deduplication_happens_before_the_split_is_drawn() -> None:
    """Ordering, stated as a consequence rather than as a tuple.

    With C1 on, no duplicate group can straddle the split, because the group no
    longer exists by the time B2 draws the boundary. With C1 off it can, and
    that is precisely the flawed protocol: a test row that is a byte-identical
    copy of a training row.
    """
    raw = _with_straddling_duplicates(
        synthetic_raw(n_samples=800, n_features=30, n_families=8, seed=0)
    )
    variants = build_panel_variants(raw, PanelSpec(min_class_size=1))
    base = _base_config()

    on = apply_controls_c1(variants, base, ExtendedControlFlags(c1_dedup=True))
    shared_on = set(on.train["grp_id"]) & set(on.test["grp_id"])
    assert not shared_on, "a duplicate group straddled the split under C1 on"

    off = apply_controls_c1(variants, base, ExtendedControlFlags(c1_dedup=False))
    shared_off = set(off.train["grp_id"]) & set(off.test["grp_id"])
    assert shared_off, (
        "the planted duplicates did not cross the cut, so this test proves "
        "nothing about C1 off"
    )
    assert len(off.test) > len(on.test)


def test_c1_on_must_be_rebuilt_not_collapsed_out_of_the_off_panel() -> None:
    """Why two panels are built rather than one panel collapsed per cell.

    ``build_panel`` collapses duplicate groups *before* the analysis-window
    filter. A group whose earliest member predates the window is collapsed onto
    that member and then dropped whole. Collapsing after the fact, from the
    panel that kept duplicates, would instead keep a later member: a sample the
    production panel has never contained.
    """
    spec = PanelSpec(min_class_size=1, year_min=2012, year_max=2024)
    raw = synthetic_raw(n_samples=400, n_features=20, n_families=6, seed=1)
    columns = feature_columns(raw)

    ancestor = raw[raw["sample_type"] == 1].iloc[0].copy()
    ancestor["sample_id"] = int(raw["sample_id"].max()) + 1
    ancestor["first_submission_date"] = 0
    ancestor["first_submission_date_year"] = 2005

    clone = ancestor.copy()
    clone["sample_id"] = int(raw["sample_id"].max()) + 2
    clone["first_submission_date"] = int(raw["first_submission_date"].max()) + 1
    clone["first_submission_date_year"] = 2018

    planted = pd.concat(
        [raw, pd.DataFrame([ancestor, clone])], ignore_index=True
    )
    assert planted.loc[planted["sample_id"] == ancestor["sample_id"], columns].to_numpy(
    ).tolist() == planted.loc[
        planted["sample_id"] == clone["sample_id"], columns
    ].to_numpy().tolist(), "the plant is not an exact duplicate"

    variants = build_panel_variants(planted, spec)
    assert clone["sample_id"] not in set(variants.with_dedup["sample_id"]), (
        "the clone survived deduplication; its group's earliest member is out "
        "of window, so the whole group must vanish"
    )
    assert clone["sample_id"] in set(variants.without_dedup["sample_id"])


# --------------------------------------------------------------------------
# The extended flag type
# --------------------------------------------------------------------------
def test_extended_index_round_trips_over_all_thirty_two_cells() -> None:
    assert N_COALITIONS_C1 == 32
    assert CONTROL_NAMES_C1 == CONTROL_NAMES + ("c1_dedup",)
    seen = set()
    for index in range(32):
        flags = ExtendedControlFlags.from_index(index)
        assert flags.to_index() == index
        seen.add(flags)
    assert len(seen) == 32
    with pytest.raises(ValueError):
        ExtendedControlFlags.from_index(32)


def test_extended_index_low_bits_are_the_four_control_index() -> None:
    """``config_id % 16`` recovers the published encoding, for every cell."""
    for index in range(32):
        flags = ExtendedControlFlags.from_index(index)
        assert flags.base().to_index() == index % 16
        assert flags.base().active() == FOUR_CONTROL_CONFIG_IDS[index % 16]
        assert flags.c1_dedup is bool(index >= 16)


def test_from_flags_lifts_a_production_cell_to_the_upper_half() -> None:
    for index in range(16):
        lifted = ExtendedControlFlags.from_flags(
            ControlFlags.from_index(index), c1_dedup=True
        )
        assert lifted.to_index() == index + 16
        assert lifted.base() == ControlFlags.from_index(index)


def test_extended_lattice_cells_agree_with_the_four_control_readings() -> None:
    assert len(LATTICE_CELLS_C1) == 32
    for index in range(32):
        cell = lattice_cell_c1(index)
        assert cell.config_id == index
        assert cell.flags == ExtendedControlFlags.from_index(index)
        assert LATTICE_CELLS[index % 16].description in cell.description
    with pytest.raises(ValueError):
        lattice_cell_c1(32)


def test_c1_is_classified_protocol_side() -> None:
    """Leaving duplicates in is an experimental error, not a property of the stream."""
    data_tier, protocol_tier = TIER_GROUPS_C1
    assert "c1_dedup" in protocol_tier
    assert "c1_dedup" not in data_tier
    assert set(data_tier) == {"a1_prior", "a2_labels"}
    assert set(data_tier) | set(protocol_tier) == set(CONTROL_NAMES_C1)
    assert not set(data_tier) & set(protocol_tier)


# --------------------------------------------------------------------------
# The fit cache
# --------------------------------------------------------------------------
def test_the_two_variants_have_different_panel_fingerprints(variants) -> None:
    assert cache_module.frame_fingerprint(
        variants.with_dedup
    ) != cache_module.frame_fingerprint(variants.without_dedup)
    assert len(variants.without_dedup) > len(variants.with_dedup)


def test_the_two_variants_have_different_fit_ids(variants) -> None:
    """Both halves of the key move, the spec and the panel content."""

    def identifier(c1_dedup: bool) -> str:
        return cache_module.fit_id(
            cache_module.build_payload(
                panel_fingerprint=cache_module.frame_fingerprint(
                    variants.select(c1_dedup)
                ),
                panel_spec=dataclasses.asdict(variants.spec_for(c1_dedup)),
                flags={name: False for name in CONTROL_NAMES},
                design="temporal",
                cut=2019,
                seed=0,
                model_name="logreg",
                model_params={},
                target="ransomware_family",
                n_features=20,
            )
        )

    assert identifier(True) != identifier(False)
    assert panel_spec_for_c1(variants.spec, True).dedup_exact is True
    assert panel_spec_for_c1(variants.spec, False).dedup_exact is False


def test_c1_on_projects_onto_the_untouched_four_control_config() -> None:
    """The ON half of an extended run is the existing run, cell for cell.

    Equality of the ``ExperimentConfig`` is what makes that true: ``run_cell``
    derives the whole cache key from the config and the panel, and the ON panel
    is the production panel by the test above, so the ``fit_id`` is the same and
    the existing fit cache serves it.
    """
    base = _base_config()
    for index in range(16):
        four = dataclasses.replace(base, flags=ControlFlags.from_index(index))
        five = config_for_cell_c1(
            base, ExtendedControlFlags.from_index(index + 16)
        )
        assert five == four
        assert extended_flags_of(five).to_index() == index + 16
        assert extended_flags_of(four).to_index() == index + 16


def test_c1_off_projects_onto_a_config_with_dedup_disabled() -> None:
    base = _base_config()
    off = config_for_cell_c1(base, ExtendedControlFlags.from_index(3))
    assert off.panel.dedup_exact is False
    assert off.flags == ControlFlags.from_index(3)
    assert extended_flags_of(off).to_index() == 3


# --------------------------------------------------------------------------
# Running a cell
# --------------------------------------------------------------------------
def test_run_cell_c1_stamps_the_extended_config_id(variants, no_cache) -> None:
    base = _base_config()
    rows = {}
    for index in (0, 16):
        flags = ExtendedControlFlags.from_index(index)
        rows[index] = run_cell_c1(variants, base, flags, min_test_samples=5)

    for index, row in rows.items():
        assert row["config_id"] == index
        assert row["c1_dedup"] is bool(index >= 16)
        assert row["a1_prior"] is False and row["b2_axis"] is False
        for column in METRICS_COLUMN_ORDER_C1:
            if column != "run_id":
                assert column in row

    assert rows[0]["n_test"] > rows[16]["n_test"], (
        "leaving duplicates in must give the test window more samples"
    )
    assert rows[0]["n_train"] > rows[16]["n_train"]


# --------------------------------------------------------------------------
# Job enumeration for the 32-cell run
# --------------------------------------------------------------------------
def test_build_jobs_c1_enumerates_sixty_eight_per_group() -> None:
    jobs = build_jobs_c1(
        _base_config(), PRODUCTION_MODELS, PRODUCTION_SEEDS, PRODUCTION_CUTS
    )
    groups = len(PRODUCTION_CUTS) * len(PRODUCTION_MODELS) * len(PRODUCTION_SEEDS)
    assert JOBS_PER_GROUP_C1 == 68
    assert len(jobs) == 68 * groups == 6800

    coalitions = {}
    for cfg in jobs:
        design = cfg.split.design
        coalitions.setdefault(design, set()).add(extended_flags_of(cfg).to_index())
    assert coalitions["temporal"] == set(range(32))
    assert coalitions["random_fully_matched"] == set(range(32))
    assert coalitions["random"] == set(BASELINE_CONFIG_IDS_C1)
    assert coalitions["random_matched"] == set(BASELINE_CONFIG_IDS_C1)


def test_half_of_an_extended_run_is_already_in_the_four_control_run() -> None:
    """The cost claim in ``.draft/c1_control.md``: 6800 planned, 3400 new."""
    args = (_base_config(), PRODUCTION_MODELS, PRODUCTION_SEEDS, PRODUCTION_CUTS)
    four = set(map(repr, build_jobs(*args)))
    five = list(build_jobs_c1(*args))
    reusable = [cfg for cfg in five if repr(cfg) in four]
    assert len(reusable) == 3400
    assert len(five) - len(reusable) == 3400
    assert all(cfg.panel.dedup_exact for cfg in reusable)


def test_extended_flags_round_trip_through_every_job() -> None:
    jobs = build_jobs_c1(_base_config(), ("logreg",), (0,), (2019,))
    assert len(jobs) == 68
    for cfg in jobs:
        flags = extended_flags_of(cfg)
        assert config_for_cell_c1(cfg, flags) == cfg


# --------------------------------------------------------------------------
# Completeness of an extended table
# --------------------------------------------------------------------------
def _extended_metrics() -> pd.DataFrame:
    from conftest import metrics_row

    rows = []
    for index in range(32):
        flags = ExtendedControlFlags.from_index(index)
        for design in ("temporal", "random_fully_matched"):
            rows.append(
                metrics_row(
                    config_id=index,
                    design=design,
                    c1_dedup=flags.c1_dedup,
                    a1_prior=flags.a1_prior,
                    a2_labels=flags.a2_labels,
                    b1_fs=flags.b1_fs,
                    b2_axis=flags.b2_axis,
                    macro_f1=0.5 + 0.01 * index + (0.1 if design == "temporal" else 0.0),
                    fit_id="f{0}_{1}".format(design, index),
                )
            )
    return pd.DataFrame(rows, columns=list(METRICS_COLUMN_ORDER_C1))


def test_assert_lattice_complete_c1_accepts_a_full_table() -> None:
    assert_lattice_complete_c1(_extended_metrics())


def test_the_extended_frame_matches_the_extended_parquet_schema() -> None:
    """What ``run_lattice_c1`` will hand to pyarrow must cast without loss.

    ``config_id`` is ``int8``, and an extended index reaches 31, so this also
    checks the published width still holds five bits.
    """
    import pyarrow as pa

    from sift.experiment import METRICS_SCHEMA_C1, _metrics_frame_c1

    records = [
        {**row, "predictions": {}}
        for row in _extended_metrics().to_dict(orient="records")
    ]
    frame = _metrics_frame_c1(records, run_id="r_c1")
    assert list(frame.columns) == list(METRICS_COLUMN_ORDER_C1)
    assert set(frame["run_id"]) == {"r_c1"}
    assert frame["config_id"].max() == 31

    table = pa.Table.from_pandas(frame, schema=METRICS_SCHEMA_C1, preserve_index=False)
    assert table.num_rows == len(frame)
    assert table.column("config_id").to_pylist()[:1] != []


def test_assert_lattice_complete_c1_rejects_a_missing_coalition() -> None:
    from sift.experiment import IncompleteLatticeError

    frame = _extended_metrics()
    frame = frame[frame["config_id"] != 17]
    with pytest.raises(IncompleteLatticeError):
        assert_lattice_complete_c1(frame)


# --------------------------------------------------------------------------
# Shapley at five players
# --------------------------------------------------------------------------
def _game_from_dividends(dividends, players):
    coalitions = [
        frozenset(c)
        for size in range(len(players) + 1)
        for c in combinations(players, size)
    ]
    return {s: sum(m for t, m in dividends.items() if t <= s) for s in coalitions}


def test_exact_shapley_handles_five_players() -> None:
    """``sift.shapley`` was already written for n players; this pins n=5.

    The game is built from Harsanyi dividends, so the answer is known in closed
    form: phi_i is the sum of m(S)/|S| over the coalitions containing i. Nothing
    in the module needed extending for this to pass.
    """
    players = CONTROL_NAMES_C1
    dividends = {
        frozenset({"a1_prior"}): 0.10,
        frozenset({"a2_labels"}): 0.20,
        frozenset({"b1_fs"}): 0.04,
        frozenset({"b2_axis"}): 0.30,
        frozenset({"c1_dedup"}): 0.11,
        frozenset({"b1_fs", "b2_axis"}): 0.12,
        frozenset({"a1_prior", "a2_labels"}): -0.06,
        frozenset({"c1_dedup", "b2_axis"}): -0.08,
        frozenset({"c1_dedup", "a2_labels"}): 0.05,
        frozenset({"a1_prior", "a2_labels", "b2_axis"}): 0.06,
        frozenset({"a1_prior", "a2_labels", "b1_fs", "b2_axis", "c1_dedup"}): 0.032,
    }
    values = _game_from_dividends(dividends, players)
    assert len(values) == 32

    expected = {name: 0.0 for name in players}
    for coalition, dividend in dividends.items():
        for member in coalition:
            expected[member] += dividend / len(coalition)

    phi = exact_shapley(values)
    assert set(phi) == set(players)
    for name in players:
        assert phi[name] == pytest.approx(expected[name], abs=1e-12)
    assert sum(phi.values()) == pytest.approx(values[frozenset(players)], abs=1e-12)


def test_decompose_verifies_a_five_player_game_with_the_c1_tiers() -> None:
    """Owen and the quotient game carry a 2+3 partition without change."""
    players = CONTROL_NAMES_C1
    dividends = {
        frozenset({name}): 0.1 * (position + 1)
        for position, name in enumerate(players)
    }
    dividends[frozenset({"c1_dedup", "b1_fs"})] = -0.05
    values = _game_from_dividends(dividends, players)

    result = decompose(
        0.9, values, groups=TIER_GROUPS_C1, group_names=("data", "protocol")
    )
    result.raise_if_any_check_failed()
    assert result.all_checks_passed
    assert set(result.shapley) == set(players)
    assert set(result.owen) == set(players)
    assert set(result.tier_shapley) == {"data", "protocol"}
    assert sum(result.shapley.values()) + result.residual == pytest.approx(0.9)


# --------------------------------------------------------------------------
# gap_values at four and five players
# --------------------------------------------------------------------------
def _gap_table(n_players: int) -> pd.DataFrame:
    from conftest import metrics_row

    names = CONTROL_NAMES if n_players == 4 else CONTROL_NAMES_C1
    order = METRICS_COLUMN_ORDER if n_players == 4 else METRICS_COLUMN_ORDER_C1
    rows = []
    for index in range(1 << n_players):
        bits = {
            name: bool(index >> position & 1) for position, name in enumerate(names)
        }
        for design, offset in (("temporal", 0.0), ("random_fully_matched", 0.2)):
            rows.append(
                metrics_row(
                    config_id=index,
                    design=design,
                    macro_f1=0.5 + 0.01 * index + offset,
                    fit_id="f{0}_{1}".format(design, index),
                    **bits,
                )
            )
    return pd.DataFrame(rows, columns=list(order))


def test_gap_values_default_stays_four_players() -> None:
    assert DEFAULT_CONTROL_COLUMNS == ("a1_prior", "a2_labels", "b1_fs", "b2_axis")
    values = gap_values(_gap_table(4), model="logreg", cut=2019, seed=0)
    assert len(values) == 16
    assert values[frozenset()] == 0.0
    assert all("c1_dedup" not in coalition for coalition in values)


def test_gap_values_reads_a_five_control_table_when_asked() -> None:
    values = gap_values(
        _gap_table(5),
        model="logreg",
        cut=2019,
        seed=0,
        control_names=CONTROL_NAMES_C1,
    )
    assert len(values) == 32
    assert values[frozenset()] == 0.0
    assert frozenset(CONTROL_NAMES_C1) in values
    phi = exact_shapley(values)
    assert set(phi) == set(CONTROL_NAMES_C1)


def test_gap_values_refuses_a_four_control_table_read_as_five() -> None:
    """A missing player is an error, never a smaller game silently accepted."""
    with pytest.raises(ValueError, match="c1_dedup"):
        gap_values(
            _gap_table(4),
            model="logreg",
            cut=2019,
            seed=0,
            control_names=CONTROL_NAMES_C1,
        )
