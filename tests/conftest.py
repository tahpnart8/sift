"""Shared fixtures for the SIFT test suite.

Run from the repository root with ``python -m pytest tests -q``.  ``python -m``
prepends the current working directory to ``sys.path``, which lets the suite
import ``sift`` before ``pip install -e .`` has been run.  No test may call
``sys.path.append``; CONTRACT.md section 2 forbids it.

Every import of ``sift`` happens lazily inside a fixture or inside a test
module.  A module-level import here would turn one missing module into a
collection error for the whole suite and destroy the per-file signal.

THE GOLDEN GAME
---------------
``GOLDEN_VALUES`` is a four-player cooperative game whose Shapley values, Owen
values and Harsanyi dividends were computed by hand and then verified along a
second, independent route (Moebius transform versus the direct marginal-
contribution sum).  Self-consistency checks such as efficiency pass even when
the implementation is wrong in a way that is symmetric across players, so the
suite pins absolute numbers rather than relations between them.

The game is built from these dividends::

    m({a1_prior})                                  =  0.10
    m({a2_labels})                                 =  0.20
    m({b1_fs})                                     =  0.04
    m({b2_axis})                                   =  0.30
    m({b1_fs, b2_axis})                            =  0.12
    m({a1_prior, a2_labels})                       = -0.06
    m({a1_prior, a2_labels, b2_axis})              =  0.06
    m({a1_prior, a2_labels, b1_fs, b2_axis})       =  0.024
    all other m(T)                                 =  0

and ``v(S) = sum of m(T) for every T contained in S``.  It is deliberately
non-additive and asymmetric: no two players are interchangeable, one pairwise
dividend is negative, and Owen differs from Shapley on three of four players.
"""

from __future__ import annotations

import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MLRAN_DIR = REPO_ROOT / "mlran"

A1, A2, B1, B2 = "a1_prior", "a2_labels", "b1_fs", "b2_axis"

# Column names as defined in sift/mock.py and sift/config.py.
TIME_COLUMN_CORRECT = "first_submission_date_year"   # control B2 on
TIME_COLUMN_NAIVE = "Year"                            # control B2 off
MODEL = "logreg"
PLAYERS = (A1, A2, B1, B2)

GOLDEN_DIVIDENDS: dict[frozenset[str], float] = {
    frozenset(): 0.0,
    frozenset({A1}): 0.10,
    frozenset({A2}): 0.20,
    frozenset({B1}): 0.04,
    frozenset({B2}): 0.30,
    frozenset({B1, B2}): 0.12,
    frozenset({A1, A2}): -0.06,
    frozenset({A1, A2, B2}): 0.06,
    frozenset({A1, A2, B1, B2}): 0.024,
}


def _all_coalitions() -> list[frozenset[str]]:
    out: list[frozenset[str]] = []
    for size in range(len(PLAYERS) + 1):
        out.extend(frozenset(c) for c in combinations(PLAYERS, size))
    return out


def _game_from_dividends(dividends: dict[frozenset[str], float]) -> dict[frozenset[str], float]:
    return {
        s: sum(m for t, m in dividends.items() if t <= s)
        for s in _all_coalitions()
    }


GOLDEN_VALUES: dict[frozenset[str], float] = _game_from_dividends(GOLDEN_DIVIDENDS)

# Hand computed, then cross-checked by the direct Shapley marginal sum.
# phi_b1 for example: 1/4*0.04 + 1/12*(0.04+0.04+0.16+0.04+0.16+0.16) + 1/4*0.184
#                   = 0.010 + 0.050 + 0.046 = 0.106
GOLDEN_SHAPLEY: dict[str, float] = {
    A1: 0.096,
    A2: 0.196,
    B1: 0.106,
    B2: 0.386,
}

# Coalition structure {{a1_prior, a2_labels}, {b1_fs, b2_axis}}.  With two groups
# of two players every Owen term carries weight 1/4.
GOLDEN_OWEN: dict[str, float] = {
    A1: 0.091,
    A2: 0.191,
    B1: 0.106,
    B2: 0.396,
}

GOLDEN_GRAND_VALUE: float = 0.784

OWEN_GROUPS: tuple[tuple[str, ...], ...] = ((A1, A2), (B1, B2))

TOL = 1e-9


def mlran_present() -> bool:
    """Return True when the MLRan clone carries its metadata csv files."""
    meta = MLRAN_DIR / "2_collected_samples_metadata" / "mlran_dataset_metadata.csv"
    return meta.is_file()


requires_mlran = pytest.mark.skipif(
    not mlran_present(),
    reason="MLRan metadata not present; test needs the real dataset",
)


@pytest.fixture(scope="session")
def panel():
    """A synthetic analysis panel that stands in for the MLRan panel."""
    from sift.mock import synthetic_panel

    return synthetic_panel(n_samples=600, n_features=40, n_families=8, seed=0)


@pytest.fixture()
def golden_values() -> dict[frozenset[str], float]:
    return dict(GOLDEN_VALUES)


def run_in_subprocess(source: str) -> str:
    """Execute ``source`` in a fresh interpreter rooted at the repository.

    Used for cross-process determinism checks.  A same-process repeat cannot see
    non-determinism that comes from hash seeding or from thread scheduling
    inside a native library, so those checks must fork.
    """
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip()[-2000:])
    return completed.stdout.strip()


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "slow: exercises every model across the full lattice"
    )


NOVEL_FAMILY = "novel_family_zzz"


@pytest.fixture(scope="session")
def panel_with_novel_family(panel):
    """A panel guaranteed to contain a family that appears only after the cut.

    ``synthetic_panel`` spreads all of its families evenly across the whole time
    range, so the temporal test window never contains an unseen class and
    control A2 removes nothing.  Every A2 assertion written against the plain
    fixture therefore passes vacuously.  This fixture relabels a dozen rows that
    the cut-2019 temporal split places in the test half, which is the only way
    to make A2 observable without the real dataset.
    """
    from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
    from sift.controls import apply_controls

    reference = ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design="temporal", cut_year=2019),
        flags=ControlFlags(),
        model_name="logreg",
        seed=0,
        n_features=20,
    )
    controlled = apply_controls(panel, reference)

    modified = panel.copy()
    victims = list(controlled.test_idx)[:12]
    modified.loc[victims, "ransomware_family"] = NOVEL_FAMILY
    return modified


@pytest.fixture()
def no_cache(monkeypatch: pytest.MonkeyPatch):
    """Disable the fit cache for tests that vary the panel.

    ``run_cell`` keys its cache on the MLRan raw file digests and never on the
    panel it is handed, so two different panels collide on one fit_id. Tests
    that hold the config fixed and vary the panel must bypass the cache or they
    silently read back the previous panel's record. The collision itself is
    asserted in ``test_cache_key.py``; this fixture only keeps unrelated tests
    from tripping over it.
    """
    import sift.cache as cache_module

    monkeypatch.setattr(cache_module, "get", lambda identifier: None)
    monkeypatch.setattr(cache_module, "put", lambda identifier, record: None)
    yield


def metrics_row(**overrides):
    """One row of metrics.parquet with contract defaults, for building fixtures."""
    from sift.experiment import LATTICE_ROLE

    row = {
        "fit_id": "f" + str(abs(hash(tuple(sorted(map(str, overrides.items())))))),
        "run_id": "r0",
        "config_id": 0,
        "a1_prior": False,
        "a2_labels": False,
        "b1_fs": False,
        "b2_axis": False,
        "design": "temporal",
        "role": LATTICE_ROLE,
        "cut": 2019,
        "seed": 0,
        "model": "logreg",
        "n_train": 500,
        "n_test": 200,
        "k_train": 20,
        "k_test": 18,
        "macro_f1": 0.5,
        "balanced_accuracy": 0.5,
        "mcc": 0.4,
        "accuracy": 0.6,
        "fit_seconds": 1.0,
    }
    row.update(overrides)
    return row


def build_metrics(temporal_scores, reference_scores, cut=2019, model="logreg", seed=0):
    """Assemble a 32-row lattice table from two score maps keyed by config_id.

    Parameters
    ----------
    temporal_scores, reference_scores : Mapping of {int: float}
        macro-F1 per config_id, 0 to 15, for the two lattice arms.
    """
    import pandas as pd

    from sift.experiment import METRICS_COLUMN_ORDER

    rows = []
    for config_id in range(16):
        flags = _flags_for(config_id)
        for design, scores in (
            ("temporal", temporal_scores),
            ("random_fully_matched", reference_scores),
        ):
            rows.append(
                metrics_row(
                    config_id=config_id,
                    design=design,
                    cut=cut,
                    model=model,
                    seed=seed,
                    macro_f1=float(scores[config_id]),
                    fit_id="f{0}_{1}".format(design, config_id),
                    **flags,
                )
            )
    return pd.DataFrame(rows, columns=list(METRICS_COLUMN_ORDER))


def _flags_for(config_id):
    from sift.config import ControlFlags

    flags = ControlFlags.from_index(config_id)
    return {
        "a1_prior": flags.a1_prior,
        "a2_labels": flags.a2_labels,
        "b1_fs": flags.b1_fs,
        "b2_axis": flags.b2_axis,
    }


# --------------------------------------------------------------------------
# Isolation from the production results directory
# --------------------------------------------------------------------------
#: Set by the session fixture below, read by tests that assert the rule holds.
REDIRECTED_RESULTS: dict[str, object] = {}


#: Scratch tree the suite writes into instead of ``results/``. Deliberately a
#: stable path under ``tests/`` rather than a per-session
#: temporary directory: a fresh directory every session means a cold fit cache
#: and roughly three extra minutes per run, and the isolation guarantee does not
#: depend on the path being unique, only on it not being ``results/``.
SCRATCH_ROOT = REPO_ROOT / "tests" / ".scratch"


@pytest.fixture(scope="session", autouse=True)
def isolate_results_directory():
    """Point every write the suite can make at a scratch directory.

    Two constants are redirected, both read at call time by the code that uses
    them, so patching the module global is sufficient:

    * ``cache.RESULTS_DIR`` is what ``run_lattice`` falls back to when its
      ``results_dir`` argument is omitted;
    * ``cache.CACHE_DIR`` is ``results/fit_cache``, which every ``run_cell``
      writes into.

    This is autouse and session-scoped on purpose. A per-test opt-in would be
    forgotten exactly once, and once is enough: on 2026-08-27 at 00:25 four
    tests in this suite ran ``run_lattice`` with the default and overwrote
    ``results/metrics.parquet`` with a 34-row mock run while a production
    lattice was in flight. Making the suite structurally unable to reach that
    directory is a better guarantee than remembering to pass an argument.

    The scratch tree persists between runs so the fit cache stays warm; it is
    under ``tests/`` and is safe to delete at any time.
    """
    import sift.cache as cache_module

    scratch = SCRATCH_ROOT
    scratch.mkdir(parents=True, exist_ok=True)
    patch = pytest.MonkeyPatch()
    patch.setattr(cache_module, "RESULTS_DIR", scratch)
    patch.setattr(cache_module, "CACHE_DIR", scratch / "fit_cache")

    REDIRECTED_RESULTS["scratch"] = scratch
    REDIRECTED_RESULTS["real"] = REPO_ROOT / "results"
    try:
        yield scratch
    finally:
        patch.undo()
        REDIRECTED_RESULTS.clear()
