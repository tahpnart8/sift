"""The suite must never write into the production results directory.

A rule nobody enforces gets broken. On 2026-08-27 at 00:25 four tests in this
suite ran ``run_lattice`` with the default ``results_dir`` and overwrote
``results/metrics.parquet``, ``predictions.parquet`` and ``run_manifest.json``
with a 34-row mock run, while a two-hour production lattice was in flight. It
was harmless that time only because the run had not yet reached its write.

Two independent guards, because they fail in different ways:

* a runtime guard, that the redirection installed by ``isolate_results_directory``
  is actually in force and points somewhere other than ``results/``;
* a static guard, that no test file calls ``run_lattice`` without naming a
  directory, which catches a new test written after the redirection is someday
  removed or scoped down.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import REPO_ROOT, REDIRECTED_RESULTS

TEST_DIR = Path(__file__).resolve().parent
REAL_RESULTS = REPO_ROOT / "results"


def test_the_results_directory_is_redirected_away_from_production() -> None:
    import sift.cache as cache_module

    assert REDIRECTED_RESULTS, "the isolation fixture did not run"
    scratch = Path(REDIRECTED_RESULTS["scratch"]).resolve()

    assert scratch != REAL_RESULTS.resolve()
    assert REAL_RESULTS.resolve() not in scratch.parents
    assert Path(cache_module.RESULTS_DIR).resolve() == scratch
    assert Path(cache_module.CACHE_DIR).resolve() == (scratch / "fit_cache").resolve()


def test_the_fit_cache_does_not_point_inside_the_production_results_tree() -> None:
    """CACHE_DIR defaults to results/fit_cache, so run_cell writes there too."""
    import sift.cache as cache_module

    cache_dir = Path(cache_module.CACHE_DIR).resolve()
    real = REAL_RESULTS.resolve()
    assert real != cache_dir
    assert real not in cache_dir.parents


def test_no_test_calls_run_lattice_without_naming_a_results_directory() -> None:
    """Static guard, independent of whether the fixture is in force."""
    offenders: list[str] = []
    for path in sorted(TEST_DIR.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"run_lattice\s*\(", source):
            tail = source[match.end() : match.end() + 600]
            depth, call = 1, []
            for char in tail:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        break
                call.append(char)
            if "results_dir" not in "".join(call):
                line = source[: match.start()].count("\n") + 1
                offenders.append("{0}:{1}".format(path.name, line))

    assert not offenders, (
        "these run_lattice calls do not name a results_dir and would write into "
        "the production directory if the isolation fixture were ever removed: "
        + ", ".join(offenders)
    )


def test_writing_through_the_runner_lands_in_the_scratch_directory(panel, tmp_path) -> None:
    """End to end: the three output files appear where they were asked to."""
    from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
    from sift.experiment import run_lattice

    before = _snapshot(REAL_RESULTS)
    run_lattice(
        panel,
        base=ExperimentConfig(
            panel=PanelSpec(),
            split=SplitSpec(design="temporal", cut_year=2019),
            flags=ControlFlags(),
            model_name="logreg",
            seed=0,
            n_features=20,
        ),
        model_names=["logreg"],
        seeds=[0],
        cut_years=[2019],
        results_dir=tmp_path,
    )
    assert (tmp_path / "metrics.parquet").is_file()
    assert _snapshot(REAL_RESULTS) == before, (
        "a run_lattice call with an explicit results_dir still touched the "
        "production directory"
    )


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Name to (size, mtime_ns) for the three run outputs only.

    Deliberately narrow. A production lattice may be writing into this directory
    concurrently, and its fit_cache churns constantly, so a broad snapshot would
    produce false failures that blame this suite for another process's writes.
    """
    names = ("metrics.parquet", "predictions.parquet", "run_manifest.json")
    out: dict[str, tuple[int, int]] = {}
    for name in names:
        path = directory / name
        if path.is_file():
            stat = path.stat()
            out[name] = (stat.st_size, stat.st_mtime_ns)
    return out
