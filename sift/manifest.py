"""Run manifest: the record that makes a result table self-describing.

A metrics table without a manifest cannot be traced back to the code, data and
environment that produced it. Every row of ``results/metrics.parquet`` carries
a ``run_id`` that resolves here.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sift import cache

MANIFEST_VERSION: str = "1"

#: Libraries recorded for provenance beyond those in the cache key. These do
#: not change predictions but do change file formats and scheduling.
EXTRA_LIBRARIES: tuple[str, ...] = ("pyarrow", "joblib", "threadpoolctl")

_CROCKFORD: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_run_id() -> str:
    """Generate a lexicographically sortable run identifier.

    Returns
    -------
    str
        26-character ULID: 48 bits of millisecond timestamp followed by 80
        bits of randomness, encoded in Crockford base32.
    """
    value = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    characters = []
    for _ in range(26):
        characters.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(characters))


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns
    -------
    str
        Timestamp with second resolution and explicit offset.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_info(repo_dir: Path) -> dict[str, Any]:
    """Describe the git state of a directory, tolerating its absence.

    Parameters
    ----------
    repo_dir : Path
        Directory expected to be inside a git working tree.

    Returns
    -------
    dict
        Keys ``commit``, ``branch``, ``dirty`` and ``reason``. On failure the
        first three are ``None`` and ``reason`` explains why.
    """
    def _run(args: list[str]) -> tuple[int, str, str]:
        try:
            done = subprocess.run(
                ["git", "-C", str(repo_dir), *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return 127, "", str(error)
        return done.returncode, done.stdout.strip(), done.stderr.strip()

    code, commit, error = _run(["rev-parse", "HEAD"])
    if code != 0:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "reason": error or "git rev-parse failed with code " + str(code),
        }

    _, branch, _ = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    status_code, status, _ = _run(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status) if status_code == 0 else None,
        "reason": None,
    }


def _extra_library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for name in EXTRA_LIBRARIES:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "<absent>"
    return resolved


def collect_environment() -> dict[str, Any]:
    """Capture interpreter, platform and library state.

    Returns
    -------
    dict
        Environment description embedded in the manifest.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except ImportError:
        physical, logical = None, os.cpu_count()

    versions = dict(cache.library_versions())
    versions.update(_extra_library_versions())
    versions["python"] = platform.python_version()

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_physical": physical,
            "cpu_logical": logical,
        },
        "python_executable": sys.executable,
        "lib_versions": versions,
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def build_manifest(
    *,
    run_id: str,
    started_utc: str,
    ended_utc: str,
    wall_seconds: float,
    counts: Mapping[str, int],
    threading: Mapping[str, Any],
    rng_policy: Mapping[str, Any],
    lattice: Mapping[str, Any] | None = None,
    code_version: str | None = None,
    code_version_scope: list[str] | None = None,
    git_at_start: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest for one invocation of the lattice runner.

    Parameters
    ----------
    run_id : str
        Identifier stamped onto every metrics row of this run.
    started_utc, ended_utc : str
        ISO-8601 boundaries of the run.
    wall_seconds : float
        Measured wall-clock duration.
    counts : Mapping of str to int
        Fit accounting: planned, cached, executed, failed.
    threading : Mapping
        Parallelism settings actually used.
    rng_policy : Mapping
        Seed list and seed derivation policy.
    lattice : Mapping, optional
        Lattice shape: control names, order, cut years, models.
    git_at_start : Mapping, optional
        Result of :func:`git_info` captured before the first fit. Pass it, or
        the manifest records only where the tree ended up.
    code_version_scope : list of str, optional
        Module filenames the version covers. Recorded so a reader can tell
        which edits would and would not have invalidated this run's cache.
    code_version : str, optional
        Version pinned for the run. Pass the value returned by
        :func:`sift.cache.pin_code_version`; recomputing it here would record
        the sources as they are now rather than as the run saw them.

    Returns
    -------
    dict
        Manifest ready for :func:`write_manifest`.
    """
    # The commit at write time is not the commit the fits ran under: a run
    # lasting hours can straddle several commits. Record both ends so a reader
    # can see whether the tree moved rather than having to assume it did not.
    git_end = git_info(cache.PROJECT_ROOT)
    git_start = dict(git_at_start) if git_at_start else git_end
    moved = git_start.get("commit") != git_end.get("commit")

    counts = dict(counts)
    executed = counts.get("fits_recorded")
    cached_hits = counts.get("fits_from_cache")
    if executed is not None and cached_hits is not None:
        counts.setdefault("fits_executed", executed - cached_hits)
    counts.setdefault("fits_failed", 0)

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": cache.SCHEMA_VERSION,
        "run_id": run_id,
        "task": cache.TASK_NAME,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "wall_seconds": round(float(wall_seconds), 3),
        "git": dict(git_at_start) if git_at_start else git_info(cache.PROJECT_ROOT),
        "git_at_end": git_end,
        "git_moved_during_run": moved,
        "code_version": code_version or cache.code_version(),
        "code_version_scope": code_version_scope or cache.fit_path_modules(),
        "dataset": cache.dataset_fingerprint(),
        "environment": collect_environment(),
        "threading": dict(threading),
        "rng_policy": dict(rng_policy),
        "counts": counts,
        # Flat aliases under the names the round-0 manifest spec used, so a
        # reader does not have to know the shape of the counts block.
        "n_fits_planned": counts.get("fits_planned"),
        "n_fits_executed": counts.get("fits_executed"),
        "n_fits_cached": counts.get("fits_from_cache"),
        "n_fits_failed": counts.get("fits_failed"),
    }
    if lattice is not None:
        manifest["lattice"] = dict(lattice)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: Path | None = None) -> Path:
    """Persist a manifest as indented JSON.

    Parameters
    ----------
    manifest : Mapping
        Output of :func:`build_manifest`.
    path : Path, optional
        Destination. Defaults to ``results/run_manifest.json``.

    Returns
    -------
    Path
        Where the manifest was written.
    """
    target = path or (cache.RESULTS_DIR / "run_manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target
