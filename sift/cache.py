"""Content-addressed cache for individual model fits in the SIFT lattice.

The cache is deliberately hand-rolled rather than built on ``joblib.Memory``.
``joblib.Memory`` identifies an entry by function name plus the pickle of its
arguments; the analysis panel is a :class:`pandas.DataFrame` whose pickle
representation is not guaranteed stable, and joblib only detects edits inside
the decorated function body, not inside helpers it calls. Both failure modes
would return silently stale fits, which is the one error this project cannot
tolerate.

The key therefore hashes the full source of every ``sift/*.py`` module, so any
edit anywhere in the pipeline invalidates every cached fit.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

SCHEMA_VERSION: str = "3"
TASK_NAME: str = "family32"

SIFT_DIR: Path = Path(__file__).resolve().parent

# Directory anchors come from sift.paths so there is exactly one definition of
# where the project lives. sift.paths imports nothing from the package, so this
# cannot cycle.
from sift.paths import MLRAN_DIR as MLRAN_DATA_DIR  # noqa: E402
from sift.paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

MLRAN_REPO: Path = PROJECT_ROOT / "mlran"
# Overridable so tests and smoke runs never write into the real result cache.
# Read from the environment rather than a module global because loky workers
# re-import this module and would otherwise miss a monkeypatch.
CACHE_DIR: Path = Path(os.environ.get("SIFT_CACHE_DIR", str(RESULTS_DIR / "fit_cache")))

#: The three raw inputs whose content defines the dataset identity.
DATA_FILES: tuple[str, ...] = (
    "MLRan_X_train_RFE.csv",
    "MLRan_X_test_RFE.csv",
    "mlran_dataset_metadata.csv",
)

#: Libraries whose version can change a fitted model's predictions.
KEYED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "scipy",
)

#: Estimator parameters excluded from the key. Verified in round 0 that
#: LightGBM predictions are bit-identical across ``n_jobs`` in {1, 2, 4, 8},
#: so keying on them would fragment the cache without protecting correctness.
UNKEYED_MODEL_PARAMS: frozenset[str] = frozenset(
    {"n_jobs", "verbose", "verbosity", "silent", "n_threads", "num_threads"}
)

_DIGEST_SIZE: int = 16


# --------------------------------------------------------------------------
# Environment and source fingerprints
# --------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file read in chunks.

    Parameters
    ----------
    path : Path
        File to digest.

    Returns
    -------
    str
        Hex digest, or the literal ``<missing>`` when the file is absent.
    """
    if not path.is_file():
        return "<missing>"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


#: Environment variable carrying a pinned code version to worker processes.
CODE_VERSION_ENV: str = "SIFT_CODE_VERSION"

#: Modules that consume results but can never change a fit. They are excluded
#: from :func:`code_version`.
#:
#: This is an exclusion list rather than an inclusion list on purpose: a module
#: E1 adds later is hashed by default, so the failure mode of forgetting to
#: update this set is a wasted refit, never a stale result served as fresh.
#:
#: Measured cost of getting this wrong: hashing the whole package meant an edit
#: to shapley.py and reporting.py, neither of which touches the fit path,
#: invalidated 2,701 completed fits and about an hour of compute.
NON_FIT_MODULES: frozenset[str] = frozenset(
    {
        "shapley.py",
        "drift.py",
        "reporting.py",
        "mock.py",
        "recovery.py",
        "driftsim.py",
    }
)


def fit_path_modules() -> list[str]:
    """Return the module filenames that feed :func:`code_version`.

    Returns
    -------
    list of str
        Sorted ``sift/*.py`` basenames excluding :data:`NON_FIT_MODULES`.
    """
    return [p.name for p in sorted(SIFT_DIR.glob("*.py")) if p.name not in NON_FIT_MODULES]


@lru_cache(maxsize=1)
def code_version() -> str:
    """Hash the source of every ``sift/*.py`` module.

    Returns
    -------
    str
        Hex digest covering module names and byte content, in path order, or
        the value pinned in :data:`CODE_VERSION_ENV` when one is set.

    Notes
    -----
    Hashing the package, not just the entry point, is the specific defect that
    disqualifies ``joblib.Memory``: an edit to a helper module must invalidate
    every cached fit that depends on it. The converse matters just as much,
    which is why :data:`NON_FIT_MODULES` is excluded -- an edit to the analysis
    or plotting layer cannot change a fit and must not throw one away.
    """
    pinned = os.environ.get(CODE_VERSION_ENV)
    if pinned:
        return pinned
    digest = hashlib.sha256()
    for path in sorted(SIFT_DIR.glob("*.py")):
        if path.name in NON_FIT_MODULES:
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pin_code_version() -> str:
    """Freeze the code version for the duration of one run.

    Returns
    -------
    str
        The pinned digest, also exported to :data:`CODE_VERSION_ENV`.

    Notes
    -----
    Each loky worker re-imports this module and would otherwise hash the
    sources itself. A source edit landing while a run is in flight would then
    give early and late fits different cache keys, silently splitting one run
    across two identities. Pinning in the parent and inheriting through the
    environment makes a run internally consistent by construction.
    """
    code_version.cache_clear()
    os.environ.pop(CODE_VERSION_ENV, None)
    value = code_version()
    os.environ[CODE_VERSION_ENV] = value
    return value


def unpin_code_version() -> None:
    """Release a pinned code version so later runs re-hash the sources."""
    os.environ.pop(CODE_VERSION_ENV, None)
    code_version.cache_clear()


@lru_cache(maxsize=1)
def library_versions() -> dict[str, str]:
    """Resolve installed versions of the libraries that affect predictions.

    Returns
    -------
    dict of str to str
        Distribution name mapped to version, or the literal ``<absent>``.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for name in KEYED_LIBRARIES:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "<absent>"
    return resolved


@lru_cache(maxsize=1)
def mlran_commit() -> str | None:
    """Return the short commit of the vendored MLRan checkout.

    Returns
    -------
    str or None
        Short hex commit, or ``None`` when the checkout is not a git repo.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(MLRAN_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


@lru_cache(maxsize=1)
def dataset_fingerprint() -> dict[str, Any]:
    """Identify the dataset by upstream commit plus raw file digests.

    Returns
    -------
    dict
        Keys ``mlran_commit`` and ``files``.
    """
    return {
        "mlran_commit": mlran_commit(),
        "files": {name: _sha256_file(MLRAN_DATA_DIR / name) for name in DATA_FILES},
    }


# --------------------------------------------------------------------------
# Canonical JSON and key construction
# --------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """Coerce an arbitrary value into a deterministically serialisable form.

    Parameters
    ----------
    value : Any
        Value drawn from a config, flag set, or estimator parameter dict.

    Returns
    -------
    Any
        JSON-compatible value with a stable ordering.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    # Estimator instances and other opaque objects fall back to their repr,
    # which for scikit-learn spells out every non-default parameter.
    return repr(value)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialise a payload with sorted keys and no incidental whitespace.

    Parameters
    ----------
    payload : Mapping
        Cache key payload.

    Returns
    -------
    str
        Canonical JSON text.
    """
    return json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def keyed_model_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop estimator parameters that provably do not change predictions.

    Parameters
    ----------
    params : Mapping
        Output of ``estimator.get_params()``.

    Returns
    -------
    dict
        Parameters retained for the cache key.
    """
    return {k: v for k, v in params.items() if k not in UNKEYED_MODEL_PARAMS}


def build_payload(
    *,
    panel_fingerprint: str,
    panel_spec: Mapping[str, Any] | Any,
    flags: Mapping[str, bool],
    design: str,
    cut: int | None,
    seed: int,
    model_name: str,
    model_params: Mapping[str, Any],
    target: str,
    n_features: int,
    task: str = TASK_NAME,
) -> dict[str, Any]:
    """Assemble the full cache key payload for one fit.

    Parameters
    ----------
    panel_fingerprint : str
        Content fingerprint of the realised analysis panel.
    panel_spec : Mapping or PanelSpec
        The specification the panel was built from.
    flags : Mapping of str to bool
        The four lattice controls.
    design : str
        Either ``random`` or ``temporal``.
    cut : int or None
        Cut year defining the temporal boundary.
    seed : int
        Base seed for this replicate.
    model_name : str
        Estimator identifier.
    model_params : Mapping
        Estimator parameters before filtering.
    target : str
        Label column.
    n_features : int
        Feature budget requested by the config.
    task : str, optional
        Task identifier.

    Returns
    -------
    dict
        Payload ready for :func:`fit_id`.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "dataset": dataset_fingerprint(),
        # Two separate facts, and both are load-bearing. The fingerprint
        # identifies the realised panel, so a preprocessing change the spec
        # does not express still invalidates the key. The spec is keyed as
        # well because primary_axis, secondary_axis and prior_reference_rate
        # are read at fit time by apply_controls and therefore change the
        # result without changing a single panel row.
        "panel_fingerprint": panel_fingerprint,
        "panel_spec": panel_spec,
        "flags": {str(k): bool(v) for k, v in flags.items()},
        "design": design,
        "cut": cut,
        "seed": int(seed),
        "model": model_name,
        "model_params": keyed_model_params(model_params),
        "target": target,
        "n_features": int(n_features),
        "code_version": code_version(),
        "lib_versions": library_versions(),
    }


def fit_id(payload: Mapping[str, Any]) -> str:
    """Derive the content address of a fit from its key payload.

    Parameters
    ----------
    payload : Mapping
        Output of :func:`build_payload`.

    Returns
    -------
    str
        32-character hex identifier.
    """
    return hashlib.blake2b(
        canonical_json(payload).encode("utf-8"), digest_size=_DIGEST_SIZE
    ).hexdigest()


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
#: Key under which each stored record carries the schema it was written with.
RECORD_SCHEMA_KEY: str = "_schema_version"

#: Marker file recording the schema a cache directory was created under.
CACHE_INFO_NAME: str = "CACHE_INFO.json"


class CacheSchemaError(RuntimeError):
    """Raised when a cache directory was written under a different schema."""


def _ensure_cache_info() -> None:
    """Write the schema marker if the directory does not already carry one.

    Notes
    -----
    Called from :func:`put` so the invariant holds no matter how the cache was
    populated. Without this, code calling :func:`run_cell` directly would build
    a directory of valid entries that :func:`assert_cache_compatible` later
    refuses. Correctness does not rest on this marker: every record carries its
    own :data:`RECORD_SCHEMA_KEY`, and a record from another schema reads as a
    miss, so the worst outcome of a wrongly-marked directory is a refit.
    """
    info = CACHE_DIR / CACHE_INFO_NAME
    if info.is_file():
        return
    try:
        info.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION}, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def assert_cache_compatible() -> None:
    """Refuse to reuse a cache directory written under an older schema.

    Raises
    ------
    CacheSchemaError
        When the directory carries a different :data:`SCHEMA_VERSION`.

    Notes
    -----
    Schema 1 keyed fits on the raw MLRan files alone, so two panels built from
    the same raw data under different ``PanelSpec`` settings collided on one
    identifier. A run reusing such a directory would silently read back the
    other panel's numbers. That is a wrong published result rather than a
    crash, so a stale directory is refused outright and must be deleted; there
    is deliberately no migration path.
    """
    info = CACHE_DIR / CACHE_INFO_NAME
    if not info.is_file():
        if any(CACHE_DIR.glob("*/*.pkl")):
            raise CacheSchemaError(
                "cache directory " + str(CACHE_DIR) + " holds entries but no "
                + CACHE_INFO_NAME
                + ", so it predates schema " + SCHEMA_VERSION
                + ". Delete the directory and re-run; do not migrate it."
            )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        info.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION}, indent=2), encoding="utf-8"
        )
        return

    found = json.loads(info.read_text(encoding="utf-8")).get("schema_version")
    if found != SCHEMA_VERSION:
        raise CacheSchemaError(
            "cache directory " + str(CACHE_DIR) + " was written under schema "
            + repr(found) + " but this code is schema " + repr(SCHEMA_VERSION)
            + ". Delete the directory and re-run; do not migrate it."
        )


def _path_for(identifier: str) -> Path:
    # Two-character fan-out keeps any single directory well under the point
    # where Windows directory enumeration degrades.
    return CACHE_DIR / identifier[:2] / (identifier + ".pkl")


def exists(identifier: str) -> bool:
    """Test cache membership without deserialising the entry.

    Parameters
    ----------
    identifier : str
        Fit identifier.

    Returns
    -------
    bool
        True when a stored record is present.
    """
    return _path_for(identifier).is_file()


def get(identifier: str) -> dict[str, Any] | None:
    """Load a cached fit record.

    Parameters
    ----------
    identifier : str
        Fit identifier.

    Returns
    -------
    dict or None
        The stored record, or ``None`` on a miss or unreadable entry.
    """
    path = _path_for(identifier)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            record = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError, ValueError):
        # A truncated entry left by an interrupted write is treated as a miss
        # rather than an error, so a resumed run repairs itself.
        return None
    if record.get(RECORD_SCHEMA_KEY) != SCHEMA_VERSION:
        # Defence in depth behind assert_cache_compatible, for a directory
        # assembled by hand or copied between checkouts.
        return None
    return record


def put(identifier: str, record: Mapping[str, Any]) -> Path:
    """Store a fit record atomically.

    Parameters
    ----------
    identifier : str
        Fit identifier.
    record : Mapping
        Record to persist.

    Returns
    -------
    Path
        Location of the stored entry.
    """
    path = _path_for(identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_cache_info()
    # Write to a process-unique temporary name then rename, so a crash never
    # leaves a partial file that a later run would read back as a hit.
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    stamped = dict(record)
    stamped[RECORD_SCHEMA_KEY] = SCHEMA_VERSION
    with tmp.open("wb") as handle:
        pickle.dump(stamped, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def stored_ids() -> set[str]:
    """List every fit identifier currently held in the cache.

    Returns
    -------
    set of str
        Identifiers present on disk.
    """
    if not CACHE_DIR.is_dir():
        return set()
    return {path.stem for path in CACHE_DIR.glob("*/*.pkl")}


# --------------------------------------------------------------------------
# Data fingerprints for notebooks and manifests
# --------------------------------------------------------------------------
def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint a DataFrame's content, column names and row order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame to fingerprint, typically the analysis panel.

    Returns
    -------
    str
        32-character hex digest.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(str(frame.shape).encode("utf-8"))
    digest.update(hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()
