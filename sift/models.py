"""Seeded estimator factory.

Four model families, all at library defaults. If all four move together across
the lattice, the finding belongs to the evaluation protocol rather than to one
inductive bias.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

__all__ = ["MODEL_NAMES", "SUPPORTS_SAMPLE_WEIGHT", "build_model"]

#: Model keys accepted by :func:`build_model`.
MODEL_NAMES: tuple[str, ...] = ("logreg", "random_forest", "lightgbm", "mlp")

#: Models whose ``fit`` accepts ``sample_weight``. A model outside this set
#: receives control A1 through the weighted metric alone.
SUPPORTS_SAMPLE_WEIGHT: frozenset[str] = frozenset({"logreg", "random_forest", "lightgbm"})

# Iteration budgets, not tuned hyperparameters: the library defaults of 100 and
# 200 leave both solvers short of convergence here, and an unconverged fit adds
# optimiser noise to every lattice cell.
_MAX_ITER_LOGREG: int = 1000
_MAX_ITER_MLP: int = 500

# Single-threaded: the measured bottleneck is memory bandwidth, and parallelism
# pays off across lattice cells rather than inside one fit.
_N_JOBS: int = 1


def build_model(name: str, seed: int, *, multiclass: bool = True) -> BaseEstimator:
    """Construct an unfitted estimator with an explicitly derived seed.

    Parameters
    ----------
    name : {'logreg', 'random_forest', 'lightgbm', 'mlp'}
        Model key.
    seed : int
        Seed for this estimator, normally produced by :func:`sift.seeding.derive_seed`
        from the run seed, the model name and the cut year.
    multiclass : bool, default True
        Reserved for objective selection; no current model branches on it.

    Returns
    -------
    sklearn.base.BaseEstimator
        An unfitted estimator.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised model key.
    """
    if name == "logreg":
        # No n_jobs: no effect since scikit-learn 1.8, removed in 1.10.
        return LogisticRegression(
            max_iter=_MAX_ITER_LOGREG,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=_N_JOBS,
        )
    if name == "lightgbm":
        # deterministic must be paired with force_row_wise: it pins the
        # histogram order, without which accumulation varies between runs.
        return LGBMClassifier(
            random_state=seed,
            n_jobs=_N_JOBS,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
    if name == "mlp":
        return MLPClassifier(
            max_iter=_MAX_ITER_MLP,
            random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
