"""Seeded estimator factory.

Four model families span the bias-variance range of interest: a linear model, a
bagged tree ensemble, a boosted tree ensemble and a shallow neural network. If
all four move together across the lattice, the finding is a property of the
evaluation protocol rather than of any one inductive bias.

Hyperparameters are library defaults by deliberate decision: the experiment
measures the effect of evaluation controls, and tuning per cell would confound
that effect with tuning quality. The two exceptions are optimiser iteration
budgets, which govern whether a fit converges at all rather than what it prefers,
and the LightGBM determinism settings required for reproducibility.

Every estimator receives an explicitly derived seed. No literal seed appears in
this module.
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

#: Models whose ``fit`` accepts ``sample_weight``. Control A1 weights the training
#: half, so a model outside this set cannot express A1 on the training side and
#: receives the correction through the weighted metric alone.
SUPPORTS_SAMPLE_WEIGHT: frozenset[str] = frozenset({"logreg", "random_forest", "lightgbm"})

# Iteration budgets, not tuned hyperparameters. The scikit-learn defaults of 100
# and 200 leave both iterative solvers short of convergence on a design matrix of
# several hundred binary features spread over 32 classes, and an unconverged fit
# would add optimiser noise to every lattice cell.
_MAX_ITER_LOGREG: int = 1000
_MAX_ITER_MLP: int = 500

# Every estimator runs single-threaded. Measured on this machine, the bottleneck
# is memory bandwidth rather than thread count, and parallelism is more useful at
# the level of lattice cells than inside a single fit.
_N_JOBS: int = 1


def build_model(name: str, seed: int, *, multiclass: bool = True) -> BaseEstimator:
    """Construct an unfitted estimator with an explicitly derived seed.

    Parameters
    ----------
    name : {'logreg', 'random_forest', 'lightgbm', 'mlp'}
        Model key.
    seed : int
        Seed for this estimator, normally produced by
        :func:`sift.seeding.derive_seed` from the run seed, the model name and
        the cut year. A literal is never acceptable here: a fixed seed inside the
        factory would leave the model's randomness constant across a run's seed
        list, which is the defect this design corrects.
    multiclass : bool, default True
        Reserved for objective selection. LightGBM infers its objective from the
        label vector, and the remaining estimators are objective-agnostic, so no
        current model branches on it.

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
        # No n_jobs: it has had no effect since scikit-learn 1.8 and is scheduled
        # for removal in 1.10, so passing it only raises a FutureWarning.
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
        # deterministic with force_row_wise pins the histogram construction order.
        # Without the pairing LightGBM may choose a column-wise layout whose
        # floating-point accumulation order varies between runs.
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
