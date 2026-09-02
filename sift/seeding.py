"""Deterministic seed derivation.

Every stochastic object in the experiment receives an explicitly derived seed.
Three rules follow from the requirement that a difference between two lattice
cells be attributable to the controls rather than to chance:
"""

from __future__ import annotations

import hashlib
import json

import numpy

__all__ = ["derive_seed", "make_rng", "SEED_UPPER_BOUND"]

#: Exclusive upper bound of derived seeds. NumPy generators and scikit-learn
#: estimators both accept any value below 2 ** 32.
SEED_UPPER_BOUND: int = 2 ** 32


def derive_seed(base_seed: int, *parts: str | int) -> int:
    """Derive a reproducible child seed from a base seed and a set of role labels.

    The mapping is a pure function of its arguments, so the same call yields the
    same seed on any machine, in any process, in any order.

    Parameters
    ----------
    base_seed : int
        Base seed of the run, taken from :attr:`sift.config.ExperimentConfig.seed`.
    *parts : str or int
        Role labels identifying the consumer, for example ``"model"``, the model name
        and the cut year.

    Returns
    -------
    int
        A seed in ``range(SEED_UPPER_BOUND)``.

    Examples
    --------
    >>> derive_seed(0, "model", "lightgbm", 2019) == derive_seed(0, "model", "lightgbm", 2019)
    True
    >>> derive_seed(0, "model", "lightgbm", 2019) == derive_seed(0, "model", "lightgbm", 2021)
    False
    """
    payload = json.dumps([base_seed, [str(part) for part in parts]], separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_UPPER_BOUND


def make_rng(seed: int) -> numpy.random.Generator:
    """Return a NumPy generator seeded deterministically.

    Parameters
    ----------
    seed : int
        Seed, normally the output of :func:`derive_seed`.

    Returns
    -------
    numpy.random.Generator
        A PCG64 generator. The modern generator API is used rather than the
        legacy :class:`numpy.random.RandomState` because its stream is guaranteed
        stable across NumPy releases.
    """
    return numpy.random.default_rng(seed)
