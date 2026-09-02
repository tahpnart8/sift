"""The fit cache must be addressed by the panel it actually scored.

R2 found that ``PanelSpec`` and the panel itself were both absent from the cache
key, so two panels built from the same raw MLRan files collided on one
``fit_id``.  E2 measured the consequence: the deduplication sensitivity analysis
would have reported 0.000 where the true effect is 0.114.

That is fixed.  ``cache.build_payload`` now takes ``panel_fingerprint`` and
``panel_spec``, and the first four tests below are regression guards on the fix.

What is NOT fixed is how the fingerprint reaches the payload.
``experiment.panel_fingerprint`` memoises the digest into ``panel.attrs``, and
pandas propagates ``attrs`` across ``.copy()`` and slicing.  Any panel derived
from a fingerprinted one therefore carries its parent's digest, and ``run_cell``
trusts it.  The last two tests fail on that and are meant to.
"""

from __future__ import annotations

import pytest

import sift.cache as cache_module
from sift.config import ControlFlags, ExperimentConfig, PanelSpec, SplitSpec
from sift.experiment import panel_fingerprint, run_cell

BASE_FLAGS = {"a1_prior": False, "a2_labels": False, "b1_fs": False, "b2_axis": False}


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design="temporal", cut_year=2019),
        flags=ControlFlags(),
        model_name="logreg",
        seed=0,
        n_features=20,
    )


def _fit_id(panel_spec: PanelSpec, fingerprint: str = "deadbeef" * 4) -> str:
    return cache_module.fit_id(
        cache_module.build_payload(
            panel_fingerprint=fingerprint,
            panel_spec=panel_spec,
            flags=BASE_FLAGS,
            design="temporal",
            cut=2019,
            seed=0,
            model_name="logreg",
            model_params={},
            target="ransomware_family",
            n_features=200,
        )
    )


# --------------------------------------------------------------------------
# Regression guards on the R2 fix
# --------------------------------------------------------------------------
def test_panel_spec_reaches_the_cache_key() -> None:
    """dedup_exact is the switch the published sensitivity analysis turns off.

    E2 measured the true effect at 0.114. Before the fix this key collided and
    the analysis would have reported 0.000.
    """
    assert _fit_id(PanelSpec(dedup_exact=True)) != _fit_id(PanelSpec(dedup_exact=False))


def test_min_class_size_reaches_the_cache_key() -> None:
    assert _fit_id(PanelSpec(min_class_size=20)) != _fit_id(PanelSpec(min_class_size=5))


def test_year_window_reaches_the_cache_key() -> None:
    assert _fit_id(PanelSpec(year_min=2012)) != _fit_id(PanelSpec(year_min=2014))


def test_panel_fingerprint_reaches_the_cache_key() -> None:
    """Two panels with the same spec but different content must not collide."""
    spec = PanelSpec()
    assert _fit_id(spec, fingerprint="a" * 32) != _fit_id(spec, fingerprint="b" * 32)


def test_frame_fingerprint_responds_to_a_single_changed_cell(panel) -> None:
    """The underlying digest is sound; only its delivery is not."""
    edited = panel.copy()
    edited.loc[edited.index[0], "ransomware_family"] = "zzz_changed"
    assert cache_module.frame_fingerprint(panel) != cache_module.frame_fingerprint(edited)


# --------------------------------------------------------------------------
# The residual defect: the memoised fingerprint goes stale
# --------------------------------------------------------------------------
def test_panel_fingerprint_is_not_inherited_by_a_modified_copy(panel) -> None:
    """pandas propagates .attrs across .copy(), so the memo outlives its panel.

    ``panel_fingerprint`` returns the cached value from ``panel.attrs`` without
    checking it still describes the frame. A derived panel therefore reports its
    parent's digest, which is the same collision R2 found, reintroduced one level
    down.
    """
    parent_digest = panel_fingerprint(panel)

    derived = panel.copy()
    derived.loc[derived.index[:5], "ransomware_family"] = "zzz_changed"

    assert cache_module.frame_fingerprint(derived) != parent_digest, "fixture is inert"
    assert panel_fingerprint(derived) != parent_digest, (
        "the derived panel reports its parent's fingerprint, so run_cell would "
        "serve the parent's cached record for a different panel"
    )


def test_cached_record_matches_the_panel_it_was_asked_about(
    panel, panel_with_novel_family
) -> None:
    """End to end consequence of the stale memo.

    The novel-family panel has twelve test rows whose family is unseen in
    training, so under A2 it must report a smaller n_test. Reading the plain
    panel's record back hides that entirely.
    """
    flags = ControlFlags(a2_labels=True)
    cfg = ExperimentConfig(
        panel=PanelSpec(),
        split=SplitSpec(design="temporal", cut_year=2019),
        flags=flags,
        model_name="logreg",
        seed=0,
        n_features=20,
    )
    plain = run_cell(panel, cfg)
    novel = run_cell(panel_with_novel_family, cfg)
    assert novel["n_test"] < plain["n_test"], (
        "the novel-family panel reported n_test={0}, identical to the plain "
        "panel; a stale cached record was served".format(novel["n_test"])
    )
