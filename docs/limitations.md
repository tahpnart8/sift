# What this repository cannot support

[Back to README](../README.md)

Read this before quoting a number from here. Everything below is a known limit,
established by us rather than found by a reviewer.

## The residual is not concept drift, and not a bound on it

`R` is what the four controls do not explain. Two readings are wrong.

It is not concept drift. The design cannot identify a change in `p(y|x)`: the
pre-registered conditional test was refuted by our own counterexample, and its
valid replacement, importance-weighted risk, returned inconclusive at all three
cuts it could run because the effective sample size after weighting falls to
2.4 to 3.5%, below the 10% support guard.

It is not an upper bound on drift either. That reading needs every uncontrolled
component to be non-negative, and `phi_b1_fs` is negative in most cells. An
earlier version of this work stated the upper-bound reading; it has been
withdrawn, and any text still carrying it is out of date.

What the data supports is narrower: a residual the four controls do not explain,
alongside a measurable input shift, with no way to say how much of the one is
the other.

## The framework validation is narrower than the main result

The recovery check runs one model, one cut and three seeds, 576 fits. The main
decomposition runs four models, five cuts and five seeds, 3,400 fits. Do not
read the validation as covering the decomposition's parameter space.

**Selectivity failed its own criterion.** The False Attribution Rate is 6 of 36
uninjected panel-control pairs under the strict definition, 16.7%, or 30.0% over
the 20 non-degenerate pairs. The pass criterion was fixed before we saw the
results, so this is recorded as a failure rather than softened. A1 is therefore
only partially confirmed.

**The drift injection scale is degenerate by construction.** R rises
monotonically with injected drift, which shows the measurement path is unbiased
under no drift and sensitive to drift magnitude. It does not show that R isolates
drift while the four confounds are active, because on those panels all four
controls are inert and R equals Delta exactly.

## Reproducibility is verified on one cell of twenty

Bit-for-bit reproduction was confirmed for cut 2019 with Logistic Regression, all
16 coalitions and 5 seeds. The other nineteen cells rest on the recorded
configuration, not on a re-run.

The exact definition of the bootstrap used in one earlier run could not be
recovered from either the code or the artefacts. The count of 58 of 80
contributions containing zero was rebuilt from the current production components
and does not compare one-to-one against that earlier figure.

## Four artefacts have no runner

Every other file in `results/` is produced by a script in `scripts/` or by
`python -m sift.driftsim`. These four are not.

`results/shapley_2024/decomposition_common_macro_f1.parquet` and
`comparison_old_vs_new.parquet` come from a one-off script written during an
earlier migration. Both compare two runs of the panel rather than producing a
reported result, and neither is the source of any number in the paper. They are
kept because [appendix-results.md](appendix-results.md) discusses the two-run
comparison.

`results/recovery/a1_hypotheses.parquet` and `a1_hypotheses_fits.parquet` come
from a sweep over three A1 injection configurations. `scripts/run/recovery.py`
reproduces the six standard recovery panels but not this sweep, so the two files
stand on their recorded configuration rather than on a command. They back the
A1 discussion in Section 4.5 of the paper and in
[appendix-results.md](appendix-results.md).

Treat all four as recorded evidence, not as reproducible output.

## Inherited feature selection

The 483 features were selected by the dataset authors with RFE on their own
split, so they carry look-ahead we cannot remove. Control B1 therefore measures
only the leakage we add on top. See [data.md](data.md).

## Scope

One dataset, one task, one preprocessing chain. SIFT here is a framework realised
on a single case study, not a framework whose generality has been demonstrated.
Nothing here transfers to binary ransomware detection: the targets differ,
`p(y = ransomware | x)` against `p(y = family | x)`, and the binary task was
dropped precisely because its gap sat below the noise floor.

Two cut-points do not complete the residual diagnosis. Cut 2017 stops at step
one, cut 2023 at step two with 84 held-out samples against a pre-registered
threshold of 100. Every table marks the 2023 row descriptive only: its test
window holds 31 samples.

Test windows overlap across cut-points. This does not affect the decomposition,
since each cut is a separate experiment, but it does mean AUT is not computable
over this series.

The reference arm matches the test window on sample size and class set, but not
on the training set's class coverage. The list of remaining mismatches between
the two designs is not closed, and we do not claim it is.

## Aggregation rules that are easy to get wrong

Three, all of which change published numbers if broken.

1. The five seeds of one (cut, model) cell are five draws of one quantity.
   Average them before anything else. Treating them as independent groups
   inflates every agreement statistic.
2. Cut-points are never pooled. Five cuts are five independent experiments, not
   five points on a curve.
3. Models are pooled only for RQ1, and always with the range reported alongside.

`sift/CONTRACT.md` fixes these, and `tests/` asserts them.
