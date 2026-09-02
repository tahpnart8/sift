# SIFT

Code and results for **SIFT: Decomposing Temporal Performance Degradation in
Ransomware Family Classification**.

Moving a ransomware family classifier from a random split to a temporal split
costs 0.450 macro-F1 on MLRan. SIFT asks how much of that number is the
evaluation protocol rather than the world changing. It turns four candidate
confounds into binary counterfactual interventions, runs all 16 of their
combinations, and splits the gap with an exact Shapley-Shorrocks decomposition
into four contributions plus a residual.

This repository holds the code, the configuration and every result table the
paper draws a number from. It proposes no new detector and no drift
countermeasure.

## Headline numbers

| Quantity | Value | Source |
|---|---|---|
| Degradation, matched random to temporal, macro-F1 | 0.450 | `results/metrics.parquet` |
| Explained by the four controls, v(N) | 0.219, that is 48.7% | `results/shapley_2024/decomposition_as_reported_macro_f1.parquet` |
| Largest single contribution, family novelty φ_A2 | 0.1227 | same |
| Residual R, unexplained | 0.2310 | `results/shapley_2024/residual_ci.parquet` |
| Explained share with duplicates added as a fifth control | 55.5% | `results/c1/decomp_c1.parquet` |

The residual is what the four controls do not explain. It is **not** an upper
bound on concept drift, and it is not named concept drift: the input
distribution shifts measurably, the conditional distribution is not identifiable
under this design. See [docs/limitations.md](docs/limitations.md).

Scale: 1,425 samples, 34 families, 483 binary features, 5 cut-points from 2015
to 2023, 4 models, 5 seeds, 3,400 training runs.

## Quickstart

Python 3.11 or newer.

```bash
git clone <this repository>
cd sift
pip install -e ".[dev]"
pytest                      # 397 tests; those needing raw MLRan skip without it
```

Library versions are pinned exactly rather than by lower bound in
`pyproject.toml`. The paper reports exact metric values, and tree ensembles and
boosted models are not guaranteed bit-identical across releases, so the version
set is part of the result.

Reproducing the results additionally needs the MLRan dataset, which is not
redistributed here. See [docs/data.md](docs/data.md) for what to obtain and
where to put it, then [docs/reproduce.md](docs/reproduce.md) for the run order.
The full chain is roughly 5 hours on one CPU machine.

```bash
make all                    # everything, in dependency order
make test                   # test suite only
make analysis               # re-derive every table from artefacts already present
```

`make analysis` refits nothing. It is the useful target if you want to check the
paper's arithmetic without spending the hours.

## Layout

```
sift/          the package: controls, splits, lattice, Shapley, diagnostics
scripts/
  run/         training runs, hours
  analysis/    derives the reported tables from artefacts, no refitting
  evidence/    supporting findings, one script per finding
  checks/      recomputes the paper's own numbers by an independent route
results/       every table the paper cites, as parquet
tests/         397 tests pinning the invariants the paper depends on
notebooks/     three narrative notebooks over the same package
docs/          everything below
```

## Documentation

| Document | What it answers |
|---|---|
| [docs/reproduce.md](docs/reproduce.md) | What to run, in what order, how long it takes |
| [docs/results-map.md](docs/results-map.md) | Which file every table and number in the paper comes from |
| [docs/data.md](docs/data.md) | Where MLRan comes from, what preprocessing does, why no malware is redistributed |
| [docs/limitations.md](docs/limitations.md) | What this repository cannot support, and which artefacts have no runner |
| [docs/appendix-results.md](docs/appendix-results.md) | Results the 12-page limit pushed out of the paper |
| [sift/CONTRACT.md](sift/CONTRACT.md) | The module contract: four players, seeding, metric, result schema |

Read `docs/limitations.md` before quoting any number from here. It is short and
it is the part of the documentation most likely to change your reading.

## Data and malware

No malware is redistributed in any form: no binaries, no hashes to look them up
with, no raw sandbox traces. The only input is the published extracted feature
representation from MLRan. Full statement in [docs/data.md](docs/data.md).

## Citing

Cite the paper for the framework and this repository for the code. `CITATION.cff`
carries the machine-readable form. Cite the dataset separately:

> F. C. Onwuegbuche, S. O. Adelodun, A. D. Jurcut, and L. Pasquale, "MLRan: A
> behavioural dataset for ransomware analysis and detection," *Journal of Network
> and Computer Applications*, vol. 250, art. no. 104475, 2026,
> doi:10.1016/j.jnca.2026.104475.

## License

MIT, see [LICENSE](LICENSE). The MLRan dataset is not covered by it and carries
its authors' own terms.
