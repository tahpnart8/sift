"""Recovery check: inject one known confound per panel, see if SIFT finds it.

The arithmetic self-check proves the decomposition adds up. It does not prove
that phi_a2_labels measures family novelty rather than something correlated with
it. This runner builds semi-synthetic panels from the real MLRan panel in which
every confound is absent by construction, injects exactly one at a known
magnitude, and asks whether the decomposition recovers it.

Six panels: a null panel that fixes the noise floor, one per control, and one
double-injecting A2 and B2 to test whether the interaction reproduces.

Scope is deliberately narrow and much narrower than the main decomposition:
one model, one cut, three seeds, 576 fits in total, roughly 25 minutes.

    python scripts/run/recovery.py
    python scripts/run/recovery.py --panels null a2_only --out /tmp/probe

Writes recovery.parquet, dividends.parquet and fits.parquet into
results/recovery/.

It does not produce a1_hypotheses.parquet. That sweep over the three A1
injection configurations has no runner in this repository; see
docs/limitations.md.
"""

from __future__ import annotations

import argparse
import time

from sift.config import PanelSpec
from sift.data import build_panel, load_mlran
from sift.recovery import PANEL_NAMES, RECOVERY_DIR, run_recovery, write_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--panels",
        nargs="+",
        default=list(PANEL_NAMES),
        choices=list(PANEL_NAMES),
        help="panels to run, default all six",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"output directory, default {RECOVERY_DIR}",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the deals")
    parser.add_argument("--metric", default="macro_f1", help="metric to decompose")
    args = parser.parse_args()

    spec = PanelSpec()
    panel, provenance = build_panel(load_mlran(), spec)
    print(f"panel: {len(panel)} rows, provenance {provenance}", flush=True)
    print(f"panels: {', '.join(args.panels)}", flush=True)

    started = time.time()
    frames = run_recovery(
        panel,
        spec,
        panel_names=tuple(args.panels),
        seed=args.seed,
        metric=args.metric,
        progress=True,
    )
    elapsed = time.time() - started

    # run_recovery names the per-fit frame "metrics"; the artefact this repository
    # ships and that docs/results-map.md points at is called fits.parquet.
    frames = {("fits" if k == "metrics" else k): v for k, v in frames.items()}

    written = write_recovery(frames, args.out) if args.out else write_recovery(frames)
    for name, path in written.items():
        print(f"wrote {path}  {frames[name].shape}", flush=True)
    print(f"done in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
