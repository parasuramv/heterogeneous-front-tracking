"""Mesh orders corrected for the finite reference: fit e(h) = C (h^r - h_ref^r).

A run compared with a reference whose own mesh error has the same sign
measures about C (h^r - h_ref^r), not C h^r. With a reference only a few
halvings finer, that steepens the finest local slopes. This fits r and C by
least squares in log e over a grid in r, with h_ref the reference mesh width
of each record.

    python tools/fit_reference_order.py docs/data/convergence-2026-11-06 [more directories]
"""
import json
import math
from pathlib import Path
import sys

import numpy as np


def corrected_order(h, err, h_ref):
    h, err = np.asarray(h, float), np.asarray(err, float)
    best = (math.inf, math.nan)
    for r in np.arange(0.5, 3.0001, 0.0005):
        basis = np.log(h ** r - h_ref ** r)
        logc = np.mean(np.log(err) - basis)
        res = float(np.sum((np.log(err) - logc - basis) ** 2))
        if res < best[0]:
            best = (res, float(r))
    return best[1], math.sqrt(best[0] / len(h))


def reference_width(record):
    """Mesh width of the reference: span over its cells (records before 'cells' count uniform nodes)."""
    ref, row = record["reference"], record["mesh"][0]
    periodic = record["problem"].startswith("circle")
    def cells(entry):
        return entry.get("cells") or (entry["nodes"] if periodic else entry["nodes"] - 1)
    return row["h"] * cells(row) / cells(ref)


def main(directories):
    for directory in directories:
        for path in sorted(Path(directory).glob("convergence_*.json")):
            record = json.loads(path.read_text())
            rows = record["mesh"]
            h, err = [r["h"] for r in rows], [r["theta_l1"] for r in rows]
            h_ref = reference_width(record)
            r, rms = corrected_order(h, err, h_ref)
            plain = float(np.polyfit(np.log(h), np.log(err), 1)[0])
            mesh = record["reference"].get("mesh", "uniform")
            label = record.get("flux") or f"|u|^{record['power']:g}"
            print(f"{record['problem']:22s} {label:14s} {mesh:8s} fitted {plain:5.2f}  "
                  f"reference-corrected {r:5.2f} (rms log residual {rms:.3f}, h_ref {h_ref:.3g})")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__.strip())
    main(sys.argv[1:])
