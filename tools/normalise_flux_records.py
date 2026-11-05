"""Normalise general-h convergence records to the current metadata layout.

Records written before the 2026-11-05 metadata fix carried the nonlinearity
under 'h' at the top level and in the reference, and 'power': 2.0 (the
dataclass default) throughout. In the rows, 'h' was always the mesh width.
This rewrites such JSON files with 'flux' labels and 'power': null, and
regenerates their CSV companions from the JSON rows. Measured values are not
touched. The script is idempotent.

    python tools/normalise_flux_records.py docs/data/convergence-2026-11-05
"""
import csv
import json
from pathlib import Path
import sys


def normalise(path):
    data = json.loads(path.read_text())
    label = data.get("flux") or data.get("h")
    if not isinstance(label, str):
        return False
    out = {}
    for key, value in data.items():
        if key == "h":
            continue
        out[key] = value
        if key == "power":
            out["power"], out["flux"] = None, label
    reference = {}
    for key, value in data["reference"].items():
        if key == "h":
            continue
        reference[key] = None if key == "power" else value
        if key == "power":
            reference["flux"] = label
    out["reference"] = reference
    for kind in ("delta", "mesh", "joint"):
        rows = []
        for row in data[kind]:
            new = {}
            for key, value in row.items():
                if key == "flux":
                    continue
                new[key] = None if key == "power" else value
                if key == "power":
                    new["flux"] = label
            rows.append(new)
        out[kind] = rows
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    with open(path.with_suffix(".csv"), "w", newline="") as handle:
        writer = None
        for kind in ("delta", "mesh", "joint"):
            for row in out[kind]:
                record = {"refinement": kind, **row}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(record)
    return True


if __name__ == "__main__":
    for directory in map(Path, sys.argv[1:]):
        for path in sorted(directory.glob("convergence_*_h-*.json")):
            print(("normalised " if normalise(path) else "skipped ") + path.name)
