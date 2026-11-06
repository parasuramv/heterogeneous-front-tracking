"""Mesh order and accuracy against the grading exponent γ, for the sign-changing problems.

    python tools/plot_grading.py docs/data/convergence-2026-11-07 docs/convergence-2026-11-07.png

Reads ``graded-g*/`` studies from the given directory and treats the aligned
studies of 2026-11-06 as γ = 1 and its uniform records as the baseline. The top
row plots the reference-corrected mesh order (``fit_reference_order.py``,
finest three levels) against the local prediction min(2, γ(2 - 1/k)). The
bottom row plots the θ-L¹ mesh error at the finest level (256 cells). Also
prints a table of both.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fit_reference_order import corrected_order, reference_width  # noqa: E402

PROBLEMS = ("line-sign-changing", "circle-sign-changing")
TOP_DEGREE = {"|u|^1.5": 1.5, "|u|^2": 2.0, "|u|^3": 3.0, "|u|^2 + |u|^3": 3.0, "|u|^2 + |u|^4": 4.0}
COLOURS = {"|u|^1.5": "C2", "|u|^2": "C0", "|u|^3": "C3", "|u|^2 + |u|^3": "C1", "|u|^2 + |u|^4": "C4"}
ALIGNED = ROOT / "docs/data/convergence-2026-11-06"
UNIFORM = [ROOT / "docs/data/convergence-2026-10-15", ROOT / "docs/data/convergence-2026-11-05",
           ROOT / "docs/data/convergence-2026-11-06/uniform"]


def label(record):
    return record.get("flux") or f"|u|^{record['power']:g}"


def records(directory):
    for path in sorted(Path(directory).glob("convergence_*.json")):
        record = json.loads(path.read_text())
        if record["problem"] in PROBLEMS and label(record) in TOP_DEGREE:
            yield record


def predicted(gamma, k):
    return np.minimum(2.0, gamma * (2.0 - 1.0 / k))


def order(record, finest=3):
    rows = record["mesh"]
    h, err = [r["h"] for r in rows][-finest:], [r["theta_l1"] for r in rows][-finest:]
    return corrected_order(h, err, reference_width(record))[0]


def main(directory, output):
    table = {}      # (problem, flux) -> {γ: (order, finest error)}
    for sub in sorted(Path(directory).glob("graded-g*")):
        for record in records(sub):
            gamma = record["reference"]["grading"]
            table.setdefault((record["problem"], label(record)), {})[gamma] = (order(record),
                                                                             record["mesh"][-1]["theta_l1"])
    for record in records(ALIGNED):
        table.setdefault((record["problem"], label(record)), {})[1.0] = (order(record), record["mesh"][-1]["theta_l1"])
    uniform = {}
    for directory_u in UNIFORM:
        for record in records(directory_u):
            uniform[(record["problem"], label(record))] = record["mesh"][-1]["theta_l1"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    gammas = np.linspace(1, 3.1, 200)
    for col, problem in enumerate(PROBLEMS):
        top, bottom = axes[0, col], axes[1, col]
        for flux, k in TOP_DEGREE.items():
            data = table.get((problem, flux))
            if not data:
                continue
            gs = sorted(data)
            colour = COLOURS[flux]
            top.plot(gs, [data[g][0] for g in gs], "o", color=colour, label=f"h = {flux}")
            if "+" not in flux:
                top.plot(gammas, predicted(gammas, k), "-", color=colour, alpha=0.5, lw=1)
            bottom.semilogy(gs, [data[g][1] for g in gs], "o-", color=colour, label=f"h = {flux}")
            if (problem, flux) in uniform:
                bottom.axhline(uniform[(problem, flux)], color=colour, ls=":", lw=1)
        top.set(title=f"{problem}: mesh order (reference-corrected, finest 3 levels)",
                xlabel="grading exponent γ (1 = aligned)", ylabel="order", ylim=(1.0, 2.3))
        bottom.set(title=f"{problem}: θ-L¹ mesh error at 256 cells (dotted: uniform mesh)",
                   xlabel="grading exponent γ (1 = aligned)", ylabel="error")
        for ax in (top, bottom):
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, frameon=False)
    axes[0, 0].text(1.02, 2.22, "lines: min(2, γ(2 − 1/k)) for |u|^k", fontsize=8)
    fig.savefig(output, dpi=120)
    print(f"saved {output}\n")
    print(f"{'problem':22s} {'h':14s} " + " ".join(f"γ={g:<13g}" for g in (1.0, 1.25, 1.5, 2.0, 3.0)) + " uniform")
    for (problem, flux), data in sorted(table.items()):
        k = TOP_DEGREE[flux]
        cells = []
        for g in (1.0, 1.25, 1.5, 2.0, 3.0):
            cells.append(f"{data[g][0]:.2f}/{float(predicted(g, k)):.2f} {data[g][1]:.1e}" if g in data else "-" * 14)
        base = uniform.get((problem, flux))
        print(f"{problem:22s} {flux:14s} " + " ".join(f"{c:15s}" for c in cells)
              + (f" {base:.1e}" if base else ""))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__.strip())
    main(*sys.argv[1:])
