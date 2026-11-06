"""Mesh-only refinement on uniform against zero-aligned g meshes, for the sign-changing problems.

    python tools/plot_alignment.py docs/data/convergence-2026-11-06 docs/convergence-2026-11-06.png \\
        docs/data/convergence-2026-10-15 docs/data/convergence-2026-11-05 docs/data/convergence-2026-11-06/uniform

The first directory holds the aligned studies; the rest hold uniform-mesh
studies of the same problems. Records are matched by problem and
nonlinearity. Each legend entry gives the local slopes, coarse to fine.
"""
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

PROBLEMS = ("line-sign-changing", "circle-sign-changing")


def label(record):
    return record.get("flux") or f"|u|^{record['power']:g}"


def mesh_label(record):
    return record["reference"].get("mesh", "uniform")    # unlabelled records predate aligned meshes


def load(directory):
    out = {}
    for path in sorted(Path(directory).glob("convergence_*.json")):
        record = json.loads(path.read_text())
        if record["problem"] in PROBLEMS:
            out[(record["problem"], label(record))] = record
    return out


def slopes(rows):
    h, err = [r["h"] for r in rows], [r["theta_l1"] for r in rows]
    return h, err, [math.log(err[i] / err[i + 1]) / math.log(h[i] / h[i + 1]) for i in range(len(err) - 1)]


def main(aligned_dir, output, *uniform_dirs):
    aligned = load(aligned_dir)
    uniform = {}
    for directory in uniform_dirs:
        uniform.update(load(directory))
    assert all(mesh_label(r) == "aligned" for r in aligned.values()), "first directory must hold aligned studies"
    assert all(mesh_label(r) == "uniform" for r in uniform.values()), "comparison directories must be uniform"
    fluxes = sorted({f for _, f in aligned}, key=lambda f: (not f.startswith("|u|^") or "+" in f, f))
    fig, axes = plt.subplots(2, len(fluxes), figsize=(3.6 * len(fluxes), 7.2), constrained_layout=True,
                             squeeze=False)
    for row, problem in zip(axes, PROBLEMS):
        for ax, flux in zip(row, fluxes):
            for source, style, colour in ((uniform, "s--", "0.55"), (aligned, "o-", "C0")):
                record = source.get((problem, flux))
                if record is None:
                    continue
                h, err, local = slopes(record["mesh"])
                ax.loglog(h, err, style, color=colour, ms=4,
                          label=f"{mesh_label(record)}: " + ", ".join(f"{s:.2f}" for s in local))
            ax.set_title(f"{problem}\nh = {flux}", fontsize=9)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("mesh width Δ", fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, frameon=False, title="local slopes", title_fontsize=7)
        row[0].set_ylabel("θ-L¹ error, mesh only")
    fig.savefig(output, dpi=120)
    print(f"saved {output}")


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__.strip())
    main(*sys.argv[1:])
