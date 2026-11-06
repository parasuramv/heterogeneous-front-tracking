"""Plot mesh-only refinement for the general-flux studies against the recorded |u|^3 runs.

    python tools/plot_flux_convergence.py docs/data/convergence-2026-11-05 docs/convergence-2026-11-05.png
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hetfront.studies import PROBLEMS, fitted_order  # noqa: E402

COLOURS = ("C3", "C0", "C2", "C1")


def main(directory, output, comparison=ROOT / "docs/data/convergence-2026-10-15"):
    records = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("convergence_*_h-*.json"))]
    fluxes = sorted({r["flux"] for r in records}, reverse=True)
    colours = dict(zip(fluxes, COLOURS))
    cubic = {json.loads(p.read_text())["problem"]: json.loads(p.read_text())
             for p in Path(comparison).glob("convergence_*_p3p0.json")}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, PROBLEMS):
        if name in cubic:
            rows = cubic[name]["mesh"]
            h, err = [r["h"] for r in rows], [r["theta_l1"] for r in rows]
            ax.loglog(h, err, "-", color="0.65", lw=3, alpha=0.6,
                      label=f"|u|^3 (2026-10-15), θ-L1: {fitted_order(h, err):.2f}")
        for record in (r for r in records if r["problem"] == name):
            rows = record["mesh"]
            h = [r["h"] for r in rows]
            for metric, style in (("theta_l1", "o-"), ("primitive", "s--")):
                err = [r[metric] for r in rows]
                ax.loglog(h, err, style, color=colours[record["flux"]], ms=4,
                          label=f"{record['flux']}, {'θ-L1' if metric == 'theta_l1' else 'primitive'}: "
                                f"{fitted_order(h, err):.2f}")
            ax.set_xticks(h, labels=[f"{v:.3g}" for v in h])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(title=f"{name}: mesh-only refinement", xlabel="mesh width Δ", ylabel="error against reference")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5, frameon=False)
    fig.savefig(output, dpi=120)
    print(f"saved {output}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__.strip())
    main(*sys.argv[1:3])
