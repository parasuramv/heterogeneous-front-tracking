"""Empirical convergence of heterogeneous front tracking in δ and Δ.

    python convergence.py                     # all four test problems
    python convergence.py --power 1.5 --power 3
    python convergence.py --flux "u^2+u^4" --problem line-sign-changing
    python convergence.py --problem line-sign-changing --levels 4

For each problem, three refinements are compared with a fine reference run:
δ alone (fine g mesh), Δ alone (fine δ), and both together. The script writes
CSV rows, JSON reference metadata, and ``results/convergence.png`` and
prints fitted log-log orders for exact θ-L¹ and sampled primitive maxima. These
are observations, not proofs. Nonquadratic filenames include the power, or
for a general h a slug such as ``_h-u2+u4``; quadratic filenames retain
their original names.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import _srcpath  # noqa: F401
from hetfront.nonlinearity import Power, parse as parse_h
from hetfront.studies import PROBLEMS, fitted_order, standard_study
from hetfront.trajectory import PowerLaw


def flux_slug(h):
    """Filename-safe form of a nonlinearity: 'u^2 + |u|^3' -> 'u2+u3'; 'u>0: u^2; u<0: u^4' -> 'u2_u4'."""
    text = str(h).replace(" ", "").replace("u>0:", "").replace("u<0:", "").replace(";", "_")
    return text.replace("|", "").replace("^", "").replace(".", "p")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), action="append")
    p.add_argument("--power", type=float, action="append", help="flux exponent p > 1; repeat for multiple powers (default: 2)")
    p.add_argument("--flux", action="append",
                   help="general convex h such as 'u^2+u^4'; repeatable, combined with any --power values")
    p.add_argument("--max-events", type=int, default=1_000_000, help="event limit per run; raise for fine circle references")
    p.add_argument("--samples", type=int, default=801, help="primitive sampling points")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--depth", type=int, default=2, help="extra halvings for the reference")
    p.add_argument("--output", default="results")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    powers = list(dict.fromkeys(args.power or ([] if args.flux else [2.0])))
    try:
        for power in powers:
            PowerLaw(power)
        fluxes = []
        for spec in dict.fromkeys(args.flux or []):
            h = parse_h(spec).validate()
            if isinstance(h, Power):
                powers.append(h.p)
            else:
                fluxes.append(h)
        if args.levels < 2 or args.depth < 1 or args.samples < 2 or args.max_events < 1:
            raise ValueError("levels >= 2, depth >= 1, samples >= 2, and max-events >= 1 are required")
    except ValueError as error:
        p.error(str(error))
    names = list(dict.fromkeys(args.problem or list(PROBLEMS)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    studies = {}
    cases = [(f"p={power:g}", {"power": power}) for power in dict.fromkeys(powers)]
    cases += [(f"h={h}", {"h": h}) for h in fluxes]
    for label, setting in cases:
        for name in names:
            print(f"Running {name}, {label} ({args.levels} levels, reference depth {args.depth})", flush=True)
            problem = replace(PROBLEMS[name], max_events=args.max_events, **setting)
            study = standard_study(problem, levels=args.levels, depth=args.depth, samples=args.samples)
            if "h" in setting:
                key = f"{name}_h-{flux_slug(setting['h'])}"
            else:
                power = setting["power"]
                key = name if power == 2 else f"{name}_p{str(power).replace('.', 'p')}"
            studies[key] = study
            ref = study["reference"]
            print(f"\n{key} ({label}): reference δ = {ref['delta']:.3g}, {ref['nodes']} nodes, "
                  f"{ref['events']} events, {ref['seconds']:.1f} s")
            with open(out / f"convergence_{key}.csv", "w", newline="") as handle:
                writer = None
                for kind in ("delta", "mesh", "joint"):
                    rows = study[kind]
                    x = [r["h"] if kind == "mesh" else r["delta"] for r in rows]
                    print(f"  {kind:6s} order θ-L1 {fitted_order(x, [r['theta_l1'] for r in rows]):5.2f}   "
                          f"primitive {fitted_order(x, [r['primitive'] for r in rows]):5.2f}")
                    for r in rows:
                        record = {"refinement": kind, **r}
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(record), lineterminator="\n")
                            writer.writeheader()
                        writer.writerow(record)
            (out / f"convergence_{key}.json").write_text(json.dumps({
                "problem": name, "power": problem.power if problem.h is None else None,
                "flux": str(parse_h(problem.h)) if problem.h is not None else str(Power(problem.power)),
                "levels": args.levels,
                "depth": args.depth, "samples": args.samples, "max_events": args.max_events, **study,
            }, indent=2, allow_nan=False) + "\n")
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter
        fig, axes = plt.subplots(len(studies), 3, figsize=(14, 3.5 * len(studies)), squeeze=False,
                                 constrained_layout=True)
        for row, name in zip(axes, studies):
            for ax, kind in zip(row, ("delta", "mesh", "joint")):
                rows = studies[name][kind]
                x = np.array([r["h"] if kind == "mesh" else r["delta"] for r in rows])
                for key, style in (("theta_l1", "o-"), ("primitive", "s--")):
                    y = np.array([r[key] for r in rows])
                    ax.loglog(x, y, style, label=f"{key} (order {fitted_order(x, y):.2f})")
                ax.set(xlabel="Δ = mesh width" if kind == "mesh" else "δ",
                       title=f"{name}\n {'Δ' if kind == 'mesh' else 'δ' if kind == 'delta' else 'δ and Δ'} refinement")
                if len(x) <= 8:
                    ax.set_xticks(x, labels=[f"{value:.3g}" for value in x])
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.legend(fontsize=7, frameon=False)
                ax.grid(True, which="both", alpha=0.3)
        fig.savefig(out / "convergence.png", dpi=120)
        print(f"\nsaved {out / 'convergence.png'}")


if __name__ == "__main__":
    main()
