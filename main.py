"""Run a named heterogeneous front-tracking example and save its results.

    python main.py --example xu2-blowup
    python main.py --example circle-sine --delta 0.005 --time 12
    python main.py --example two-zeros --power 3
    python main.py --list

Files appear in ``results/`` (or ``--output``):

- ``<name>.png``              g_Δ, θ in space–time with exact fronts, u profile, diagnostics
- ``<name>.npz``              arrays x, times, levels, theta, u, delta, power, nodes, g_values
- ``<name>_pieces.csv``       every closed-form front piece: t0,t1,x0,sigma0,alpha,kappa,kL,kR,cell
- ``<name>_diagnostics.csv``  time, fronts, total variation of θ, mass, absorbed mass, edge inflow, events
"""
import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

import _srcpath  # noqa: F401  (makes src/hetfront importable without installation)
from hetfront.examples import EXAMPLES


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example", default="xu2-blowup", choices=sorted(EXAMPLES))
    p.add_argument("--list", action="store_true", help="describe the examples and exit")
    p.add_argument("--time", type=float, help="final time (default: the example's)")
    p.add_argument("--delta", type=float, help="flux-level spacing δ")
    p.add_argument("--nodes", type=int, help="nodes of the g mesh")
    p.add_argument("--cells", type=int, help="cells used to sample the initial data")
    p.add_argument("--power", type=float, help="p in h(u) = |u|^p, p > 1 (default: the example's)")
    p.add_argument("--interface-layer", type=float, default=0.0, help="θ = 0 within this distance of zeros")
    p.add_argument("--samples", type=int, default=801, help="spatial samples of the colour field")
    p.add_argument("--frames", type=int, default=151, help="observation times")
    p.add_argument("--max-events", type=int, default=2_000_000)
    p.add_argument("--output", default="results")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.list:
        for name in sorted(EXAMPLES):
            print(f"{name:24s} {EXAMPLES[name].description}")
        return 0
    example = EXAMPLES[args.example]
    if args.power is not None and not (math.isfinite(args.power) and args.power > 1):
        raise SystemExit("--power must be finite and greater than 1")
    tracker = example.build(args.delta, args.nodes, args.cells, args.power,
                            interface_layer=args.interface_layer, max_events=args.max_events)
    final = example.final_time if args.time is None else args.time
    if final < 0:
        raise SystemExit("--time must be nonnegative")
    x = np.linspace(*example.window, args.samples, endpoint=not example.periodic)
    times = np.linspace(0.0, final, max(2, args.frames)) if final > 0 else np.array([0.0])

    from hetfront.visualization import observe_tracker
    history = observe_tracker(tracker, x, times)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / example.name
    g = tracker.g
    np.savez_compressed(stem.with_suffix(".npz"), x=x, times=times, levels=history.levels,
                        theta=history.theta, u=history.u, delta=tracker.flux.delta,
                        power=tracker.flux.power,
                        nodes=g.nodes, g_values=g.values)
    with open(f"{stem}_pieces.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t0", "t1", "x0", "sigma0", "alpha", "kappa", "kL", "kR", "cell"])
        writer.writerows(history.pieces)
    with open(f"{stem}_diagnostics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        d = history.diagnostics
        writer.writerow(["time", "fronts", "total_variation", "mass", "absorbed", "inflow", "events"])
        writer.writerows(zip(times, d["fronts"], d["total_variation"], d["mass"], d["absorbed"],
                             d["inflow"], d["events"]))

    print(example.description)
    print(f"g mesh: {len(g.nodes)} nodes, {len(g.zeros)} zeros; δ = {tracker.flux.delta:g}; "
          f"h(u) = |u|^{tracker.flux.power:g}; t = {final:g}")
    print("events:", ", ".join(f"{k} {v}" for k, v in tracker.counts.items() if v))
    print(f"fronts: {history.diagnostics['fronts'][0]} -> {history.diagnostics['fronts'][-1]}")
    d = history.diagnostics
    budget = d["mass"] + d["absorbed"] - d["inflow"]
    print(f"mass budget residual (∫u + absorbed − inflow): {budget[-1] - budget[0]:.3e}")

    if not args.no_plot and len(times) > 1:
        import matplotlib
        matplotlib.use("Agg")
        from hetfront.visualization import overview
        characteristics = example.characteristics(tracker.flux, final) if example.characteristic_starts else None
        fig = overview(history, exact_g=example.g, characteristics=characteristics,
                       title=f"{example.name}: {example.description}")
        fig.savefig(stem.with_suffix(".png"), dpi=130)
        print(f"saved {stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
