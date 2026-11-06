"""Mesh-only refinement at several fixed flux spacings δ, each against its own reference.

Tests whether falling mesh slopes at the finest levels come from a floor
set by the fixed δ of the mesh-only runs. For each δ, the reference has the
same δ and ``ref_nodes`` coefficient nodes, so the remaining error is
coefficient-mesh error at that δ. Writes one JSON record per (problem,
nonlinearity, δ).

    python tools/check_delta_floor.py --problem circle-sign-changing --power 3 \\
        --delta 0.0015625 --delta 0.000390625 --output docs/data/convergence-2026-11-05/delta-floor
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hetfront.nonlinearity import parse as parse_h  # noqa: E402
from hetfront.studies import PROBLEMS, fitted_order, reference_run, refinement  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--power", type=float)
    group.add_argument("--flux")
    p.add_argument("--delta", type=float, action="append", required=True)
    p.add_argument("--nodes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--ref-nodes", type=int, default=1024)
    p.add_argument("--samples", type=int, default=801)
    p.add_argument("--max-events", type=int, default=20_000_000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    setting = {"power": args.power} if args.flux is None else {"h": args.flux}
    slug = str(parse_h(args.flux)).replace(" ", "").replace("u>0:", "").replace("u<0:", "").replace(";", "_")
    label = (f"p{str(args.power).replace('.', 'p')}" if args.flux is None else
             "h-" + slug.replace("|", "").replace("^", "").replace(".", "p"))
    problem = replace(PROBLEMS[args.problem], max_events=args.max_events, **setting)
    args.output.mkdir(parents=True, exist_ok=True)
    for delta in args.delta:
        ref, info = reference_run(problem, delta, args.ref_nodes)
        rows = refinement(problem, [(delta, m) for m in args.nodes], ref, samples=args.samples)
        h, err = [r["h"] for r in rows], [r["theta_l1"] for r in rows]
        local = [math.log(err[i] / err[i + 1]) / math.log(h[i] / h[i + 1]) for i in range(len(err) - 1)]
        record = {"problem": args.problem, "setting": label, "delta": delta, "reference": info, "mesh": rows,
                  "theta_order": fitted_order(h, err),
                  "primitive_order": fitted_order(h, [r["primitive"] for r in rows]), "local_slopes": local}
        path = args.output / f"{args.problem}_{label}_delta{delta:.3g}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(f"{args.problem} {label} δ={delta:g}: θ-L1 order {record['theta_order']:.2f}, "
              f"local slopes {[round(s, 2) for s in local]}, reference events {info['events']}", flush=True)


if __name__ == "__main__":
    main()
