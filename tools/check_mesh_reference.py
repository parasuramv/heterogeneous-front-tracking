"""Recheck recorded mesh rates with a finer spatial reference and denser sampling.

Keep delta fixed to isolate coefficient-mesh error. Recompute every mesh run
against twice as many reference nodes, sampling its primitive on nested grids.
This checks sensitivity of measured rates, not error against an exact solution.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hetfront.studies import PROBLEMS, errors, fitted_order, reference_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory of baseline convergence JSON files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--power", type=float, action="append", help="optional exponent filter; repeat as needed")
    parser.add_argument("--max-events", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.max_events < 1:
        parser.error("max-events must be positive")
    paths = sorted(args.directory.glob("convergence_*.json"))
    if not paths:
        parser.error("no baseline convergence records found")
    records = []
    for path in paths:
        baseline = json.loads(path.read_text())
        if args.power and baseline["power"] not in args.power:
            continue
        problem = replace(PROBLEMS[baseline["problem"]], power=baseline["power"], max_events=args.max_events)
        settings = baseline["reference"]
        print(f"Checking {problem.name}, p={problem.power:g}", flush=True)
        reference, info = reference_run(problem, settings["delta"], 2 * settings["nodes"],
                                        final_time=settings["final_time"])
        samples = baseline["samples"]
        rows = []
        for old in baseline["mesh"]:
            tracker = problem.tracker(old["delta"], old["nodes"]).run(reference.time)
            theta, primitive = errors(tracker, reference, problem.window, samples=samples)
            _, dense = errors(tracker, reference, problem.window, samples=2 * samples - 1)
            rows.append({"delta": old["delta"], "nodes": old["nodes"], "h": old["h"],
                         "theta_l1": theta, "primitive": primitive, "primitive_dense": dense})
        h = [r["h"] for r in rows]
        orders = {metric: fitted_order(h, [r[metric] for r in rows])
                  for metric in ("theta_l1", "primitive", "primitive_dense")}
        records.append({"baseline": path.name, "problem": problem.name, "power": problem.power,
                        "samples": samples, "dense_samples": 2 * samples - 1,
                        "reference": info, "mesh": rows, "orders": orders})
        # Preserve completed cases if a later expensive reference fails.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(f"{problem.name}, p={problem.power:g}: {orders}", flush=True)

    if not records:
        parser.error("no baseline records match the requested powers")


if __name__ == "__main__":
    main()
