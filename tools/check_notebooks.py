"""Execute every notebook with this Python interpreter; ``--write`` saves the outputs.

01 and 02 run in an empty temporary directory, so they must work from their
bundled source alone. 03-08 run inside notebooks/ and use the local checkout.
Saved widget outputs are replaced by a short note, because widget state does
not render without a kernel.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
WIDGET_NOTE = ("Interactive widget: run this notebook in Jupyter or Colab to use it. "
               "Viewers without a kernel (GitHub, VS Code preview) cannot display it.\n")


def strip_volatile(nb):
    nb.metadata.pop("widgets", None)
    for cell in nb.cells:
        if cell.cell_type == "code" and any(WIDGET_MIME in o.get("data", {}) for o in cell.outputs):
            cell.outputs = [o for o in cell.outputs if WIDGET_MIME not in o.get("data", {})]
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=WIDGET_NOTE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("names", nargs="*", help="notebook filename prefixes (default: all)")
    args = parser.parse_args()
    paths = sorted((ROOT / "notebooks").glob("0*.ipynb"))
    if args.names:
        paths = [p for p in paths if any(p.name.startswith(n) for n in args.names)]
    if not paths:
        parser.error("no notebooks match the requested prefixes")
    with tempfile.TemporaryDirectory(prefix="hetfront-check-") as scratch:
        scratch = Path(scratch)
        kernel = scratch / "kernels" / "hetfront-check"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "hetfront check", "language": "python",
            "env": {"MPLCONFIGDIR": str(scratch / "mpl")},
        }))
        failures = 0
        for path in paths:
            bundled = path.name[:2] in ("01", "02")
            work = scratch / path.stem if bundled else path.parent
            work.mkdir(exist_ok=True)
            nb = nbformat.read(path, as_version=4)
            manager = KernelManager(kernel_name="hetfront-check",
                                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
            start = time.perf_counter()
            try:
                NotebookClient(nb, km=manager, timeout=900, record_timing=False,
                               resources={"metadata": {"path": str(work)}}).execute()
            except Exception as error:  # report and continue with the others
                failures += 1
                print("FAIL:", path.name, type(error).__name__, str(error)[:2000], flush=True)
                continue
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            if args.write:
                strip_volatile(nb)
                nbformat.write(nb, path)
            print(f"PASS: {path.name} ({time.perf_counter() - start:.0f} s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
