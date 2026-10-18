"""Optional reference solver (tests skip without it).

``fronttrack`` (https://github.com/parasuramv/scalar-front-tracking) is used
if installed, else from a checkout named by HOMO_FRONT_TRACKING_PATH. It
provides the real-line ``FrontTracker`` and, in ``fronttrack.periodic``, the
circle ``PeriodicFrontTracker`` used as the S¹ oracle.
"""
import importlib
import os
from pathlib import Path
import sys


def fronttrack():
    try:
        return importlib.import_module("fronttrack")
    except ModuleNotFoundError:
        pass
    checkout = os.environ.get("HOMO_FRONT_TRACKING_PATH")
    if not checkout:
        return None
    src = Path(checkout).expanduser().resolve() / "src"
    if not (src / "fronttrack" / "__init__.py").is_file():
        return None
    sys.path.insert(0, str(src))
    return importlib.import_module("fronttrack")


def circle_fronttrack():
    """``fronttrack`` if it ships the circle tracker (fronttrack.periodic), else None."""
    module = fronttrack()
    return module if module is not None and hasattr(module, "PeriodicFrontTracker") else None
