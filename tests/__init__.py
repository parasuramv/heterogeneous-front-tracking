"""Test package: make ``src/hetfront`` importable from a plain checkout."""
from pathlib import Path
import importlib.util
import sys

if importlib.util.find_spec("hetfront") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
