"""Let scripts in this checkout import ``hetfront`` without installing it.

Prepend this checkout's ``src/`` even when another version is installed.
A checkout script must execute the code beside it, not silently test an older
site-packages copy. Import this before any ``hetfront`` import.
"""
from pathlib import Path
import sys

src = str(Path(__file__).resolve().parent / "src")
if src not in sys.path:
    sys.path.insert(0, src)
