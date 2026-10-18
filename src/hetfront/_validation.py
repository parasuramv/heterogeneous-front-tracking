"""Shared validation for lattice levels and mesh sizes."""

import numbers

import numpy as np


def positive_count(value, name):
    """Reject fractional/bool counts instead of silently truncating a mesh."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def integer_levels(values):
    """Validate a one-dimensional integer lattice input before conversion.

    A fractional level is a different initial condition, not a harmless type
    conversion. Float inputs are accepted only when exactly integral. The open
    upper bound avoids float64 rounding int64.max up to the overflow boundary.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError("levels must be a one-dimensional array of finite integers")
    if values.dtype.kind == "f":
        invalid = (not np.all(np.isfinite(values)) or np.any(values != np.floor(values))
                   or np.any(values < -(2. ** 63)) or np.any(values >= 2. ** 63))
    elif values.dtype.kind == "u":
        invalid = np.any(values > np.iinfo(np.int64).max)
    else:
        invalid = False  # signed NumPy integer dtypes fit in int64
    if invalid:
        raise ValueError("levels must be finite integers representable as int64")
    return values.astype(np.int64)
