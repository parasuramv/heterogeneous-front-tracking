"""Heterogeneous front tracking for u_t + (g(x) h(u))_x = 0 with piecewise-affine g.

h(u) = u² by default, |u|^p with p > 1 (``MultiplicativeFlux(..., power=p)``), or
a general convex h such as u² + u⁴, possibly with different branches for u > 0
and u < 0 (``MultiplicativeFlux(..., h="u^2+u^4")``, ``h="u^2; u^2+u^4"``).
The derivative of g is discretised (g_Δ is the nodal interpolant). For powers,
fronts on each cell are straight lines in σ = |g_Δ|^{(p-1)/p}; for p = 2,
σ = sqrt|g_Δ| and w = σu solves Burgers' equation, and event times come from
linear equations. For a general h no coordinate straightens the fronts.
Arrival times are closed forms, and positions and meeting times are monotone
roots. Corners of g_Δ emit no waves; zeros of g are absorbing interfaces.

Public API (read the modules in this order):

- ``PiecewiseAffine``      (heterogeneity.py)  g_Δ on the line or a circle
- ``MultiplicativeFlux``   (flux.py)           g_Δ h(u) with flux levels θ = kδ
- ``trajectory``, ``PowerLaw`` (trajectory.py) closed-form front laws
- ``GeneralLaw``, ``nonlinearity`` (trajectory.py, nonlinearity.py) general convex h
  of power-type growth: ``Power``, ``AbsPolynomial``, ``Custom``, ``Asymmetric``
- ``solve_riemann``        (riemann.py)        shock or δ-fan between levels
- ``Front``, ``Interface`` (fronts.py)
- ``HeteroFrontTracker``   (solver.py)         whole-line driver
- ``PeriodicHeteroFrontTracker``, ``Circle`` (periodic.py, domains.py)
- ``visualization``        (matplotlib plots, imported on demand)
"""
from .heterogeneity import PiecewiseAffine
from .flux import MultiplicativeFlux
from .riemann import solve_riemann
from .fronts import Front, Interface
from .solver import HeteroFrontTracker, Piece
from .domains import Circle
from .periodic import PeriodicHeteroFrontTracker
from . import trajectory
from .trajectory import PowerLaw, GeneralLaw
from . import nonlinearity

__version__ = "0.1.0"
__all__ = [
    "PiecewiseAffine", "MultiplicativeFlux", "solve_riemann", "Front", "Interface",
    "HeteroFrontTracker", "Piece", "Circle", "PeriodicHeteroFrontTracker", "trajectory", "PowerLaw",
    "GeneralLaw", "nonlinearity",
]
