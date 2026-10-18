"""Exact straightening and error measurements for well-signed periodic g.

For h(u) = |u|^p (p = 2 by default) write eta = sign(g),
m = mean(|g|**(-1/p)), rho = 1/(m |g|**(1/p)). The increasing coordinate
z(x) = origin + integral_origin^x rho satisfies z(x+L) = z(x)+L. Substitution
u(x,t) = rho(x) v(z(x),t) turns the PDE into v_t + (c |v|^p)_z = 0, c = eta/m^p.
This holds for either sign of g, including shocks: z' times the physical
Rankine-Hugoniot speed is the homogeneous speed. The flux levels agree,
sgn(u) g |u|^p = sgn(v) c |v|^p, so integer levels map unchanged.

This identity is also exact for the quantised scheme. Map the initial front
positions by z and keep their integer theta levels unchanged. Comparing that
homogeneous run with the heterogeneous run is a strong solver regression test.
It is distinct from homogenisation, whose macroscopic initial jumps stay at x.
"""

import math

import numpy as np

from .heterogeneity import PiecewiseAffine
from .trajectory import PowerLaw


class PeriodicStraightening:
    """Closed-form coordinate map of a nonvanishing periodic interpolant.

    All integrals of |g|^{-1/p} use dx/M(σ_left, σ_right), with σ = |g|^{1-1/p}
    and M the two-point mean of ``trajectory.PowerLaw`` (for p = 2,
    2 dx/(sqrt|g_left| + sqrt|g_right|)). This avoids both quadrature error
    and division by cell slopes, including on constant cells. The effective
    coefficient belongs to this interpolant, not to a smooth function from
    which its nodal values may have been sampled.
    """

    def __init__(self, g, power=2.0):
        if not isinstance(g, PiecewiseAffine) or not g.periodic or not g.is_well_signed:
            raise ValueError("straightening requires periodic g with no zeros")
        self.g, self.law = g, PowerLaw(power)
        self.power = self.law.p
        self._sigma_left = self.law.sigma_of(g.absg_left)
        lengths = g.right - g.left
        weights = lengths / self.law.mean_array(self._sigma_left, self.law.sigma_of(g.absg_right))
        total = math.fsum(weights.tolist())
        self.mean_inverse_root = total / g.period  # mean of |g|^{-1/p}
        self.coefficient = float(g.sign[0]) / self.mean_inverse_root ** self.power
        self._cumulative = np.r_[0., np.cumsum(weights)] / self.mean_inverse_root
        self._cumulative[-1] = g.period  # close the chart exactly

    def density(self, x):
        """Stationary corrector rho, positive with period mean exactly one."""
        return 1 / (self.mean_inverse_root * np.abs(self.g(x)) ** (1 / self.power))

    def coordinate(self, x):
        """Evaluate z(x) on any finite lift, preserving whole-period offsets."""
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("coordinates must be finite")
        g = self.g
        turns = np.floor((x - g.origin) / g.period)
        local = g.wrap(x)
        cells = g.cell_of(local)
        distance = local - g.left[cells]
        weight = distance / self.law.mean_array(self._sigma_left[cells], self.law.sigma_of(np.abs(g(local))))
        result = g.origin + turns * g.period + self._cumulative[cells] + weight / self.mean_inverse_root
        return float(result) if result.ndim == 0 else result

    def displacement_bound(self):
        """Maximum |z(x)-x| on a period, from all possible extrema.

        On a cell the derivative is rho-1. An interior extremum therefore
        occurs where |g|=1/m^p; since |g| is affine there is at most one unless
        the displacement is constant. Nodes cover that flat case.
        """
        g = self.g
        target = 1 / self.mean_inverse_root ** self.power
        candidates = list(g.nodes) + [g.origin + g.period]
        for c in range(g.n_cells):
            if g.alpha[c] != 0:
                x = g.left[c] + (target - g.absg_left[c]) / g.alpha[c]
                if g.left[c] < x < g.right[c]:
                    candidates.append(x)
        x = np.asarray(candidates)
        return float(np.max(np.abs(self.coordinate(x) - x)))

    def corrector_l1(self, heterogeneous, homogeneous):
        """Exact integral |u_het(x) - rho(x) u_hom(x)| over one period.

        Both trackers must use this map's coefficient and matching times.
        After splitting at both sets of jumps and at g's nodes, the difference
        is a constant times |g|^{-1/p}, so no dense sampling is necessary.
        This compares the stationary corrector at x, not the exact conjugacy
        v(z(x),t), and is therefore a meaningful homogenisation diagnostic.
        """
        g = self.g
        if heterogeneous.g is not g or not homogeneous.cyclic or heterogeneous.time != homogeneous.time:
            raise ValueError("trackers must share the coefficient map and comparison time")
        if heterogeneous.flux.power != self.power or homogeneous.flux.power != self.power:
            raise ValueError("trackers must use this map's power p")
        hg = homogeneous.g
        if (hg.period != g.period or hg.origin != g.origin
                or not np.allclose(hg.values, self.coefficient, rtol=1e-13, atol=0)):
            raise ValueError("homogeneous tracker must have the matching effective coefficient and chart")
        o, end = g.origin, g.origin + g.period
        breaks = np.unique(np.r_[o, end, g.nodes, heterogeneous.positions(), homogeneous.positions()])
        left, right = breaks[:-1], breaks[1:]
        mids = (left + right) / 2
        cells = g.cell_of(mids)
        # u_het = eta*s(theta)/|g|^{1/p}, rho*u_hom = u_hom/(m*|g|^{1/p}).
        amplitude = (g.sign[cells] * heterogeneous.flux.s(heterogeneous.sample_levels(mids))
                     - homogeneous.sample_u(mids) / self.mean_inverse_root)
        law = self.law
        weights = (right - left) / law.mean_array(law.sigma_of(g.abs_g_array(left, cells)),
                                                  law.sigma_of(g.abs_g_array(right, cells)))
        return math.fsum((np.abs(amplitude) * weights).tolist())
