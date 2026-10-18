"""Continuous piecewise-affine heterogeneity g_Δ on the line or on a circle.

The *derivative* of g is discretised: on each cell g′_Δ is the cell average of
g′, so g_Δ is the nodal interpolant of g. Every zero of g_Δ is a node, hence
each open cell carries a single sign of g_Δ and the closed-form front laws of
``trajectory.py`` apply cell by cell.

Standing assumption (A2): zeros are isolated and transverse, |g′(z±)| ≥ β > 0.
Tangential zeros are rejected, because linear interpolation would silently
turn them into simple zeros and change the analysis near the interface.

Cells on the line are numbered 0..M for nodes n_0 < ... < n_{M-1}: cell 0 is
(-inf, n_0), cell c is [n_{c-1}, n_c), and cell M is [n_{M-1}, inf). Outside
the nodes g_Δ is extended either as a constant or affinely. On a circle of
period L the cells are [n_c, n_{c+1}) for c = 0..M-1 with n_M = n_0 + L, and
the canonical chart is [n_0, n_0 + L).
"""

import math

import numpy as np

from ._validation import positive_count


def _readonly(array):
    array = np.array(array, dtype=float, copy=True)
    array.flags.writeable = False
    return array


def _linear_root(x0, g0, x1, g1):
    """Zero of the chord through (x0,g0), (x1,g1); requires g0*g1 < 0."""
    return x0 + (x1 - x0) * (g0 / (g0 - g1))


class PiecewiseAffine:
    """Continuous piecewise-affine g_Δ with every zero a node.

    ``nodes`` must be strictly increasing; ``values`` are g at the nodes.
    Sign changes between nodes are resolved by inserting the chord's zero as a
    node. A node with value zero must be a genuine sign change. ``beta``, when
    given, is a lower bound on |slope| for the cells on both sides of a zero.

    ``period=L`` makes a circle; nodes must then lie in [n_0, n_0 + L).
    ``extend`` ("constant" or "affine") sets g_Δ beyond the outer nodes on the
    line; an affine extension that would change sign gets that zero inserted.
    """

    def __init__(self, nodes, values, *, period=None, extend="constant", beta=None):
        nodes = np.array(nodes, dtype=float, copy=True)
        values = np.array(values, dtype=float, copy=True)
        if nodes.ndim != 1 or nodes.shape != values.shape or len(nodes) < 2:
            raise ValueError("nodes and values must be 1-D arrays of equal length >= 2")
        if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(values))):
            raise ValueError("nodes and values must be finite")
        if np.any(np.diff(nodes) <= 0):
            raise ValueError("nodes must be strictly increasing")
        if extend not in ("constant", "affine"):
            raise ValueError("extend must be 'constant' or 'affine'")
        self.periodic = period is not None
        if self.periodic:
            period = float(period)
            if not math.isfinite(period) or period <= 0:
                raise ValueError("period must be positive and finite")
            if not math.isfinite(nodes[0] + period) or nodes[-1] >= nodes[0] + period:
                raise ValueError("periodic nodes must lie in a finite chart [n_0, n_0 + period)")
            extend = None
        self.period = period
        self.extend = extend

        nodes, values = self._insert_zeros(nodes, values)
        self._check_zeros(nodes, values)
        self.nodes, self.values = _readonly(nodes), _readonly(values)
        self._build_cells()
        if beta is not None:
            self._check_beta(beta)

    # ------------------------------------------------------------------ setup
    def _insert_zeros(self, nodes, values):
        xs, gs = [nodes[0]], [values[0]]
        ends = list(zip(nodes[1:], values[1:]))
        if self.periodic:
            ends.append((nodes[0] + self.period, values[0]))
        for x1, g1 in ends:
            x0, g0 = xs[-1], gs[-1]
            if g0 * g1 < 0:
                xs.append(_linear_root(x0, g0, x1, g1))
                gs.append(0.0)
            xs.append(x1)
            gs.append(g1)
        if self.periodic:
            xs.pop()
            gs.pop()  # the closing node is n_0 + L again
        nodes, values = np.array(xs), np.array(gs)
        if self.extend == "affine":
            # Outer slopes continue the first and last chords. Insert the zero
            # if the extension crosses it at a finite distance.
            a_left = (values[1] - values[0]) / (nodes[1] - nodes[0])
            a_right = (values[-1] - values[-2]) / (nodes[-1] - nodes[-2])
            if values[0] != 0 and a_left != 0 and values[0] / a_left > 0:
                z = nodes[0] - values[0] / a_left
                nodes = np.concatenate(([2 * z - nodes[0], z], nodes))
                values = np.concatenate(([-values[0], 0.0], values))
            if values[-1] != 0 and a_right != 0 and values[-1] / a_right < 0:
                z = nodes[-1] - values[-1] / a_right
                nodes = np.concatenate((nodes, [z, 2 * z - nodes[-1]]))
                values = np.concatenate((values, [0.0, -values[-1]]))
        return nodes, values

    def _check_zeros(self, nodes, values):
        n = len(nodes)
        for i in np.flatnonzero(values == 0):
            if self.periodic:
                left, right = values[(i - 1) % n], values[(i + 1) % n]
            else:
                if i in (0, n - 1) and self.extend == "constant":
                    raise ValueError("g must not vanish at an outer node with constant extension")
                left = values[i - 1] if i > 0 else -values[1]  # affine continuation
                right = values[i + 1] if i < n - 1 else -values[n - 2]
            if not left * right < 0:
                raise ValueError(f"zero of g at x={nodes[i]:.6g} is not a transverse sign change")

    def _build_cells(self):
        nodes, values = self.nodes, self.values
        if self.periodic:
            right_nodes = np.append(nodes[1:], nodes[0] + self.period)
            right_values = np.append(values[1:], values[0])
            left, right, gl, gr = nodes, right_nodes, values, right_values
            slope = (gr - gl) / (right - left)
        else:
            inner_slope = np.diff(values) / np.diff(nodes)
            if self.extend == "affine":
                outer = (inner_slope[0], inner_slope[-1])
            else:
                outer = (0.0, 0.0)
            left = np.concatenate(([-np.inf], nodes))
            right = np.concatenate((nodes, [np.inf]))
            slope = np.concatenate(([outer[0]], inner_slope, [outer[1]]))
            gl = np.concatenate(([np.nan], values))
            gr = np.concatenate((values, [np.nan]))
        # Sign of g_Δ on the open cell: from whichever endpoint is nonzero.
        ref = np.where(np.isfinite(gl) & (gl != 0), gl, gr)
        sign = np.sign(ref)
        if not self.periodic:
            # An outer cell whose only finite endpoint is a zero (affine case).
            for c, inner in ((0, 1), (len(sign) - 1, len(sign) - 2)):
                if sign[c] == 0 or not np.isfinite(sign[c]):
                    # At a rightmost zero, gl=0 and gr=nan (infinite endpoint),
                    # so the endpoint fallback above is nan, not zero.
                    sign[c] = -sign[inner]
        if np.any(sign == 0):
            raise ValueError("g vanishes on a whole cell")
        self.left, self.right = _readonly(left), _readonly(right)
        self.slope = _readonly(slope)
        self.sign = _readonly(sign)
        self.alpha = _readonly(sign * slope)  # d|g|/dx on each cell
        self.absg_left, self.absg_right = _readonly(np.abs(gl)), _readonly(np.abs(gr))
        self.sigma_left = _readonly(np.sqrt(np.abs(gl)))
        self.sigma_right = _readonly(np.sqrt(np.abs(gr)))
        self.zero_left = np.asarray(gl == 0)
        self.zero_right = np.asarray(gr == 0)
        self.zero_left.flags.writeable = self.zero_right.flags.writeable = False
        self.zeros = _readonly(nodes[values == 0])

    def _check_beta(self, beta):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        for c in range(self.n_cells):
            if (self.zero_left[c] or self.zero_right[c]) and abs(self.slope[c]) < beta:
                raise ValueError(f"|g'| = {abs(self.slope[c]):.3g} < beta = {beta:.3g} next to a zero")

    # ----------------------------------------------------------- constructors
    @classmethod
    def constant(cls, value, *, period=None):
        """g ≡ value (nonzero): the homogeneous flux value·h(u)."""
        if value == 0:
            raise ValueError("constant g must be nonzero")
        nodes = [0.0, 0.5 * period] if period else [0.0, 1.0]
        return cls(nodes, [value, value], period=period)

    @classmethod
    def from_function(cls, g, nodes, *, dg=None, beta=1e-6, period=None,
                      extend="constant", oversample=16):
        """Interpolate g at ``nodes`` after inserting its true zeros as nodes.

        Zeros are located by bisection on sign changes detected on an
        ``oversample``-times finer grid, so two zeros inside one cell are not
        missed when the fine grid separates them. Each zero must satisfy
        |g′(z)| ≥ beta, using ``dg`` if supplied and a central difference
        otherwise. A sampled local minimum of |g| whose parabola reaches zero
        without a sign change triggers a bounded minimum search. Values at
        roundoff scale are rejected as unresolved. Finite sampling cannot
        certify all zeros of an arbitrary callable; resolve its oscillations
        in the supplied mesh and provide dg when slope accuracy matters.
        """
        oversample = positive_count(oversample, "oversample")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        nodes = np.array(nodes, dtype=float)
        if (nodes.ndim != 1 or len(nodes) < 2 or not np.all(np.isfinite(nodes))
                or np.any(np.diff(nodes) <= 0)):
            raise ValueError("nodes must be strictly increasing with at least two entries")
        if period is not None:
            if not math.isfinite(period) or period <= 0 or not math.isfinite(nodes[0] + period):
                raise ValueError("period and chart endpoint must be positive/finite")
            if nodes[-1] >= nodes[0] + period:
                raise ValueError("periodic nodes must lie in [n_0, n_0 + period)")

        def G(x):
            return np.asarray(g(np.asarray(x, dtype=float)), dtype=float) * np.ones_like(x, dtype=float)

        ends = np.append(nodes, nodes[0] + period) if period is not None else nodes
        fine = np.concatenate([np.linspace(a, b, oversample, endpoint=False)
                               for a, b in zip(ends[:-1], ends[1:])] + [ends[-1:]])
        gf = G(fine)
        if not np.all(np.isfinite(gf)):
            raise ValueError("g must be finite on the sampling grid")
        scale = max(1.0, float(np.max(np.abs(gf))))

        zeros = list(fine[gf == 0])
        for i in np.flatnonzero(gf[:-1] * gf[1:] < 0):
            a, b, ga = fine[i], fine[i + 1], gf[i]
            for _ in range(200):
                m = 0.5 * (a + b)
                if m in (a, b):
                    break
                gm = float(G(np.array(m)))
                if gm == 0:
                    a = b = m
                    break
                if (gm < 0) == (ga < 0):
                    a, ga = m, gm
                else:
                    b = m
            zeros.append(0.5 * (a + b))
        zeros = np.unique(np.array(zeros, dtype=float))
        if period is not None:
            zeros = nodes[0] + np.remainder(zeros - nodes[0], period)
            zeros = np.unique(np.where(zeros >= nodes[0] + period, nodes[0], zeros))

        span = float(ends[-1] - ends[0])
        for z in zeros:
            if dg is not None:
                slope = float(np.asarray(dg(np.array(z)), dtype=float))
            else:
                h = 1e-6 * max(1.0, abs(z), span)
                slope = float(G(np.array(z + h)) - G(np.array(z - h))) / (2 * h)
            if not abs(slope) >= beta:
                raise ValueError(f"zero of g at x={z:.6g} is not transverse: |g'| = {abs(slope):.3g} < beta = {beta:.3g}")

        # Tangential zeros without a sign change: local minima of |g| on the
        # fine grid whose interpolating parabola dips to zero.
        absf = np.abs(gf)
        for i in range(1, len(fine) - 1):
            if absf[i] <= absf[i - 1] and absf[i] <= absf[i + 1] and gf[i - 1] * gf[i + 1] > 0:
                if any(abs(fine[i] - z) <= 2 * (fine[i + 1] - fine[i - 1]) for z in zeros):
                    continue
                h0, h1 = fine[i] - fine[i - 1], fine[i + 1] - fine[i]
                d0, d1 = (absf[i] - absf[i - 1]) / h0, (absf[i + 1] - absf[i]) / h1
                curv = 2 * (d1 - d0) / (h0 + h1)
                # The derivative at the middle sample uses unequal spacings.
                derivative = (h1 * d0 + h0 * d1) / (h0 + h1)
                vertex = absf[i] - derivative ** 2 / (2 * curv) if curv > 0 else absf[i]
                threshold = 64 * np.finfo(float).eps * scale
                if vertex <= threshold:
                    # A quadratic fit to a strictly positive quartic can dip
                    # below zero. Confirm against g, not against the fitted
                    # parabola. This bounded search is a numerical diagnostic,
                    # never a proof that an arbitrary callable has no zeros.
                    a, b = fine[i - 1], fine[i + 1]
                    ratio = (math.sqrt(5) - 1) / 2
                    for _ in range(64):
                        l, r = b - ratio * (b - a), a + ratio * (b - a)
                        if abs(float(G(l))) < abs(float(G(r))):
                            b = r
                        else:
                            a = l
                    if abs(float(G(0.5 * (a + b)))) <= threshold:
                        raise ValueError(f"g has a numerically unresolved or tangential zero near x={fine[i]:.6g}")

        # Merge zeros into the nodes; drop nodes closer to a zero than roundoff.
        tol = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(ends))))
        keep = [x for x in nodes if not np.any(np.abs(zeros - x) <= tol)]
        merged = np.unique(np.concatenate((np.array(keep), zeros)))
        # Two zeros with no node between them would make g_Δ vanish on a cell.
        is_zero = np.isin(merged, zeros)
        pairs = [(a, b) for a, b, za, zb in zip(merged[:-1], merged[1:], is_zero[:-1], is_zero[1:])
                 if za and zb]
        if period is not None and len(merged) > 1 and is_zero[-1] and is_zero[0]:
            pairs.append((merged[-1], merged[0] + period))
        mids = [0.5 * (a + b) for a, b in pairs]
        if period is not None:
            mids = [m - period if m >= nodes[0] + period else m for m in mids]
        merged = np.unique(np.concatenate((merged, np.array(mids, dtype=float))))
        values = G(merged)
        values[np.isin(merged, zeros)] = 0.0
        return cls(merged, values, period=period, extend=extend, beta=beta)

    # ---------------------------------------------------------------- queries
    @property
    def n_cells(self):
        return len(self.left)

    @property
    def origin(self):
        return float(self.nodes[0])

    @property
    def is_well_signed(self):
        return len(self.zeros) == 0

    def wrap(self, x):
        """Canonical chart coordinates on a circle; identity on the line."""
        x = np.asarray(x, dtype=float)
        if not self.periodic:
            return x
        o, L = self.nodes[0], self.period
        w = o + np.remainder(x - o, L)
        return np.where(w >= o + L, o, w)

    def __call__(self, x):
        """Evaluate g_Δ (vectorised)."""
        x = np.asarray(x, dtype=float)
        if self.periodic:
            out = np.interp(x, self.nodes, self.values, period=self.period)
        else:
            out = np.interp(x, self.nodes, self.values)
            if self.extend == "affine":
                out = np.where(x < self.nodes[0], self.values[0] + self.slope[0] * (x - self.nodes[0]), out)
                out = np.where(x > self.nodes[-1], self.values[-1] + self.slope[-1] * (x - self.nodes[-1]), out)
        return float(out) if out.ndim == 0 else out

    def cell_of(self, x, side="right"):
        """Cell index containing x. At a node, ``side`` picks the right or left cell."""
        x = self.wrap(x)
        if self.periodic:
            if side == "right":
                c = np.searchsorted(self.nodes, x, side="right") - 1
            else:
                c = np.searchsorted(self.nodes, x, side="left") - 1
                c = np.where(c < 0, self.n_cells - 1, c)
        else:
            c = np.searchsorted(self.nodes, x, side=side)
        return int(c) if np.ndim(c) == 0 else c

    def abs_g(self, c, x):
        """|g_Δ(x)| for x in cell c, from the nearer finite endpoint (scalar)."""
        lo, hi = self.left[c], self.right[c]
        a = self.alpha[c]
        if hi != math.inf and (lo == -math.inf or x - lo > hi - x):
            value = self.absg_right[c] - a * (hi - x)
        else:
            value = self.absg_left[c] + a * (x - lo)
        return value if value > 0.0 else 0.0

    def sigma(self, c, x):
        """σ = sqrt|g_Δ(x)| for x in cell c (scalar): the p = 2 coordinate.

        For h(u) = |u|^p use ``MultiplicativeFlux.sigma``, which is |g_Δ|^{1-1/p}.
        """
        return math.sqrt(self.abs_g(c, x))

    def sigma_array(self, x, cells):
        """Vectorised σ for coordinates x already assigned to cells."""
        return np.sqrt(self.abs_g_array(x, cells))

    def abs_g_array(self, x, cells):
        """Vectorised |g_Δ(x)| for coordinates x already assigned to cells."""
        x, cells = np.asarray(x, dtype=float), np.asarray(cells)
        lo, hi, a = self.left[cells], self.right[cells], self.alpha[cells]
        use_right = np.isfinite(hi) & (~np.isfinite(lo) | (x - lo > hi - x))
        with np.errstate(invalid="ignore"):
            from_right = self.absg_right[cells] - a * (hi - x)
            from_left = self.absg_left[cells] + a * (x - lo)
        value = np.where(use_right, from_right, from_left)
        return np.maximum(value, 0.0)

    def next_cell(self, c, direction):
        """Neighbouring cell in ``direction`` (+1 or -1), or None off the line."""
        c = c + (1 if direction > 0 else -1)
        if self.periodic:
            return c % self.n_cells
        return c if 0 <= c < self.n_cells else None

    def sign_intervals(self):
        """Maximal intervals of constant sign as (start, end, sign).

        On the line the ends may be ±inf. On a circle each arc starts at a zero
        and may end beyond n_0 + L (it then crosses the chart seam); a
        well-signed circle is one arc (n_0, n_0 + L).
        """
        if self.periodic:
            zs = list(self.zeros)
            if not zs:
                return [(self.origin, self.origin + self.period, float(self.sign[0]))]
            out = []
            for i, z in enumerate(zs):
                end = zs[i + 1] if i + 1 < len(zs) else zs[0] + self.period
                c = self.cell_of(z, side="right")
                out.append((float(z), float(end), float(self.sign[c])))
            return out
        edges = [-math.inf, *map(float, self.zeros), math.inf]
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            if math.isfinite(a) and math.isfinite(b):
                m = 0.5 * (a + b)
            elif math.isfinite(b):
                m = b - 1.0
            elif math.isfinite(a):
                m = a + 1.0
            else:
                m = 0.0
            out.append((a, b, float(self.sign[self.cell_of(m)])))
        return out

    def __repr__(self):
        kind = f"period={self.period}" if self.periodic else f"extend={self.extend!r}"
        return f"PiecewiseAffine({len(self.nodes)} nodes, {len(self.zeros)} zeros, {kind})"
