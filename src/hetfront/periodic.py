"""Heterogeneous front tracking on the circle S¹ = R/(L Z).

The setup follows ``fronttrack.periodic`` (scalar-front-tracking):
items live in cyclic order on the canonical chart [n_0, n_0 + L), and the
last item's right neighbour is the first one, one period further on. Here
the heterogeneity g_Δ is L-periodic. Each front stays inside one cell of the
chart, and the seam n_0 ≡ n_0 + L is a node. So a front crosses the seam
through an ordinary corner event, after which the list is rotated.

* Well-signed g: genuine cyclic adjacency; fronts circulate indefinitely and
  the total mass is conserved exactly.
* g with zeros: the circle splits into sign arcs separated by absorbing
  interfaces. Arcs evolve independently, and one may cross the chart seam.
"""

import math

import numpy as np

from .domains import Circle
from .fronts import Front
from .solver import _TrackerCore
from ._validation import integer_levels, positive_count


class PeriodicHeteroFrontTracker(_TrackerCore):
    """Evolve u_t + (g_Δ(x) h(u))_x = 0 on the circle of g_Δ's period (h = u² or |u|^p).

    N breakpoints require N levels: ``levels[j]`` occupies the positively
    oriented arc from ``positions[j]`` to ``positions[j+1]``, including the
    final arc through the seam. For a constant level, give no breakpoints and
    one level. ``interface_layer`` behaves as on the line.
    """

    cyclic = True

    def __init__(self, flux, positions, levels, *, interface_layer=0.0, enforce_interface=True,
                 position_tol=1e-12, max_events=1_000_000, record_history=True):
        self._init_core(flux, position_tol, max_events, record_history)
        self.enforce_interface = bool(enforce_interface)
        g = self.g
        if not g.periodic:
            raise TypeError("PeriodicHeteroFrontTracker needs a periodic heterogeneity")
        self.domain = Circle(length=g.period, origin=g.origin)
        L, o = g.period, g.origin
        if not 0 <= position_tol < L / 4:
            raise ValueError("position_tol must be in [0, period/4)")
        p = np.asarray(positions, dtype=float)
        levels = integer_levels(levels).tolist()
        if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(np.diff(p) <= 0):
            raise ValueError("positions must be finite and strictly increasing")
        if len(p) and (p[0] < o or p[-1] >= o + L):
            raise ValueError("positions must lie in the chart [n_0, n_0 + period)")
        if len(levels) != (len(p) if len(p) else 1):
            raise ValueError("need N positions and N levels, or none and one constant level")
        eps = float(interface_layer)
        zeros = np.asarray(g.zeros)
        if len(zeros) > 1:
            spacing = min(np.min(np.diff(zeros)), zeros[0] + L - zeros[-1])
        else:
            spacing = L
        if not math.isfinite(eps) or eps < 0 or 2 * eps >= spacing:
            raise ValueError("interface_layer must be >= 0 and less than half the zero spacing")
        self.interface_layer = eps
        self._constant_level = levels[0]
        self._unsorted = {}  # see _normalize

        def circular(a, b):
            d = abs(a - b) % L
            return min(d, L - d)

        def level_at(m):
            if eps > 0 and any(circular(m, z) < eps for z in zeros):
                return 0
            if not len(p):
                return levels[0]
            return levels[int(np.searchsorted(p, m, side="right")) - 1]

        layer = [float(self.domain.wrap(z + d)) for z in zeros for d in (-eps, eps)] if eps > 0 else []
        points = np.unique(np.concatenate((p, zeros, np.array(layer, dtype=float))))
        if not len(points):
            return
        ends = np.append(points[1:], points[0] + L)
        arcs = [level_at(float(self.domain.wrap(0.5 * (a + b)))) for a, b in zip(points, ends)]
        zero_set = set(zeros.tolist())
        items = []
        for j, x in enumerate(points):
            kL, kR = arcs[j - 1], arcs[j]
            if float(x) in zero_set:
                items.append(self._make_interface(x, kL, kR))
                self._record_interface(items[-1])
            elif kL != kR:
                items.extend(self._new_fronts(float(x), kL, kR, 0.0, g.cell_of(x)))
        self._link(items)
        if not items:
            self._constant_level = arcs[0]
        for iface in self.interfaces:
            self._settle(iface, initial=True)
        self._normalize()
        self._refresh_all()

    @classmethod
    def from_theta(cls, flux, positions, theta_values, **kwargs):
        """Quantise flux-level values θ to the nearest levels on δZ."""
        return cls(flux, positions, flux.level(theta_values), **kwargs)

    @classmethod
    def from_function(cls, flux, u0, cells=100, **kwargs):
        """Midpoint-sample u0 on ``cells`` equal arcs as θ = G(x, u0(x)); these are not cell averages."""
        g = flux.g
        cells = positive_count(cells, "cells")
        edges = g.origin + np.arange(cells) * g.period / cells
        mids = edges + g.period / (2 * cells)
        theta = flux.theta_of(mids, np.asarray(u0(mids), dtype=float) * np.ones_like(mids))
        return cls.from_theta(flux, edges, theta, **kwargs)

    # --------------------------------------------------------- cyclic order
    def _normalize(self, touched=None):
        """Rotate so canonical positions increase from the largest cyclic descent.

        The order starts (at the head) after the largest descent x(prev) - x
        between cyclic neighbours, the first one in order on ties. Between
        events no pair of neighbours changes order: a front stays inside its
        cell until a corner event, closing neighbours collide first, and
        interfaces do not move. So a positive descent can only appear at a
        pair an event touched, and it stays positive until an event touches
        it again. Normally the only one is the wrap pair (last item, head).
        Simultaneous seam crossings (coincident fronts that corner one at a
        time) leave others for a while; those are remembered in _unsorted.

        Hence the largest descent lies among the wrap pair, the pairs ending
        at or starting from a touched item, and the remembered pairs: found
        in O(len(touched)) instead of O(F). With ``touched=None`` (setup)
        every pair is compared.
        """
        if self._count < 2:
            self._unsorted = {}
            return False
        head = self._head
        if touched is None:
            items = self.items
            pos = self.positions()
            descents = np.roll(pos, 1) - pos  # descents[i] = x[i-1] - x[i]
            start = int(np.argmax(descents))
            new_head = items[start]
            self._unsorted = {id(items[i]): items[i] for i in np.flatnonzero(descents > 0) if i != start}
        else:
            t = self.time

            def x(item):
                return item.x(t) if isinstance(item, Front) else item.x

            # A pair is named by its second item; the head names the wrap pair.
            candidates = {id(head): head}
            for item in [*touched, *self._unsorted.values()]:
                if item is not None and self._alive(item):
                    candidates.setdefault(id(item), item)
                    candidates.setdefault(id(item.next), item.next)
            descent = {key: x(item.prev) - x(item) for key, item in candidates.items()}
            best = max(descent.values())
            if descent[id(head)] == best:
                new_head = head  # argmax's first index is the head itself
            else:
                winners = {key for key, d in descent.items() if d == best}
                new_head = head.next
                while id(new_head) not in winners:  # first in order: only on ties
                    new_head = new_head.next
            self._unsorted = {key: item for key, item in candidates.items()
                              if descent[key] > 0 and item is not new_head}
        if new_head is head:
            return False
        old_last = head.prev
        self._head = new_head
        self._items = None
        # Rotation keeps adjacency; only the wrap pair changes, so refresh the
        # former and the new last items (their pair events depend on wrapping).
        self._refresh_around([old_last, new_head.prev])
        return True

    # --------------------------------------------------------------- queries
    @property
    def constant_level(self):
        """The level of the circle when it carries no items."""
        return self._constant_level

    def sample_levels(self, x):
        """Integer levels at any finite coordinates (wrapped), right-continuous."""
        x = np.asarray(self.domain.wrap(x))
        if not self.items:
            out = np.full(x.shape, self._constant_level, dtype=int)
        else:
            pos = self.positions()
            first = pos[0]
            lifted = first + np.remainder(x - first, self.domain.length)
            index = np.searchsorted(pos, lifted, side="right") - 1
            out = np.array([item.kR for item in self.items], dtype=int)[index]
        return int(out) if out.ndim == 0 else out

    def integral(self):
        """Exact total mass ∫_{S¹} u dx at the current time."""
        o = self.domain.origin
        return self._integrate(o, o + self.domain.length)
