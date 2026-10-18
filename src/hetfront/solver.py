"""Event-driven heterogeneous front tracking for u_t + (g_Δ(x) h(u))_x = 0.

h(u) = u² by default, or |u|^p (p > 1) through ``MultiplicativeFlux(power=p)``.

Three kinds of event occur, all at times obtained in closed form:

* **collision** of adjacent fronts in one cell: the cluster is replaced by the
  generalised Riemann solution between its two exterior levels (HetFT
  Theorem 2.4: a single front when levels lie on δZ);
* **corner**: a front reaches a node where g_Δ ≠ 0. Nothing is emitted; the
  front only switches to the affine law of the next cell;
* **absorption**: a front reaches a zero of g_Δ and merges into the
  stationary interface there. The interface then holds the levels the
  Lax-type condition permits (θ ≥ 0 on its left, θ ≤ 0 on its right). A
  violation is resolved by a δ-fan leaving the zero and is counted under
  ``counts["emission"]``; with levels on δZ it never occurs after t = 0.

``_TrackerCore`` holds everything shared with the circle driver in
``periodic.py``; ``HeteroFrontTracker`` is the whole-line driver.
"""

from collections import namedtuple
from dataclasses import replace
import heapq
import itertools
import math

import numpy as np

from .flux import MultiplicativeFlux
from .fronts import Front, Interface
from .riemann import solve_riemann
from ._validation import integer_levels, positive_count

_ROUNDOFF_FACTOR = 16 * np.finfo(float).eps

Piece = namedtuple("Piece", "t0 t1 x0 s0 alpha kappa kL kR cell")
Piece.__doc__ = "One closed-form segment of a front's path (constant law on [t0, t1])."


class _TrackerCore:
    """Shared state and event handling for the line and circle drivers."""

    cyclic = False

    def _init_core(self, flux, position_tol, max_events, record_history):
        if not isinstance(flux, MultiplicativeFlux):
            raise TypeError("flux must be a MultiplicativeFlux")
        if not math.isfinite(position_tol) or position_tol < 0:
            raise ValueError("position_tol must be finite and nonnegative")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.flux, self.g, self.law = flux, flux.g, flux.law
        self.position_tol, self.max_events = position_tol, max_events
        self.record_history = record_history
        self.time = 0.0
        self.event_count = 0
        self.counts = {"collision": 0, "corner": 0, "absorption": 0,
                       "emission": 0, "initial_emission": 0, "multi_front_collision": 0}
        self.items = []        # Fronts and Interfaces in spatial (cyclic) order
        self.pieces = []       # retired Piece records, when record_history
        self.events = []       # (t, x, kind), when record_history
        self.interface_log = []  # (t, x, kL, kR) after every interface change
        # Event queue with lazy invalidation: entries carry the item's version.
        self._heap, self._seq = [], itertools.count()

    # ------------------------------------------------------------- geometry
    def _tol(self, x):
        scale = max(1.0, abs(x), self.g.period or 0.0)
        return self.position_tol + _ROUNDOFF_FACTOR * scale

    def _place(self, x, kappa, cell, s_hint=None):
        """Choose the cell a new front at x enters, snapping to nodes within tolerance."""
        g, flux = self.g, self.flux
        tol = self._tol(x)
        lo, hi = g.left[cell], g.right[cell]
        if hi != math.inf and abs(hi - x) <= tol:
            if kappa > 0 and not g.zero_right[cell]:
                nxt = g.next_cell(cell, 1)
                if nxt is not None:
                    return nxt, float(g.left[nxt]), float(flux.sigma_left[nxt])
            return cell, float(hi), float(flux.sigma_right[cell])
        if lo != -math.inf and abs(x - lo) <= tol:
            if kappa < 0 and not g.zero_left[cell]:
                prv = g.next_cell(cell, -1)
                if prv is not None:
                    return prv, float(g.right[prv]), float(flux.sigma_right[prv])
            return cell, float(lo), float(flux.sigma_left[cell])
        s = flux.sigma(cell, x) if s_hint is None else s_hint
        return cell, float(x), float(s)

    def _new_fronts(self, x, kL, kR, t, cell, s_hint=None):
        out = []
        for a, b in solve_riemann(kL, kR):
            kappa = self.flux.kappa(a, b)
            c, xx, ss = self._place(x, kappa, cell, s_hint)
            out.append(Front(a, b, kappa, c, t, xx, ss, float(self.g.alpha[c]), law=self.law))
        return out

    def _make_interface(self, z, kL, kR):
        g = self.g
        lc, rc = g.cell_of(z, side="left"), g.cell_of(z, side="right")
        return Interface(float(z), int(kL), int(kR), float(g.sign[lc]), float(g.sign[rc]),
                         int(lc), int(rc), last_time=self.time)

    # --------------------------------------------------------- event times
    def _own_event(self, f):
        if f.kappa == 0.0:
            return math.inf, ""
        g, flux, c = self.g, self.flux, f.cell
        if f.kappa > 0:
            target, s_target, zero = g.right[c], flux.sigma_right[c], g.zero_right[c]
        else:
            target, s_target, zero = g.left[c], flux.sigma_left[c], g.zero_left[c]
        if not math.isfinite(target):
            return math.inf, ""
        tau = self.law.travel_time(f.x0, f.s0, float(target), float(s_target), f.kappa)
        return f.t0 + tau, ("absorption" if zero else "corner")

    def _pair_event(self, i):
        items, n = self.items, len(self.items)
        if n < 2 or (not self.cyclic and i >= n - 1):
            return math.inf
        a, b = items[i], items[(i + 1) % n]
        if not (isinstance(a, Front) and isinstance(b, Front)):
            return math.inf
        wrap = self.cyclic and i == n - 1
        t = self.time
        xa, xb = a.x(t), b.x(t) + (self.g.period if wrap else 0.0)
        closing = a.kappa > b.kappa
        if closing and xb - xa <= self._tol(xa):
            return t
        if a.cell == b.cell and not wrap:
            return t + self.law.meeting_time(xa, a.sigma(t), a.kappa, xb, b.sigma(t), b.kappa)
        return math.inf

    def _refresh(self, indices):
        """Recompute cached own events and pair events for the given item indices."""
        n = len(self.items)
        if n == 0:
            return
        seen = set()
        for i in indices:
            if self.cyclic:
                i %= n
            elif not 0 <= i < n:
                continue
            if i in seen:
                continue
            seen.add(i)
            item = self.items[i]
            item.version += 1
            if isinstance(item, Front):
                item.t_own, item.own_kind = self._own_event(item)
                if item.t_own < math.inf:
                    heapq.heappush(self._heap, (item.t_own, 1, next(self._seq), item,
                                                item.version, item.own_kind))
            item.t_pair = self._pair_event(i)
            if item.t_pair < math.inf:
                # Priority 0: at equal times a collision precedes a corner/absorption.
                heapq.heappush(self._heap, (item.t_pair, 0, next(self._seq), item,
                                            item.version, "collision"))

    def _refresh_all(self):
        self._heap.clear()
        self._refresh(range(len(self.items)))

    def _refresh_around(self, objects):
        """Refresh each live object and its left neighbour (whose pair it closes)."""
        items = self.items
        indices = []
        for obj in objects:
            if obj is None:
                continue
            try:
                i = items.index(obj)
            except ValueError:
                continue
            indices.extend((i - 1, i))
        self._refresh(indices)

    def _drop(self, item):
        item.version += 1  # removes its queued events

    def _next_event(self):
        """Earliest valid queued event as (time, kind, item), or (inf, None, None)."""
        heap = self._heap
        if len(heap) > 8 * len(self.items) + 4096:
            self._refresh_all()  # compact stale entries
        while heap:
            t, _, _, item, version, kind = heap[0]
            if item.version == version:
                return t, kind, item
            heapq.heappop(heap)
        return math.inf, None, None

    # ------------------------------------------------------------ event code
    def _retire(self, f, t):
        if self.record_history:
            self.pieces.append(Piece(f.t0, t, f.x0, f.s0, f.alpha, f.kappa, f.kL, f.kR, f.cell))

    def _log(self, x, kind):
        if self.record_history:
            self.events.append((self.time, float(x), kind))

    def _corner(self, i):
        f, g, t = self.items[i], self.g, self.time
        self._retire(f, t)
        d = f.direction
        nxt = g.next_cell(f.cell, d)
        if d > 0:
            x, s = float(g.left[nxt]), float(self.flux.sigma_left[nxt])
        else:
            x, s = float(g.right[nxt]), float(self.flux.sigma_right[nxt])
        seam = self.cyclic and ((d > 0 and nxt == 0) or (d < 0 and nxt == g.n_cells - 1))
        f.cell, f.t0, f.x0, f.s0, f.alpha = nxt, t, x, s, float(g.alpha[nxt])
        self.counts["corner"] += 1
        self._log(x, "corner")
        if seam:
            self._normalize()
        self._refresh_around([f])

    def _absorb(self, i):
        items, t = self.items, self.time
        f = items[i]
        n = len(items)
        j = (i + f.direction) % n if self.cyclic else i + f.direction
        iface = items[j]
        if not isinstance(iface, Interface):
            raise RuntimeError("front reached a zero of g without meeting its interface")
        self._retire(f, t)
        iface.accrue(t, self.flux.delta)
        if f.direction > 0:
            if iface.kL != f.kR:
                raise RuntimeError("level chain broken at interface")
            iface.kL = f.kL
        else:
            if iface.kR != f.kL:
                raise RuntimeError("level chain broken at interface")
            iface.kR = f.kR
        del items[i]
        self._drop(f)
        left_neighbour = items[(items.index(iface) - 1) % len(items)] if f.direction > 0 else iface
        self.counts["absorption"] += 1
        self._log(iface.x, "absorption")
        self._record_interface(iface)
        before = set(map(id, items))
        self._settle(items.index(iface))
        emitted = [item for item in items if id(item) not in before]
        if self.cyclic:
            self._normalize()
        self._refresh_around([left_neighbour, iface, *emitted])

    def _settle(self, j, initial=False):
        """Enforce the interface condition at items[j] by emitting δ-fans; returns the count."""
        if not self.enforce_interface:
            return 0
        iface, g, t = self.items[j], self.g, self.time
        emitted = 0
        key = "initial_emission" if initial else "emission"
        if iface.kR > 0:
            c = iface.right_cell
            fronts = [Front(a, b, self.flux.kappa(a, b), c, t, float(g.left[c]), 0.0, float(g.alpha[c]),
                            law=self.law)
                      for a, b in solve_riemann(0, iface.kR)]
            iface.accrue(t, self.flux.delta)
            iface.kR = 0
            self.items[j + 1:j + 1] = fronts
            emitted += len(fronts)
            self.counts[key] += 1
        if iface.kL < 0:
            c = iface.left_cell
            fronts = [Front(a, b, self.flux.kappa(a, b), c, t, float(g.right[c]), 0.0, float(g.alpha[c]),
                            law=self.law)
                      for a, b in solve_riemann(iface.kL, 0)]
            iface.accrue(t, self.flux.delta)
            iface.kL = 0
            self.items[j:j] = fronts
            emitted += len(fronts)
            self.counts[key] += 1
        if emitted:
            self._log(iface.x, "emission")
            self._record_interface(iface)
        return emitted

    def _record_interface(self, iface):
        self.interface_log.append((self.time, iface.x, iface.kL, iface.kR))

    def _lifted(self, k, t):
        """Position of item k at time t; on a circle, k may be any integer.

        The cyclic list is sorted on the chart, so item k lies on the lift
        x(items[k mod n]) + (k div n)·L. Zero-width arcs therefore stay zero
        width, and the wrap gap keeps its full length even when every front is
        at one point.
        """
        n = len(self.items)
        item = self.items[k % n] if self.cyclic else self.items[k]
        x = item.x(t) if isinstance(item, Front) else item.x
        return x + (k // n) * self.g.period if self.cyclic else x

    def _collide(self, i):
        items, t, g = self.items, self.time, self.g
        n = len(items)
        a, b = items[i], items[(i + 1) % n]
        x = 0.5 * (self._lifted(i, t) + self._lifted(i + 1, t))
        s_event = 0.5 * (a.sigma(t) + b.sigma(t))
        tol = self._tol(x)
        first, last = i, i + 1
        while last - first + 1 < n:
            k = first - 1
            if (not self.cyclic and k < 0) or not isinstance(items[k % n], Front):
                break
            if abs(self._lifted(k, t) - x) > tol:
                break
            first = k
        while last - first + 1 < n:
            k = last + 1
            if (not self.cyclic and k >= n) or not isinstance(items[k % n], Front):
                break
            if abs(self._lifted(k, t) - x) > tol:
                break
            last = k
        members = [items[k % n] for k in range(first, last + 1)]
        kL, kR = members[0].kL, members[-1].kR
        left_neighbour = None
        if len(members) < n and (self.cyclic or first > 0):
            left_neighbour = items[(first - 1) % n]
        for m in members:
            self._retire(m, t)
            self._drop(m)
        if self.cyclic:
            x = float(g.wrap(x))
        hint = g.cell_of(x)
        outgoing = self._new_fronts(x, kL, kR, t, hint, s_event)
        self.counts["collision"] += 1
        if len(outgoing) > 1:
            self.counts["multi_front_collision"] += 1
        self._log(x, "collision")
        if self.cyclic and (first < 0 or last >= n):
            start = first % n
            items[:] = items[start:] + items[:start]
            first, last = 0, len(members) - 1
        items[first:last + 1] = outgoing
        if not items:
            self._constant_level = kL
        if self.cyclic:
            self._normalize()
        self._refresh_around([left_neighbour, *outgoing])

    def _normalize(self):
        """Keep a cyclic item list sorted on the chart (overridden on the circle)."""
        return False

    # ------------------------------------------------------------- driving
    def run(self, final_time):
        """Advance to absolute ``final_time`` (events at exactly that time are resolved).

        Returns self. Calling again continues; going backward is rejected. An
        event-limit error leaves the last completed state intact.
        """
        if not math.isfinite(final_time) or final_time < self.time:
            raise ValueError("final_time must be finite and at least the current time")
        while True:
            t_event, kind, item = self._next_event()
            if item is None or t_event > final_time:
                self.time = float(final_time)
                return self
            if self.event_count >= self.max_events:
                raise RuntimeError("Event limit reached; inspect data or raise max_events")
            self.time = max(self.time, t_event)
            heapq.heappop(self._heap)
            i = self.items.index(item)
            if kind == "collision":
                self._collide(i)
            elif kind == "corner":
                self._corner(i)
            else:
                self._absorb(i)
            self.event_count += 1

    # ------------------------------------------------------------- queries
    @property
    def fronts(self):
        """Live moving fronts (not interfaces), in order. Do not modify."""
        return [item for item in self.items if isinstance(item, Front)]

    @property
    def interfaces(self):
        return [item for item in self.items if isinstance(item, Interface)]

    def positions(self, t=None):
        """Current positions of all items (fronts and interfaces), in list order."""
        t = self.time if t is None else t
        return np.array([item.x(t) if isinstance(item, Front) else item.x for item in self.items])

    def front_count(self):
        return sum(isinstance(item, Front) for item in self.items)

    def snapshot(self):
        """Independent copies of the current items."""
        return [replace(item) for item in self.items]

    def total_variation(self, include_interfaces=True):
        """Σ |θ_R - θ_L| over fronts (and interfaces): the variation of θ."""
        return self.flux.delta * math.fsum(
            abs(item.kR - item.kL) for item in self.items
            if include_interfaces or isinstance(item, Front))

    def mass_absorbed(self):
        """Mass that has disappeared into zeros of g up to the current time."""
        return math.fsum(item.absorbed_until(self.time, self.flux.delta) for item in self.interfaces)

    def interface_admissible(self):
        """The stronger selection rule: theta_left >= 0 >= theta_right."""
        return all(item.is_admissible() for item in self.interfaces)

    def interface_entropy_admissible(self):
        """Check the zero-interface entropy inequality theta_right <= theta_left.

        At a transverse zero the limiting Kruzhkov entropy flux is theta on
        either side, for every finite comparison state. Its distributional
        jump must be nonpositive. Disabling selection does not ensure this.
        Moving rarefaction fronts still have the usual finite-delta defect.
        """
        return all(item.kR <= item.kL for item in self.interfaces)

    def check_levels(self):
        """Raise if adjacent items disagree on the level between them."""
        items, n = self.items, len(self.items)
        pairs = range(n) if self.cyclic else range(n - 1)
        for i in pairs:
            a, b = items[i], items[(i + 1) % n]
            if a.kR != b.kL:
                raise AssertionError(f"level chain broken between items {i} and {(i + 1) % n}")

    def pieces_until(self, t=None):
        """All closed-form path pieces: retired ones plus live fronts up to time t."""
        t = self.time if t is None else t
        live = [Piece(f.t0, t, f.x0, f.s0, f.alpha, f.kappa, f.kL, f.kR, f.cell)
                for f in self.fronts]
        return list(self.pieces) + live

    def sample_theta(self, x):
        """Flux-level values θ(x) at the current time (right-continuous)."""
        return self.flux.delta * self.sample_levels(x)

    def sample_u(self, x):
        """u(x) at the current time; ±inf exactly at a zero of g with a nonzero level."""
        x = np.asarray(x, dtype=float)
        return self.flux.u(self.sample_levels(x), x)

    def _breakpoints(self, a, b, *, nodes=False):
        """Interior jumps (and optionally coefficient nodes) in a lifted window.

        Sampling wraps coordinates; integration cannot: every period must be
        split separately, otherwise a cell law is extrapolated across its seam.
        """
        points = self.positions()
        if nodes:
            points = np.concatenate((points, self.g.nodes))
        if self.cyclic:
            # Include the seam even for a constant state with no moving fronts.
            points = np.append(points, self.g.origin)
            first = math.floor((a - self.g.origin) / self.g.period)
            last = math.floor((b - self.g.origin) / self.g.period)
            points = np.concatenate([points + k * self.g.period
                                     for k in range(first, last + 1)])
        return points[(points > a) & (points < b)]

    def _pieces(self, xs_extra):
        """Breakpoints (fronts, nodes, extras) and exact ∫u over each piece between them."""
        g = self.g
        a, b = float(np.min(xs_extra)), float(np.max(xs_extra))
        inner = self._breakpoints(a, b, nodes=True)
        xs = np.unique(np.concatenate((np.asarray(xs_extra, dtype=float), inner)))
        x1, x2 = xs[:-1], xs[1:]
        if len(x1) == 0:
            return xs, np.zeros(0)
        mids = 0.5 * (x1 + x2)
        levels = self.sample_levels(mids)
        cells = g.cell_of(mids)
        s = self.flux.s(levels)
        # Translate both endpoints together, keeping the right endpoint on the
        # closing side of a seam rather than wrapping it into the first cell.
        shift = (np.floor((mids - g.origin) / g.period) * g.period
                 if self.cyclic else 0.0)
        s1 = self.flux.sigma_array(x1 - shift, cells)
        s2 = self.flux.sigma_array(x2 - shift, cells)
        # ∫ |g|^{-1/p} dx over [x1, x2] = (x2 - x1)/M(σ1, σ2); for p = 2, 2(x2 - x1)/(σ1 + σ2).
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.law.quadratic:
                parts = np.where(s == 0, 0.0, g.sign[cells] * s * 2.0 * (x2 - x1) / (s1 + s2))
            else:
                parts = np.where(s == 0, 0.0, g.sign[cells] * s * (x2 - x1) / self.law.mean_array(s1, s2))
        return xs, parts

    def _integrate(self, a, b):
        """Exact ∫_a^b u dx at the current time, splitting at fronts and nodes."""
        _, parts = self._pieces(np.array([a, b], dtype=float))
        return math.fsum(parts.tolist())

    def primitive(self, xs):
        """Exact running integral ∫_{xs[0]}^{x} u dx at each x in the sorted array xs."""
        xs = np.asarray(xs, dtype=float)
        if xs.ndim != 1 or len(xs) < 1 or not np.all(np.isfinite(xs)) or np.any(np.diff(xs) < 0):
            raise ValueError("xs must be a sorted 1-D array")
        breaks, parts = self._pieces(xs)
        running = np.concatenate(([0.0], np.cumsum(parts)))
        return running[np.searchsorted(breaks, xs)]

    def u_l1_distance(self, other, a, b):
        """Exact physical L1 distance for trackers sharing the same g object.

        Split at both sets of fronts and g's nodes. On each resulting interval
        u-v is a constant times |g|^{-1/p}, so it has one sign: the absolute
        value of its integral equals its L1 integral, even beside a zero.
        """
        if self.g is not other.g:
            raise ValueError("exact u distance requires the same heterogeneity object")
        if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
            raise ValueError("require finite a < b")
        breaks = np.unique(np.r_[a, b, self._breakpoints(a, b, nodes=True),
                                  other._breakpoints(a, b)])
        _, ours = self._pieces(breaks)
        _, theirs = other._pieces(breaks)
        return math.fsum(np.abs(ours - theirs).tolist())

    def theta_l1_distance(self, other, a, b):
        """Exact ∫_a^b |θ - θ_other| dx at the current times of both trackers."""
        if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
            raise ValueError("require finite a < b")
        inner = np.concatenate((self._breakpoints(a, b), other._breakpoints(a, b)))
        xs = np.unique(np.concatenate(([a, b], inner)))
        mids = 0.5 * (xs[:-1] + xs[1:])
        diff = np.abs(self.flux.delta * self.sample_levels(mids) - other.flux.delta * other.sample_levels(mids))
        return math.fsum((diff * np.diff(xs)).tolist())


class HeteroFrontTracker(_TrackerCore):
    """Front tracking for u_t + (g_Δ(x) h(u))_x = 0 on the whole line, h = u² or |u|^p.

    Supply N breakpoints and N+1 integer levels (θ = level·δ); the outer
    levels extend to ±∞ as stationary profiles. Breakpoints at zeros of g_Δ
    become interfaces. ``interface_layer=ε > 0`` sets θ = 0 on (z-ε, z+ε)
    around each zero, as in the manuscript; with ε = 0 an inadmissible level
    next to a zero emits a δ-fan from the zero instead. Inadmissible means
    θ < 0 just left of the zero, or θ > 0 just right of it.

    ``enforce_interface=False`` keeps inadmissible levels at zeros (no
    emission). Suitable data reproduce non-selected solutions, but arbitrary
    interface traces need not satisfy the entropy inequality. Check
    ``interface_entropy_admissible()``; this requires theta_left >= theta_right.
    """

    def __init__(self, flux, positions, levels, *, interface_layer=0.0, enforce_interface=True,
                 position_tol=1e-12, max_events=1_000_000, record_history=True):
        self._init_core(flux, position_tol, max_events, record_history)
        self.enforce_interface = bool(enforce_interface)
        if flux.g.periodic:
            raise TypeError("use PeriodicHeteroFrontTracker for a periodic heterogeneity")
        p = np.asarray(positions, dtype=float)
        levels = integer_levels(levels).tolist()
        if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(np.diff(p) <= 0):
            raise ValueError("positions must be finite and strictly increasing")
        if len(levels) != len(p) + 1:
            raise ValueError("need N positions and N+1 levels")
        eps = float(interface_layer)
        zeros = np.asarray(self.g.zeros)
        if not math.isfinite(eps) or eps < 0 or (eps > 0 and len(zeros) > 1 and 2 * eps >= np.min(np.diff(zeros))):
            raise ValueError("interface_layer must be >= 0 and less than half the zero spacing")
        self.interface_layer = eps
        self.left_level = levels[0]

        def level_at(m):
            if eps > 0 and np.any(np.abs(m - zeros) < eps):
                return 0
            return levels[int(np.searchsorted(p, m, side="right"))]

        layer = [z + d for z in zeros for d in (-eps, eps)] if eps > 0 else []
        points = np.unique(np.concatenate((p, zeros, np.array(layer, dtype=float))))
        between = [level_at(0.5 * (u + v)) for u, v in zip(points[:-1], points[1:])]
        interval_levels = [levels[0], *between, levels[-1]]
        zero_set = set(zeros.tolist())
        for j, x in enumerate(points):
            kL, kR = interval_levels[j], interval_levels[j + 1]
            if float(x) in zero_set:
                self.items.append(self._make_interface(x, kL, kR))
                self._record_interface(self.items[-1])
            elif kL != kR:
                self.items.extend(self._new_fronts(float(x), kL, kR, 0.0, self.g.cell_of(x)))
        for iface in self.interfaces:
            self._settle(self.items.index(iface), initial=True)
        self._refresh_all()

    @classmethod
    def from_theta(cls, flux, positions, theta_values, **kwargs):
        """Quantise flux-level values θ to the nearest levels on δZ."""
        return cls(flux, positions, flux.level(theta_values), **kwargs)

    @classmethod
    def from_function(cls, flux, u0, window, cells=100, **kwargs):
        """Midpoint-sample u0 on ``cells`` equal cells of ``window`` as θ = G(x, u0(x)).

        Outside the window the data continue as the stationary profiles of the
        levels at the window edges. The cell edges are the breakpoints.
        """
        a, b = map(float, window)
        if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
            raise ValueError("window must have finite endpoints a < b")
        cells = positive_count(cells, "cells")
        edges = np.linspace(a, b, cells + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        sample = np.concatenate(([a], mids, [b]))
        theta = flux.theta_of(sample, np.asarray(u0(sample), dtype=float) * np.ones_like(sample))
        return cls.from_theta(flux, edges, theta, **kwargs)

    def sample_levels(self, x):
        """Integer levels at x at the current time (right-continuous)."""
        x = np.asarray(x, dtype=float)
        pos = self.positions()
        table = np.array([self.left_level] + [item.kR for item in self.items], dtype=int)
        out = table[np.searchsorted(pos, x, side="right")]
        return int(out) if out.ndim == 0 else out

    def integral(self, a, b):
        """Exact ∫_a^b u(x, t) dx at the current time (u is integrable at zeros)."""
        if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
            raise ValueError("require finite a < b")
        return self._integrate(a, b)

    def flux_through(self, x, times):
        """Exact ∫_{times[0]}^{t} f(x, u(x, s)) ds at each t in ``times``.

        The level at a fixed x is left_level plus the jumps kR - kL of every
        item to its left, so it changes only when a front crosses x. Each
        recorded piece is monotone in its cell, and its crossing time is the
        closed-form ``travel_time`` to x. Interfaces left of x contribute their
        logged jumps. The result never uses ∫u, so a mass budget built from it
        is a genuine check, not an identity. Requires ``record_history``.
        """
        if not self.record_history:
            raise ValueError("flux_through needs record_history=True")
        x = float(x)
        times = np.asarray(times, dtype=float)
        if (not math.isfinite(x) or times.ndim != 1 or len(times) < 1 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) < 0) or times[0] < 0 or times[-1] > self.time):
            raise ValueError("x must be finite and times sorted within [0, current time]")
        g, flux, law = self.g, self.flux, self.law
        if g(x) == 0:
            raise ValueError("x is a zero of g, where the flux has two one-sided values")
        t_lo, t_hi = float(times[0]), float(times[-1])
        event_t, event_dk = [], []

        def active(start, end, jump):
            """The jump counts towards the level at x on [start, end)."""
            start = max(start, t_lo)
            if jump == 0 or start >= end or start >= t_hi:
                return
            event_t.append(start)
            event_dk.append(jump)
            if end < t_hi:
                event_t.append(end)
                event_dk.append(-jump)

        for p in self.pieces_until():
            c, jump = p.cell, p.kR - p.kL
            if p.kappa == 0.0:
                if p.x0 < x:
                    active(p.t0, p.t1, jump)
            elif p.kappa > 0:
                # Left of x until it reaches x, if x lies in its cell.
                if p.x0 < x:
                    end = p.t1
                    if x <= g.right[c]:
                        end = min(end, p.t0 + law.travel_time(p.x0, p.s0, x, flux.sigma(c, x), p.kappa))
                    active(p.t0, end, jump)
            elif p.x0 <= x:
                active(p.t0, p.t1, jump)
            elif x >= g.left[c]:
                # Right of x until it reaches x; travel_time is inf if it never does.
                start = p.t0 + law.travel_time(p.x0, p.s0, x, flux.sigma(c, x), p.kappa)
                active(start, p.t1, jump)
        logs = {}
        for t, z, kL, kR in self.interface_log:
            if z < x:
                logs.setdefault(z, []).append((t, kR - kL))
        for entries in logs.values():
            for (t, jump), (t_next, _) in zip(entries, entries[1:] + [(math.inf, 0)]):
                active(t, t_next, jump)

        order = np.argsort(event_t, kind="stable")
        event_t = np.asarray(event_t, dtype=float)[order]
        level_change = np.concatenate(([0], np.cumsum(np.asarray(event_dk, dtype=np.int64)[order])))
        grid = np.unique(np.concatenate((event_t, times)))
        levels = self.left_level + level_change[np.searchsorted(event_t, grid, side="right")]
        rates = flux.flux_value(levels, np.full(len(grid), x))
        running = np.concatenate(([0.0], np.cumsum(rates[:-1] * np.diff(grid))))
        return running[np.searchsorted(grid, times)]
