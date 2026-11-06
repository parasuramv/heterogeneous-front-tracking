"""Convex nonlinearities h for the multiplicative flux g(x) h(u).

A ``Nonlinearity`` is an even function h(u) = h(|u|) with

* h(0) = 0 and h > 0 elsewhere, strictly convex on [0, ∞);
* power-type growth: h(r) ≳ r^k as r → ∞ for some ``tail_exponent`` k > 1.

Power-type growth is slightly stronger than the Osgood condition
∫^∞ dr/h < ∞; for example u log²u is Osgood but not power-type. It is what
makes stationary profiles integrable at transverse zeros of g, fronts reach
those zeros in finite time, and the characteristic equation ṗ = -g'(q) h(p)
blow up in finite time, all at rates comparable with |x - z|^{-1/k}.
Every closed form in ``trajectory.GeneralLaw`` is built from three
functions on [0, ∞):

    h,     H = h⁻¹ (``inverse``),     J(u) = ∫_u^∞ dr / h(r) (``osgood``),

together with a cancellation-free divided difference ``dd(a, b)`` of h and
the mean of 1/h over an interval (``inverse_mean``). All scalar methods take
and return Python floats for u, y ≥ 0; the sign of u is handled by the caller.

**Factored Osgood primitive.** J has opposite singular behaviour at its two
ends: it diverges like u^{1-k_low} as u → 0 and decays like u^{1-k_top} as
u → ∞. Both rates are carried exactly by writing

    J(u) = (u / h(u)) · R(u),     R = h J / u = Q/u - 1,

where Q(u) = u + h(u)J(u) is the profile primitive. For |u|^p, R is the
constant 1/(p-1). For a polynomial h, R is bounded and analytic in s = log u,
running from 1/(k_low - 1) to 1/(k_top - 1). It is tabulated once per h as
piecewise Chebyshev series in s on a window [u_lo, u_hi]; every node is
checked against quadrature when the table is built. ``AbsPolynomial`` replaces J
outside the window by its exact asymptotic series: at infinity a series in
powers u^{-(k_top - k_i)}, and at zero the local series plus the finite-part
constant, with log terms where exponents resonate. ``Custom`` uses exp-sinh
quadrature outside the window. It is exact but slower, and needs only the
tail exponent.

``Power(p)`` is |u|^p with exact formulas. ``AbsPolynomial`` is
Σ c_k |u|^{k} with c_k ≥ 0 and real k > 1. ``Custom`` wraps user callables.
``Asymmetric(plus, minus)`` is h₊(u) for u ≥ 0 and h₋(-u) for u < 0, from two
even branches. ``GeneralLaw`` picks the branch of a level θ on a cell where g
has sign η from sgn u = η sgn θ.
"""

import math

import numpy as np

from .trajectory import divided_power

# ----------------------------------------------------------------- quadrature
_GAUSS = {}


def gauss_rule(n):
    """Gauss–Legendre nodes and weights on [0, 1] (cached), as Python lists."""
    rule = _GAUSS.get(n)
    if rule is None:
        x, w = np.polynomial.legendre.leggauss(n)
        rule = _GAUSS[n] = ((0.5 * (x + 1.0)).tolist(), (0.5 * w).tolist())
    return rule


def gauss_points(lo, hi):
    """Number of Gauss–Legendre points for full precision on [lo, hi] with 0 < lo ≤ hi ≤ 2 lo.

    The integrands used here (states and slownesses as functions of γ, or
    1/h(r)) are analytic except at 0. With r = (hi - lo)/(hi + lo), the
    Bernstein ellipse through 0 has ρ = (1 + sqrt(1 - r²))/r, and the error
    decays like ρ^{-2n}. Two extra points cover the constant.
    """
    r = (hi - lo) / (hi + lo)
    if r <= 0.0:
        return 1
    rho = (1.0 + math.sqrt(max(0.0, 1.0 - r * r))) / r
    return min(24, 2 + math.ceil(41.5 / (2.0 * math.log(rho))))


def gauss_mean(f, a, b):
    """Mean of a smooth scalar f over [a, b] (0 < a, b ≤ 2a) by adaptive Gauss–Legendre."""
    if a == b:
        return f(a)
    lo, hi = (a, b) if a < b else (b, a)
    xs, ws = gauss_rule(gauss_points(lo, hi))
    width = hi - lo
    return math.fsum(w * f(lo + width * x) for x, w in zip(xs, ws))


# ------------------------------------------------------- Chebyshev table of R
class _RatioTable:
    """Piecewise Chebyshev interpolant of R(s), s = log u, on uniform pieces."""

    __slots__ = ("s_lo", "s_hi", "width", "coefficients")

    def __init__(self, ratio, u_lo, u_hi, degree=24, tol=4e-15):
        s_lo, s_hi = math.log(u_lo), math.log(u_hi)
        if not s_hi > s_lo:
            s_hi = s_lo + 1e-3
        pieces = max(1, math.ceil(s_hi - s_lo))
        nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))  # Chebyshev points
        check = np.cos(np.pi * np.arange(1, degree + 1) / (degree + 1))        # interleaved points
        for _ in range(6):
            width = (s_hi - s_lo) / pieces
            coefficients, ok = [], True
            for j in range(pieces):
                mid = s_lo + (j + 0.5) * width
                values = np.array([ratio(math.exp(mid + 0.5 * width * x)) for x in nodes])
                coef = np.polynomial.chebyshev.chebfit(nodes, values, degree)
                test = np.array([ratio(math.exp(mid + 0.5 * width * x)) for x in check])
                error = np.max(np.abs(np.polynomial.chebyshev.chebval(check, coef) - test) / np.abs(test))
                if not error <= tol:
                    ok = False
                    break
                coefficients.append(coef.tolist())
            if ok:
                break
            pieces *= 2
        else:
            raise ArithmeticError("R = hJ/u could not be tabulated to full precision; is h smooth?")
        self.s_lo, self.s_hi, self.width, self.coefficients = s_lo, s_hi, width, coefficients

    def __call__(self, u):
        s = math.log(u)
        j = min(int((s - self.s_lo) / self.width), len(self.coefficients) - 1)
        x = 2.0 * (s - self.s_lo - j * self.width) / self.width - 1.0
        c = self.coefficients[j]
        b1 = b2 = 0.0  # Clenshaw
        x2 = 2.0 * x
        for ck in reversed(c[1:]):
            b1, b2 = ck + x2 * b1 - b2, b1
        return c[0] + x * b1 - b2


# ----------------------------------------------------------------- base class
class Nonlinearity:
    """Base class: generic numerics from ``h`` and ``dh`` alone.

    Subclasses provide ``h`` and ``dh`` (vectorised on arrays of u ≥ 0, and
    fast on Python floats) and ``tail_exponent``.
    """

    tail_exponent = None
    _table_window = (1e-6, 1e6)
    symmetric = True

    def branch(self, sign):
        """The even nonlinearity used where sgn u = sign; an even h is its own branch."""
        return self

    # ------------------------------------------------------------ required
    def h(self, u):
        raise NotImplementedError

    def dh(self, u):
        raise NotImplementedError

    # ------------------------------------------------------------ inverse
    def __call__(self, u):
        return self.h(np.abs(u))

    def inverse(self, y):
        """H(y) = h⁻¹(y) on [0, ∞), by Newton from above (monotone for convex h)."""
        y = float(y)
        if y <= 0.0:
            return 0.0
        if math.isinf(y):
            return math.inf
        return self._newton_from_above(y, self._inverse_upper_bound(y))

    def inverse_array(self, y):
        y = np.asarray(y, dtype=float)
        return np.array([self.inverse(v) for v in y.ravel()]).reshape(y.shape)

    def _inverse_upper_bound(self, y):
        hi = 1.0
        while self.h(hi) < y:
            hi *= 2.0
        while hi > 1e-300 and self.h(0.5 * hi) >= y:
            hi *= 0.5
        return hi

    def _newton_from_above(self, y, u):
        """Newton iterates for convex increasing h decrease monotonically to the root.

        Convergence is quadratic, so once a step is below 1e-9 relative the
        next iterate is exact to rounding and is returned without a check.
        """
        h, dh = self.h, self.dh
        for _ in range(200):
            excess = h(u) - y
            if excess <= 0.0:
                return u
            slope = dh(u)
            if not slope > 0.0:
                return u
            step = excess / slope
            nxt = u - step
            if not nxt < u:
                return u
            if nxt <= 0.0:
                u = 0.5 * u
                continue
            if step <= 1e-9 * nxt:
                return nxt
            u = nxt
        return u

    # --------------------------------------------------------- Osgood primitive
    def _osgood_quadrature(self, u):
        """J(u) = ∫_u^∞ dr/h by composite Gauss–Legendre in s = log r.

        The integrand e^s/h(e^s) is analytic in a strip around the real axis
        (the zeros of h nearest the positive axis), decays like e^{-(k-1)s},
        and has O(1)-wide crossovers in s wherever one power of h overtakes
        another. Unit panels with 12 points resolve all of these to rounding.
        Panels are added until their contribution falls below 1e-18 of the
        running total. This is the reference for the table and for ``Custom``.
        """
        xs, ws = gauss_rule(12)
        nodes, weights = np.asarray(xs), np.asarray(ws)
        h = self.h
        s0, total, parts = math.log(u), 0.0, []
        rate = max(self.tail_exponent - 1.0, 1e-3)
        for j in range(100000):
            r = np.exp(s0 + j + nodes)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                values = r / np.asarray(h(r), dtype=float)
            values = np.where(np.isfinite(values), values, 0.0)
            part = float(np.dot(weights, values))
            parts.append(part)
            total += part
            # Remaining tail ≲ part·e^{-rate}/(1 - e^{-rate}) once the decay has set in.
            if j > 2 and part <= 1e-18 * total * (1.0 - math.exp(-rate)):
                break
        return math.fsum(parts)

    def _ratio_quadrature(self, u):
        return float(self.h(u)) * self._osgood_quadrature(u) / u

    def _table(self):
        table = self.__dict__.get("_ratio_table")
        if table is None:
            table = self._ratio_table = _RatioTable(self._ratio_quadrature, *self._table_window)
        return table

    def _tail(self, u):
        """J(u) outside the table window: quadrature here, exact series in subclasses."""
        return self._osgood_quadrature(u)

    def osgood(self, u):
        """J(u) = ∫_u^∞ dr / h(r) for u > 0 (inf at 0, 0 at inf)."""
        u = float(u)
        if u <= 0.0:
            return math.inf
        if math.isinf(u):
            return 0.0
        lo, hi = self._table_window
        if lo <= u <= hi:
            return u * self._table()(u) / self.h(u)
        return self._tail(u)

    def osgood_ratio(self, u):
        """R(u) = h(u) J(u) / u, bounded between 1/(k_low - 1) and 1/(k_top - 1)."""
        u = float(u)
        lo, hi = self._table_window
        if lo <= u <= hi:
            return self._table()(u)
        if u <= 0.0 or math.isinf(u):
            return self._ratio_limit(u)
        return self.h(u) * self._tail(u) / u

    def _ratio_limit(self, u):
        return 1.0 / (self.tail_exponent - 1.0) if math.isinf(u) else math.nan

    # ------------------------------------------------------ divided differences
    def dd(self, a, b):
        """(h(b) - h(a))/(b - a) for a, b ≥ 0, without cancellation when b ≈ a."""
        lo, hi = (a, b) if a <= b else (b, a)
        if lo == hi:
            return float(self.dh(lo))
        if lo <= 0.0 or hi > 2.0 * lo:
            return (float(self.h(hi)) - float(self.h(lo))) / (hi - lo)
        return gauss_mean(self.dh, lo, hi)

    def inverse_mean(self, a, b):
        """Mean of 1/h over [a, b] with 0 < a ≤ b ≤ 2a (smooth there)."""
        h = self.h
        return gauss_mean(lambda r: 1.0 / h(r), a, b)

    def q_function(self, u):
        """Q(u) = u + h(u) J(u) = u (1 + R(u)): P(θ, γ) = γ Q(H(θ/γ))."""
        u = float(u)
        if u <= 0.0:
            return 0.0
        if math.isinf(u):
            return math.inf
        return u * (1.0 + self.osgood_ratio(u))

    def q_divided(self, a, b):
        """(Q(b) - Q(a))/(b - a) for a, b ≥ 0, without cancellation when b ≈ a.

        Q(b) - Q(a) = (b - a) + (h(b) - h(a)) J(b) - h(a) ∫_a^b dr/h, so for
        close arguments DDQ = 1 + dd(a, b) J(b) - h(a) mean_{[a,b]}(1/h).
        """
        lo, hi = (a, b) if a <= b else (b, a)
        if lo == hi:
            return float(self.dh(lo)) * self.osgood(lo) if lo > 0.0 else math.inf
        if lo <= 0.0:
            return self.q_function(hi) / hi
        if hi > 2.0 * lo:
            return (self.q_function(hi) - self.q_function(lo)) / (hi - lo)
        return 1.0 + self.dd(lo, hi) * self.osgood(hi) - float(self.h(lo)) * self.inverse_mean(lo, hi)

    # ------------------------------------------------------------ validation
    def validate(self):
        """Check h(0) = 0, positivity, convexity on samples, and a tail exponent k > 1."""
        if float(self.h(0.0)) != 0.0:
            raise ValueError("h(0) must be 0: a nonzero minimum adds the source m g'(x)")
        k = self.tail_exponent
        if k is None or not math.isfinite(k) or k <= 1.0:
            raise ValueError("the tail exponent must exceed 1 (power-type Osgood tail ∫^∞ dr/h < ∞)")
        r = np.geomspace(1e-6, 1e6, 241)
        values = np.asarray(self.h(r), dtype=float)
        if not np.all(values > 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("h must be positive and finite away from 0")
        slopes = np.asarray(self.dh(r), dtype=float)
        if not np.all(slopes > 0.0) or np.any(np.diff(slopes) < -1e-12 * np.abs(slopes[1:])):
            raise ValueError("h must be strictly increasing and convex on (0, ∞)")
        return self

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class Power(Nonlinearity):
    """h(u) = |u|^p, p > 1, with exact H, J and divided differences."""

    def __init__(self, p):
        if isinstance(p, bool):
            raise TypeError("power must be a real number")
        p = float(p)
        if not math.isfinite(p) or p <= 1.0:
            raise ValueError("power p must be finite and greater than 1")
        self.p = self.tail_exponent = p

    def h(self, u):
        if type(u) is float or not np.ndim(u):
            return float(u) ** self.p
        return np.power(u, self.p)

    def dh(self, u):
        if type(u) is float or not np.ndim(u):
            return self.p * float(u) ** (self.p - 1.0)
        return self.p * np.power(u, self.p - 1.0)

    def inverse(self, y):
        return float(y) ** (1.0 / self.p) if y > 0 else 0.0

    def osgood(self, u):
        return float(u) ** (1.0 - self.p) / (self.p - 1.0) if u > 0 else math.inf

    def osgood_ratio(self, u):
        return 1.0 / (self.p - 1.0)

    def dd(self, a, b):
        return divided_power(a, b, self.p)

    def q_divided(self, a, b):
        return self.p / (self.p - 1.0)  # Q(u) = m u

    def __eq__(self, other):
        return isinstance(other, Power) and other.p == self.p

    def __hash__(self):
        return hash(("Power", self.p))

    def __repr__(self):
        return f"Power({self.p:g})"

    def __str__(self):
        return f"|u|^{self.p:g}"


def _series(ratios, gaps, order, cutoff):
    """Coefficients of 1/(1 + Σ a_i z^{d_i}) = Σ_e b_e z^e, dropping |b_e| < cutoff.

    ``ratios`` a_i > 0 and ``gaps`` d_i > 0; returns a dict exponent → b_e.
    The variable is normalised so that z ≤ 1 on the region of use and
    Σ a_i ≤ 1/20, so |b_e| bounds each term there.
    """
    key = lambda e: round(e, 12)
    total = {0.0: 1.0}
    power = {0.0: 1.0}
    for n in range(1, order + 1):
        nxt = {}
        for e, b in power.items():
            for a, d in zip(ratios, gaps):
                k = key(e + d)
                nxt[k] = nxt.get(k, 0.0) - b * a
        power = {e: b for e, b in nxt.items() if abs(b) > cutoff}
        if not power:
            break
        for e, b in power.items():
            total[e] = total.get(e, 0.0) + b
    return {e: b for e, b in total.items() if b != 0.0}


class AbsPolynomial(Nonlinearity):
    """h(u) = Σ c_k |u|^k with c_k > 0 and real exponents k > 1.

    ``terms`` maps exponents to coefficients, e.g. ``{2: 1, 4: 1}`` for
    u² + u⁴, or ``{2: 1, 3: 0.5}`` for u² + |u|³/2. The divided difference is
    exact termwise. H uses Newton from the tight upper bound
    min_k (y/c_k)^{1/k}. Near a zero of g the top exponent governs
    (U ~ |x - z|^{-1/k_max}); for small amplitudes the bottom exponent does.
    J is the exact asymptotic series outside a window [u_lo, u_hi], where the
    lower (respectively upper) terms are at most 1/20 of the leading one, and
    the factored Chebyshev table inside it.
    """

    def __init__(self, terms):
        items = sorted((float(k), float(c)) for k, c in dict(terms).items() if float(c) != 0.0)
        if not items:
            raise ValueError("at least one nonzero coefficient is required")
        for k, c in items:
            if not (math.isfinite(k) and k > 1.0):
                raise ValueError("exponents must be finite and greater than 1")
            if not (math.isfinite(c) and c > 0.0):
                raise ValueError("coefficients must be positive and finite")
        self.exponents = tuple(k for k, _ in items)
        self.coefficients = tuple(c for _, c in items)
        self._terms = list(zip(self.exponents, self.coefficients))
        self._slope_terms = [(k - 1.0, c * k) for k, c in self._terms]
        self.tail_exponent = self.exponents[-1]
        self.single = len(items) == 1
        if not self.single:
            self._build_tails()

    # ------------------------------------------------------------- values
    def h(self, u):
        if type(u) is float or not np.ndim(u):
            total = 0.0
            for k, c in self._terms:
                total += c * float(u) ** k
            return total
        u = np.asarray(u, dtype=float)
        return sum(c * np.power(u, k) for k, c in self._terms)

    def dh(self, u):
        if type(u) is float or not np.ndim(u):
            total = 0.0
            for k, c in self._slope_terms:
                total += c * float(u) ** k
            return total
        u = np.asarray(u, dtype=float)
        return sum(c * np.power(u, k) for k, c in self._slope_terms)

    def _inverse_upper_bound(self, y):
        return min((y / c) ** (1.0 / k) for k, c in self._terms)

    def dd(self, a, b):
        return math.fsum(c * divided_power(a, b, k) for k, c in self._terms)

    # ------------------------------------------------- J: series and table
    def _build_tails(self):
        (k0, c0), (K, cK) = self._terms[0], self._terms[-1]
        target, cutoff = 0.05, 1e-19

        def window(ratios, gaps, sign):
            # Smallest (sign = +1: largest) z with Σ a_i z^{sign·d_i} ≤ target, by bisection in log z.
            def excess(s):
                return sum(a * math.exp(sign * d * s) for a, d in zip(ratios, gaps)) - target
            lo, hi = -300.0, 300.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if (excess(mid) > 0) == (sign > 0):
                    hi = mid
                else:
                    lo = mid
            return math.exp(lo if sign > 0 else hi)

        # Small u: 1/h = r^{-k0}/c0 · Σ_e b_e r^e.
        small_a = [c / c0 for _, c in self._terms[1:]]
        small_d = [k - k0 for k, _ in self._terms[1:]]
        u_lo = window(small_a, small_d, +1)
        # Expand in z = r/u_lo ≤ 1, so the cutoff is relative to the leading term.
        small = {e: b / u_lo ** e for e, b in
                 _series([a * u_lo ** d for a, d in zip(small_a, small_d)], small_d, 80, cutoff).items()}
        # Large u: 1/h = r^{-K}/cK · Σ_e b_e r^{-e}.
        large_a = [c / cK for _, c in self._terms[:-1]]
        large_d = [K - k for k, _ in self._terms[:-1]]
        u_hi = window(large_a, large_d, -1)
        large = {e: b * u_hi ** e for e, b in
                 _series([a * u_hi ** -d for a, d in zip(large_a, large_d)], large_d, 80, cutoff).items()}
        if u_hi < u_lo:  # windows overlap: no middle table is needed
            u_lo = u_hi = math.sqrt(u_lo * u_hi)
        self._table_window = (u_lo, u_hi)
        # J(u) = (1/cK) Σ b_e u^{1-K-e}/(K+e-1) for u ≥ u_hi.
        self._large = [(1.0 - K - e, b / (cK * (K + e - 1.0))) for e, b in sorted(large.items())]
        # J(u) = C - (1/c0) Σ b_e F_e(u), F_e = u^β/β (β = e - k0 + 1) or log u, for u ≤ u_lo.
        small_terms = []
        for e, b in sorted(small.items()):
            beta = e - k0 + 1.0
            small_terms.append((beta, b / c0, abs(beta) < 1e-12))
        self._small = small_terms
        self._small_constant = 0.0
        self._small_constant = self._osgood_quadrature(u_lo) - self._small_series(u_lo)

    def _small_series(self, u):
        """-(1/c0) Σ b_e F_e(u), so that J = C + this near 0."""
        log_u = math.log(u)
        terms = [-b * (log_u if logarithmic else u ** beta / beta) for beta, b, logarithmic in self._small]
        return math.fsum(terms)

    def _tail(self, u):
        lo, hi = self._table_window
        if u >= hi:
            return math.fsum(b * u ** a for a, b in self._large)
        return self._small_constant + self._small_series(u)

    def _ratio_limit(self, u):
        return 1.0 / ((self.exponents[-1] if math.isinf(u) else self.exponents[0]) - 1.0)

    def osgood(self, u):
        if self.single:
            (k, c), = self._terms
            return float(u) ** (1.0 - k) / (c * (k - 1.0)) if u > 0 else math.inf
        return Nonlinearity.osgood(self, u)

    def osgood_ratio(self, u):
        if self.single:
            return 1.0 / (self.exponents[0] - 1.0)
        return Nonlinearity.osgood_ratio(self, u)

    def __eq__(self, other):
        return (isinstance(other, AbsPolynomial) and other.exponents == self.exponents
                and other.coefficients == self.coefficients)

    def __hash__(self):
        return hash(("AbsPolynomial", self.exponents, self.coefficients))

    def __repr__(self):
        return f"AbsPolynomial({dict(self._terms)!r})"

    def __str__(self):
        return " + ".join(f"{'' if c == 1 else f'{c:g}'}|u|^{k:g}" for k, c in self._terms)


class Custom(Nonlinearity):
    """User-supplied even convex h from callables on u ≥ 0.

    ``h`` and ``dh`` must accept numpy arrays. ``tail_exponent`` is a k > 1
    with h(r) ≳ r^k at infinity. It cannot be certified from samples, so the
    caller must state it. ``inverse`` and ``osgood`` may be given as closed
    forms. Otherwise H uses Newton, and J the factored Chebyshev table on
    ``window`` with exp-sinh quadrature outside it.
    """

    def __init__(self, h, dh, tail_exponent, inverse=None, osgood=None, name="h", window=(1e-6, 1e6)):
        self._h, self._dh, self.name = h, dh, name
        self.tail_exponent = float(tail_exponent)
        self._table_window = tuple(map(float, window))
        if inverse is not None:
            self.inverse = lambda y: float(inverse(float(y))) if y > 0 else 0.0
        if osgood is not None:
            self.osgood = lambda u: float(osgood(float(u))) if u > 0 else math.inf
            self.osgood_ratio = lambda u: self.h(u) * self.osgood(u) / u

    def h(self, u):
        return self._h(u) if np.ndim(u) else float(self._h(np.array([float(u)]))[0])

    def dh(self, u):
        return self._dh(u) if np.ndim(u) else float(self._dh(np.array([float(u)]))[0])

    def __repr__(self):
        return f"Custom({self.name})"

    __str__ = __repr__


class Asymmetric(Nonlinearity):
    """h(u) = h₊(u) for u ≥ 0 and h₋(-u) for u < 0, from two even branches h₊, h₋.

    Each branch satisfies the hypotheses on its own: h(0) = 0, strictly
    convex, power-type growth. The glued h is then convex on ℝ with its
    minimum 0 at u = 0, so the θ shock/fan rule is unchanged. A level θ on a
    cell where g has sign η lies on the branch of sgn u = η sgn θ. In
    particular the admissible traces at a zero lie on a single branch: h₊
    where g' < 0 at the zero, h₋ where g' > 0. Its blow-up rate there is
    that branch's top degree.

    ``str`` gives 'u>0: …; u<0: …', which ``parse`` reads back.
    """

    symmetric = False

    def __init__(self, plus, minus):
        self.plus, self.minus = parse(plus), parse(minus)
        if not (self.plus.symmetric and self.minus.symmetric):
            raise ValueError("each branch of an asymmetric h must be even")
        self.tail_exponent = min(self.plus.tail_exponent, self.minus.tail_exponent)

    def branch(self, sign):
        return self.plus if sign > 0 else self.minus

    def h(self, u):
        """h at signed u (the branch is chosen by the sign of u)."""
        if np.ndim(u):
            u = np.asarray(u, dtype=float)
            return np.where(u >= 0, self.plus.h(np.abs(u)), self.minus.h(np.abs(u)))
        u = float(u)
        return self.plus.h(u) if u >= 0 else self.minus.h(-u)

    def dh(self, u):
        """h' at signed u: h₊'(u) for u > 0, -h₋'(-u) for u < 0."""
        if np.ndim(u):
            u = np.asarray(u, dtype=float)
            return np.where(u >= 0, self.plus.dh(np.abs(u)), -self.minus.dh(np.abs(u)))
        u = float(u)
        return self.plus.dh(u) if u >= 0 else -self.minus.dh(-u)

    def __call__(self, u):
        return self.h(u)

    def _branch_only(self, *_):
        raise TypeError("an asymmetric h has no single inverse or Osgood primitive; use branch(sign)")

    inverse = inverse_array = osgood = osgood_ratio = dd = q_function = q_divided = inverse_mean = _branch_only

    def validate(self):
        self.plus.validate()
        self.minus.validate()
        return self

    def __eq__(self, other):
        return isinstance(other, Asymmetric) and other.plus == self.plus and other.minus == self.minus

    def __hash__(self):
        return hash(("Asymmetric", self.plus, self.minus))

    def __repr__(self):
        return f"Asymmetric({self.plus!r}, {self.minus!r})"

    def __str__(self):
        return f"u>0: {self.plus}; u<0: {self.minus}"


def parse(spec):
    """Parse a nonlinearity specification into a ``Nonlinearity``.

    Accepted: a power ('2.5' or 2.5), a sum of terms c|u|^k ('|u|^3 + 0.5u^2',
    'u^2+u^4'), or two branches separated by ';' for an asymmetric h. In the
    last case the plus branch comes first, or each is marked
    ('u>0: u^2; u<0: 3u^4'). Equal branches give the even h.
    """
    if isinstance(spec, Nonlinearity):
        return spec
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Power(spec)
    text = str(spec)
    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"cannot read {spec!r}: an asymmetric h needs exactly two branches 'h+; h-'")
        branches = {}
        for i, part in enumerate(parts):
            key = "plus" if i == 0 else "minus"
            compact = part.replace(" ", "")
            for marker, name in (("u>0:", "plus"), ("u<0:", "minus")):
                if compact.startswith(marker):
                    key, part = name, compact[len(marker):]
            if key in branches:
                raise ValueError(f"cannot read {spec!r}: branch {key} given twice")
            branches[key] = parse(part)
        if branches["plus"] == branches["minus"]:
            return branches["plus"]
        return Asymmetric(branches["plus"], branches["minus"])
    text = text.replace(" ", "").replace("|u|", "u").replace("**", "^").replace("-", "+-")
    try:
        return Power(float(text))
    except ValueError:
        pass
    terms = {}
    for part in filter(None, text.split("+")):
        coefficient, _, exponent = part.partition("u^")
        if not _ and part.endswith("u"):
            coefficient, exponent = part[:-1], "1"
        coefficient = coefficient.rstrip("*")
        try:
            c = float(coefficient) if coefficient not in ("", "+") else 1.0
            k = float(exponent)
        except ValueError:
            raise ValueError(f"cannot read {spec!r}: expected a power p or a sum of terms c|u|^k "
                             "(e.g. 'u^2+u^4'); use nonlinearity.Custom for other h") from None
        terms[k] = terms.get(k, 0.0) + c
    if len(terms) == 1:
        (k, c), = terms.items()
        if c == 1.0:
            return Power(k)
    return AbsPolynomial(terms)
