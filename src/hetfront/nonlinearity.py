"""Convex nonlinearities h for the multiplicative flux g(x) h(u).

A ``Nonlinearity`` is an even function h(u) = h(|u|) with

* h(0) = 0 and h > 0 elsewhere, strictly convex on [0, ∞);
* the Osgood tail  ∫^∞ dr / h(r) < ∞.

The tail condition is exactly what makes stationary profiles integrable at
transverse zeros of g, makes fronts reach those zeros in finite time, and
makes the characteristic equation ṗ = -g'(q) h(p) blow up in finite time.
Every closed form in ``trajectory.GeneralLaw`` is built from three functions
on [0, ∞):

    h,     H = h⁻¹ (``inverse``),     J(u) = ∫_u^∞ dr / h(r) (``osgood``),

together with a cancellation-free divided difference ``dd(a, b)`` of h and
the mean of 1/h over an interval (``inverse_mean``). All methods take and
return Python floats for u, y ≥ 0; the sign of u is handled by the caller.

``Power(p)`` is |u|^p with exact formulas. ``AbsPolynomial`` is
Σ c_k |u|^{k} with c_k ≥ 0 and real k > 1. ``Custom`` wraps user callables
with numerical fallbacks.
"""

import math

import numpy as np

from .trajectory import divided_power

# Gauss–Legendre nodes and weights on [0, 1].
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def gauss_mean(f, a, b):
    """Mean of a smooth vectorised f over [a, b] by 16-point Gauss–Legendre."""
    if a == b:
        return float(f(np.array([a]))[0])
    return float(np.dot(_GL_W, f(a + (b - a) * _GL_X)))


def _exp_sinh_nodes(t_max, step=1.0 / 16.0):
    """Nodes x and weights w with ∫_0^∞ F(x) dx ≈ Σ w F(x) (exp-sinh rule)."""
    t = np.arange(-6.0, t_max + step / 2, step)
    x = np.exp(0.5 * math.pi * np.sinh(t))
    w = step * 0.5 * math.pi * np.cosh(t) * x
    return x, w


class Nonlinearity:
    """Base class: generic numerics from ``h`` and ``dh`` alone.

    Subclasses must provide ``h`` and ``dh`` (vectorised on arrays of u ≥ 0)
    and ``tail_exponent``, a number k > 1 with h(r) ≳ r^k as r → ∞. That is
    the exponent that governs the profiles' blow-up rate at a zero.
    """

    tail_exponent = None

    # ------------------------------------------------------------ required
    def h(self, u):
        raise NotImplementedError

    def dh(self, u):
        raise NotImplementedError

    # ------------------------------------------------------------ generic
    def __call__(self, u):
        return self.h(np.abs(u))

    def inverse(self, y):
        """H(y) = h⁻¹(y) on [0, ∞), by Newton from above (monotone for convex h)."""
        y = float(y)
        if y <= 0.0:
            return 0.0
        if math.isinf(y):
            return math.inf
        hi = self._inverse_upper_bound(y)
        return self._newton_from_above(y, hi)

    def _inverse_upper_bound(self, y):
        hi = 1.0
        while self.h(hi) < y:
            hi *= 2.0
        while hi > 1e-300 and self.h(0.5 * hi) >= y:
            hi *= 0.5
        return hi

    def _newton_from_above(self, y, u):
        """Newton iterates for convex increasing h decrease monotonically to the root."""
        for _ in range(200):
            excess = self.h(u) - y
            if excess <= 0.0:
                return u
            slope = self.dh(u)
            if not slope > 0.0:
                return u
            nxt = u - excess / slope
            if not nxt < u:
                return u
            u = nxt if nxt > 0.0 else 0.5 * u
        return u

    def osgood(self, u):
        """J(u) = ∫_u^∞ dr / h(r) for u > 0 (inf at u = 0), by exp-sinh quadrature."""
        u = float(u)
        if u <= 0.0:
            return math.inf
        if math.isinf(u):
            return 0.0
        x, w = self._de_rule()
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            values = w / self.h(u * (1.0 + x))
        values = np.where(np.isfinite(values), values, 0.0)
        return u * math.fsum(values.tolist())

    def _de_rule(self):
        rule = getattr(self, "_de_cache", None)
        if rule is None:
            k = float(self.tail_exponent)
            # Need (k-1)·(π/2)·sinh(t_max) ≳ 40, so that x^{1-k} at the cut is below 1e-17.
            t_max = max(4.0, math.asinh(80.0 / (math.pi * (k - 1.0))))
            rule = self._de_cache = _exp_sinh_nodes(t_max)
        return rule

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
        return gauss_mean(lambda r: 1.0 / self.h(r), a, b)

    def q_function(self, u):
        """Q(u) = u + h(u) J(u), the profile primitive: P(θ, γ) = γ Q(H(θ/γ))."""
        u = float(u)
        if u <= 0.0:
            return 0.0
        return u + float(self.h(u)) * self.osgood(u)

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

    def validate(self):
        """Check h(0) = 0, positivity, convexity on samples, and the Osgood tail."""
        if float(self.h(0.0)) != 0.0:
            raise ValueError("h(0) must be 0: a nonzero minimum adds the source m g'(x)")
        k = self.tail_exponent
        if k is None or not math.isfinite(k) or k <= 1.0:
            raise ValueError("the tail exponent must exceed 1 (Osgood condition ∫^∞ dr/h < ∞)")
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
        p = float(p)
        if not math.isfinite(p) or p <= 1.0:
            raise ValueError("power p must be finite and greater than 1")
        self.p = self.tail_exponent = p

    def h(self, u):
        return np.power(u, self.p) if np.ndim(u) else float(u) ** self.p

    def dh(self, u):
        return self.p * (np.power(u, self.p - 1.0) if np.ndim(u) else float(u) ** (self.p - 1.0))

    def inverse(self, y):
        return float(y) ** (1.0 / self.p) if y > 0 else 0.0

    def osgood(self, u):
        return float(u) ** (1.0 - self.p) / (self.p - 1.0) if u > 0 else math.inf

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


class AbsPolynomial(Nonlinearity):
    """h(u) = Σ c_k |u|^k with c_k ≥ 0 and real exponents k > 1.

    ``terms`` maps exponents to coefficients, e.g. ``{2: 1, 4: 1}`` for
    u² + u⁴, or ``{2: 1, 3: 0.5}`` for u² + |u|³/2. The divided difference is
    exact termwise. H uses Newton from the tight upper bound
    min_k (y/c_k)^{1/k}. Near a zero of g the top exponent governs
    (U ~ |x - z|^{-1/k_max}), while for small amplitudes the bottom exponent does.
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
        self.tail_exponent = self.exponents[-1]

    def h(self, u):
        if np.ndim(u):
            u = np.asarray(u, dtype=float)
            return sum(c * np.power(u, k) for k, c in zip(self.exponents, self.coefficients))
        u = float(u)
        return math.fsum(c * u ** k for k, c in zip(self.exponents, self.coefficients))

    def dh(self, u):
        if np.ndim(u):
            u = np.asarray(u, dtype=float)
            return sum(c * k * np.power(u, k - 1.0) for k, c in zip(self.exponents, self.coefficients))
        u = float(u)
        return math.fsum(c * k * u ** (k - 1.0) for k, c in zip(self.exponents, self.coefficients))

    def _inverse_upper_bound(self, y):
        return min((y / c) ** (1.0 / k) for k, c in zip(self.exponents, self.coefficients))

    def dd(self, a, b):
        return math.fsum(c * divided_power(a, b, k) for k, c in zip(self.exponents, self.coefficients))

    def __eq__(self, other):
        return (isinstance(other, AbsPolynomial) and other.exponents == self.exponents
                and other.coefficients == self.coefficients)

    def __hash__(self):
        return hash(("AbsPolynomial", self.exponents, self.coefficients))

    def __repr__(self):
        terms = " + ".join(f"{c:g}|u|^{k:g}" for k, c in zip(self.exponents, self.coefficients))
        return f"AbsPolynomial({terms})"


class Custom(Nonlinearity):
    """User-supplied even convex h from callables on u ≥ 0.

    ``h`` and ``dh`` must accept numpy arrays. ``tail_exponent`` is a k > 1
    with h(r) ≳ r^k at infinity. It cannot be certified from samples, so the
    caller must state it. ``inverse`` and ``osgood`` may be given as closed
    forms; otherwise Newton and exp-sinh quadrature are used.
    """

    def __init__(self, h, dh, tail_exponent, inverse=None, osgood=None, name="h"):
        self._h, self._dh, self.name = h, dh, name
        self.tail_exponent = float(tail_exponent)
        if inverse is not None:
            self.inverse = lambda y: float(inverse(float(y))) if y > 0 else 0.0
        if osgood is not None:
            self.osgood = lambda u: float(osgood(float(u))) if u > 0 else math.inf

    def h(self, u):
        return self._h(u) if np.ndim(u) else float(self._h(np.array([float(u)]))[0])

    def dh(self, u):
        return self._dh(u) if np.ndim(u) else float(self._dh(np.array([float(u)]))[0])

    def __repr__(self):
        return f"Custom({self.name})"


def parse(spec):
    """Parse '|u|^3 + 0.5u^2', 'u^2+u^4' or '2.5' (a power) into a Nonlinearity."""
    if isinstance(spec, Nonlinearity):
        return spec
    if isinstance(spec, (int, float)):
        return Power(spec)
    text = str(spec).replace(" ", "").replace("|u|", "u").replace("**", "^").replace("-", "+-")
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
        c = float(coefficient) if coefficient not in ("", "+") else 1.0
        k = float(exponent)
        terms[k] = terms.get(k, 0.0) + c
    if len(terms) == 1:
        (k, c), = terms.items()
        if c == 1.0:
            return Power(k)
    return AbsPolynomial(terms)
