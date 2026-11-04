"""Closed-form motion of fronts and characteristics inside one affine cell.

On a cell where |g_Δ(x)| = σ(x)² is affine with slope α = d|g|/dx, a front
joining flux levels θ_L, θ_R moves with ẋ = σ κ, where κ = s(θ_L) + s(θ_R)
and s(θ) = sgn(θ) sqrt|θ|. Then σ̇ = α κ / 2 is constant: in the straightened
coordinate σ every front is a straight line, and in x it is a parabola in t,

    σ(τ) = σ₀ + α κ τ / 2,        x(τ) = x₀ + κ τ (σ₀ + α κ τ / 4).

A genuine characteristic carrying level θ = s|s| is the same with κ = 2s.
Event times solve *linear* equations in σ. They are written below in forms
that never divide by α, so flat cells (α = 0) need no special case.

The module functions are the quadratic case h(u) = u². ``PowerLaw`` gives the
same laws for h(u) = |u|^p, p > 1, with σ = |g_Δ|^{(p-1)/p}; for p = 2 it
calls the module functions unchanged. ``GeneralLaw`` covers general convex
h with the Osgood tail ∫^∞ dr/h < ∞ (``nonlinearity``). Arrival times stay
in closed form; positions at a given time and meeting times become monotone
scalar roots, since only power laws admit a straightening coordinate.
"""

import math

import numpy as np


def position(x0, s0, alpha, kappa, tau):
    """x after time tau from anchor (x0, σ0)."""
    return x0 + kappa * tau * (s0 + 0.25 * alpha * kappa * tau)


def sigma(s0, alpha, kappa, tau):
    """σ after time tau from anchor σ0 (never negative within a cell)."""
    value = s0 + 0.5 * alpha * kappa * tau
    return value if value > 0.0 else 0.0


def travel_time(x0, s0, x1, s1, kappa):
    """Time to move from (x0, σ0) to (x1, σ1) in the same cell, or inf.

    From σ1 - σ0 = α(x1 - x0)/(σ0 + σ1) and σ̇ = ακ/2:
    τ = 2 (x1 - x0) / (κ (σ0 + σ1)).
    """
    if x1 == x0:
        return 0.0
    denominator = kappa * (s0 + s1)
    if denominator == 0.0:
        return math.inf
    tau = 2.0 * (x1 - x0) / denominator
    return tau if tau >= 0.0 else math.inf


def meeting_time(xa, sa, ka, xb, sb, kb):
    """Waiting time until front a (left) meets front b (right) in one cell.

    Both are given at the same instant. They close iff κ_a > κ_b because σ is
    monotone in x on the cell; then τ = 2 (x_b - x_a) / ((σ_a + σ_b)(κ_a - κ_b)).
    """
    closing = ka - kb
    if closing <= 0.0:
        return math.inf
    gap = xb - xa
    if gap <= 0.0:
        return 0.0
    total = sa + sb
    if total <= 0.0:
        return math.inf
    return 2.0 * gap / (total * closing)


def sample_path(x0, s0, alpha, kappa, t0, t1, samples=24):
    """Points (x, t) along one closed-form piece, for plotting."""
    t = np.linspace(t0, t1, max(2, int(samples)))
    tau = t - t0
    return position(x0, s0, alpha, kappa, tau), t


def divided_power(a, b, e):
    """(b^e - a^e)/(b - a) for a, b >= 0, accurate as b -> a (limit e a^(e-1)).

    Scalars only. For nearby arguments the difference of powers is written as
    a^e expm1(e log1p((b-a)/a)), so there is no cancellation when b ≈ a.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    if hi == lo:
        return e * lo ** (e - 1.0) if lo > 0.0 else (0.0 if e > 1.0 else math.inf)
    if lo <= 0.0:
        return hi ** (e - 1.0)
    if hi > 2.0 * lo:  # well separated: no cancellation, and no underflow of lo**e
        return (hi ** e - lo ** e) / (hi - lo)
    return lo ** e * math.expm1(e * math.log1p((hi - lo) / lo)) / (hi - lo)


def divided_power_array(a, b, e):
    """Vectorised ``divided_power``."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        close = lo ** e * np.expm1(e * np.log1p((hi - lo) / lo)) / (hi - lo)
        apart = (hi ** e - lo ** e) / (hi - lo)
        generic = np.where(hi > 2.0 * lo, apart, close)
        from_zero = hi ** (e - 1.0)
        equal = np.where(lo > 0.0, e * lo ** (e - 1.0), 0.0 if e > 1.0 else np.inf)
    return np.where(hi == lo, equal, np.where(lo <= 0.0, from_zero, generic))


class PowerLaw:
    """Closed-form front laws for h(u) = |u|^p with p > 1, cell by cell.

    With q = 1 - 1/p and σ = |g_Δ|^q, a front whose levels give the secant
    coefficient K moves with ẋ = |g_Δ|^{1/p} K and σ̇ = q α K: it is straight
    in σ. Since |g_Δ| = σ^m (m = p/(p-1)) is affine, α (x1 - x0) = σ1^m - σ0^m,
    and every law uses the same two-point mean of the speed factor |g|^{1/p},

        M(σ0, σ1) = q (σ1^m - σ0^m)/(σ1 - σ0) = (x1 - x0)/(K τ),

    so no formula divides by α. For p = 2, σ = sqrt|g|, M = (σ0 + σ1)/2, and
    the quadratic functions above are used unchanged.
    """

    __slots__ = ("p", "q", "m", "quadratic")

    def __init__(self, p=2.0):
        if isinstance(p, bool):
            raise TypeError("power must be a real number")
        p = float(p)
        if not math.isfinite(p) or p <= 1.0:
            raise ValueError("power p must be finite and greater than 1")
        self.p, self.q, self.m = p, 1.0 - 1.0 / p, p / (p - 1.0)
        self.quadratic = p == 2.0

    def __eq__(self, other):
        return isinstance(other, PowerLaw) and other.p == self.p

    def __hash__(self):
        return hash(self.p)

    def __repr__(self):
        return f"PowerLaw(p={self.p:g})"

    # ------------------------------------------------------------ σ and M
    def sigma_of(self, absg):
        """σ = |g|^q from |g| (scalar or array)."""
        if np.ndim(absg):
            return np.sqrt(absg) if self.quadratic else np.power(absg, self.q)
        return math.sqrt(absg) if self.quadratic else absg ** self.q

    def mean(self, s0, s1):
        """M(σ0, σ1), the mean speed factor between two points of one cell."""
        if self.quadratic:
            return 0.5 * (s0 + s1)
        return self.q * divided_power(s0, s1, self.m)

    def mean_array(self, s0, s1):
        if self.quadratic:
            return 0.5 * (np.asarray(s0) + np.asarray(s1))
        return self.q * divided_power_array(s0, s1, self.m)

    # --------------------------------------------------------- front laws
    def sigma(self, s0, alpha, kappa, tau):
        """σ after time tau: σ0 + q α K τ (never negative within a cell)."""
        if self.quadratic and not np.ndim(tau):
            return sigma(s0, alpha, kappa, tau)
        value = s0 + self.q * alpha * kappa * tau
        if np.ndim(value):
            return np.maximum(value, 0.0)
        return value if value > 0.0 else 0.0

    def position(self, x0, s0, alpha, kappa, tau):
        """x after time tau from anchor (x0, σ0): x0 + K τ M(σ0, σ(τ))."""
        if self.quadratic:
            return position(x0, s0, alpha, kappa, tau)
        s1 = self.sigma(s0, alpha, kappa, tau)
        if np.ndim(tau):
            return x0 + kappa * np.asarray(tau) * self.mean_array(s0, s1)
        return x0 + kappa * tau * self.mean(s0, s1)

    def travel_time(self, x0, s0, x1, s1, kappa):
        """Time to move from (x0, σ0) to (x1, σ1) in one cell: (x1 - x0)/(K M), or inf."""
        if self.quadratic:
            return travel_time(x0, s0, x1, s1, kappa)
        if x1 == x0:
            return 0.0
        denominator = kappa * self.mean(s0, s1)
        if denominator == 0.0:
            return math.inf
        tau = (x1 - x0) / denominator
        return tau if tau >= 0.0 else math.inf

    def meeting_time(self, xa, sa, ka, xb, sb, kb):
        """Waiting time until front a (left) meets front b (right) in one cell.

        σ is monotone in x on the cell and linear in t along each front, so
        τ = (σ_b - σ_a)/(q α (K_a - K_b)) = (x_b - x_a)/(M(σ_a, σ_b)(K_a - K_b)).
        """
        if self.quadratic:
            return meeting_time(xa, sa, ka, xb, sb, kb)
        closing = ka - kb
        if closing <= 0.0:
            return math.inf
        gap = xb - xa
        if gap <= 0.0:
            return 0.0
        speed = self.mean(sa, sb)
        if speed <= 0.0:
            return math.inf
        return gap / (speed * closing)

    def characteristic_kappa(self, theta):
        """K of a genuine characteristic carrying level θ: h'(r) = p sgn(θ)|θ|^q."""
        if self.quadratic:
            return 2.0 * (math.copysign(math.sqrt(abs(theta)), theta) if theta else 0.0)
        return math.copysign(self.p * abs(theta) ** self.q, theta) if theta else 0.0

    def sample_path(self, x0, s0, alpha, kappa, t0, t1, samples=24):
        """Points (x, t) along one closed-form piece, for plotting."""
        t = np.linspace(t0, t1, max(2, int(samples)))
        return self.position(x0, s0, alpha, kappa, t - t0), t


QUADRATIC = PowerLaw(2.0)


# ---------------------------------------------------------------------------
# General convex h: closed-form arrival times, monotone root solves.
# ---------------------------------------------------------------------------

def _monotone_root(F, lo, hi, f_lo=None, f_hi=None):
    """Root of a monotone F on [lo, hi] with F(lo) ≤ 0 ≤ F(hi) (Illinois + bisection).

    lo may exceed hi (a decreasing direction); the bracket is shrunk until
    it is a few ulps wide or F vanishes. Returns the end with |F| smallest.
    """
    f_lo = F(lo) if f_lo is None else f_lo
    f_hi = F(hi) if f_hi is None else f_hi
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    side = 0
    for _ in range(300):
        width = abs(hi - lo)
        if width <= 4.0 * math.ulp(max(abs(lo), abs(hi))):
            break
        x = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        if not (min(lo, hi) < x < max(lo, hi)) or side == 3 or side == -3:
            x, side = 0.5 * (lo + hi), 0
        fx = F(x)
        if fx == 0.0:
            return x
        if fx < 0.0:
            lo, f_lo = x, fx
            if side < 0:
                f_hi *= 0.5
            side = side - 1 if side <= 0 else -1
        else:
            hi, f_hi = x, fx
            if side > 0:
                f_lo *= 0.5
            side = side + 1 if side >= 0 else 1
    return lo if abs(f_lo) <= abs(f_hi) else hi


class FrontCoef(tuple):
    """(θ_L, θ_R, |θ_R| - |θ_L|, direction) of one front or characteristic (θ_L = θ_R)."""

    __slots__ = ()

    def __new__(cls, thL, thR):
        thL, thR = float(thL), float(thR)
        dabs = abs(thR) - abs(thL)
        if thL == thR:
            direction = (thL > 0) - (thL < 0)
        else:
            ratio = dabs / (thR - thL)
            direction = (ratio > 0) - (ratio < 0)
        return tuple.__new__(cls, (thL, thR, dabs, direction))

    thL = property(lambda self: self[0])
    thR = property(lambda self: self[1])
    dabs = property(lambda self: self[2])
    direction = property(lambda self: self[3])

    @property
    def characteristic(self):
        return self[0] == self[1]


class GeneralLaw:
    """Front laws for f = g_Δ(x) h(u) with a general convex ``Nonlinearity`` h.

    The anchor coordinate is γ = |g_Δ| itself (``sigma_of`` is the identity),
    affine on a cell with slope α. With H = h⁻¹, J(u) = ∫_u^∞ dr/h and
    Q(u) = u + h(u)J(u), the profile primitive P(θ, γ) = γ Q(H(θ/γ)) gives

        ∂_γ P = H(θ/γ)  (the state),   ∂_θ P = J(u),   ∂_θ∂_γ P = 1/(γ h'(u)).

    A front (θ_L, θ_R) has slowness S = dt/dx = [H(θ/γ)]/[|θ|]. Its
    γ-antiderivative is L = (P(θ_R, γ) - P(θ_L, γ))/[|θ|], so

        α (t1 - t0) = L(γ1) - L(γ0).

    A characteristic has S = 1/(γ h'(u)) and L = sgn θ · J(u). For h = |u|^p,
    L ∝ σ = γ^{1-1/p}, which is ``PowerLaw``. In general no single coordinate
    straightens all fronts (that requires h'(u) = c h(u)^r, i.e. a power law),
    so arrival times at a given point are explicit, while positions at a given
    time and meeting times are monotone scalar roots. Times are computed as
    (x1 - x0) times a mean slowness, with no division by α: a closed form when
    γ0 and γ1 are far apart, and Gauss–Legendre on the smooth S otherwise.

    The front coefficient is a ``FrontCoef`` built by ``coef(θ_L, θ_R)``.
    """

    quadratic = False

    def __init__(self, nonlinearity):
        from .nonlinearity import parse
        self.h = parse(nonlinearity).validate()

    def __eq__(self, other):
        return isinstance(other, GeneralLaw) and other.h == self.h

    def __hash__(self):
        return hash(("GeneralLaw", self.h))

    def __repr__(self):
        return f"GeneralLaw({self.h!r})"

    @staticmethod
    def coef(thL, thR):
        return FrontCoef(thL, thR)

    @staticmethod
    def direction(coef):
        return coef.direction

    # -------------------------------------------------------------- σ ≡ γ
    @staticmethod
    def sigma_of(absg):
        return np.array(absg, dtype=float) if np.ndim(absg) else float(absg)

    def state(self, theta, gamma):
        """|u| = H(|θ|/γ) on the profile of level θ where |g_Δ| = γ (inf at γ = 0)."""
        if theta == 0.0:
            return 0.0
        if gamma <= 0.0:
            return math.inf
        return self.h.inverse(abs(theta) / gamma)

    def speed(self, coef, gamma):
        """ẋ at |g_Δ| = γ (0 for a stationary front)."""
        s = self.slowness(coef, gamma)
        return 0.0 if math.isinf(s) else 1.0 / s

    def closing(self, ca, cb, gamma):
        """True if front a (left) is faster than front b (right).

        The order of the secant speeds at a common γ does not depend on γ,
        because H is monotone and h is convex, so any γ on the cell decides.
        """
        return self.speed(ca, gamma) > self.speed(cb, gamma)

    # ------------------------------------------------ slowness and its primitive
    def slowness(self, coef, gamma):
        """S = dt/dx at |g_Δ| = γ > 0 (signed; ±inf for a stationary front)."""
        thL, thR, dabs, direction = coef
        if direction == 0:
            return math.inf
        h = self.h
        if thL == thR:
            return direction / (gamma * h.dh(self.state(thL, gamma)))
        if thL * thR >= 0.0:
            a, b = self.state(thL, gamma), self.state(thR, gamma)
            sign = 1.0 if thL + thR > 0 else -1.0
            return sign / (gamma * h.dd(a, b))
        uR = math.copysign(self.state(thR, gamma), thR)
        uL = math.copysign(self.state(thL, gamma), thL)
        return (uR - uL) / dabs

    def primitive(self, coef, gamma):
        """L(γ) with dL/dγ = S and L(0) = 0, so that α Δt = ΔL."""
        if gamma <= 0.0:
            return 0.0
        thL, thR, dabs, direction = coef
        h = self.h
        if thL == thR:
            return direction * h.osgood(self.state(thL, gamma)) if thL else 0.0
        if thL * thR >= 0.0:
            a, b = self.state(thL, gamma), self.state(thR, gamma)
            sign = 1.0 if thL + thR > 0 else -1.0
            return sign * h.q_divided(a, b) / h.dd(a, b)
        qR = math.copysign(h.q_function(self.state(thR, gamma)), thR)
        qL = math.copysign(h.q_function(self.state(thL, gamma)), thL)
        return gamma * (qR - qL) / dabs if dabs else math.copysign(math.inf, qR - qL)

    def mean_slowness(self, coef, g0, g1):
        """Mean of S over γ between g0 and g1 (equivalently over x on one cell)."""
        if g0 == g1:
            return self.slowness(coef, g0)
        lo, hi = (g0, g1) if g0 < g1 else (g1, g0)
        if lo <= 0.0 or hi > 2.0 * lo:
            return (self.primitive(coef, g1) - self.primitive(coef, g0)) / (g1 - g0)
        values = [self.slowness(coef, lo + (hi - lo) * float(s)) for s in _gl_unit()[0]]
        return math.fsum(w * v for w, v in zip(_gl_unit()[1].tolist(), values))

    # -------------------------------------------------------------- front laws
    def travel_time(self, x0, s0, x1, s1, coef):
        """Time to move from (x0, γ0) to (x1, γ1) in one cell, or inf."""
        if x1 == x0:
            return 0.0
        if coef.direction == 0 or (x1 - x0) * coef.direction < 0:
            return math.inf
        tau = (x1 - x0) * self.mean_slowness(coef, s0, s1)
        return tau if tau >= 0.0 else math.inf

    def _zero_location(self, x0, s0, alpha, direction):
        """Where γ reaches 0 along the direction of motion, or None."""
        if alpha == 0.0 or alpha * direction >= 0.0:
            return None
        return x0 - s0 / alpha

    def position(self, x0, s0, alpha, coef, tau):
        """x after time tau from anchor (x0, γ0), by a monotone root solve.

        Arrays of tau are evaluated elementwise. A front that reaches a zero
        of g_Δ stays there, as σ is clamped at 0 in ``PowerLaw``.
        """
        if np.ndim(tau):
            return np.array([self.position(x0, s0, alpha, coef, float(t)) for t in np.ravel(tau)]
                            ).reshape(np.shape(tau))
        d = coef.direction
        if tau <= 0.0 or d == 0:
            return x0
        if alpha == 0.0:
            return x0 + tau / self.slowness(coef, s0)

        def gamma(x):
            value = s0 + alpha * (x - x0)
            return value if value > 0.0 else 0.0

        def F(x):
            return self.travel_time(x0, s0, x, gamma(x), coef) - tau

        z = self._zero_location(x0, s0, alpha, d)
        if z is not None:
            f_z = F(z)
            if f_z <= 0.0:
                return z
            far, f_far = z, f_z
        else:
            step = tau / abs(self.slowness(coef, s0))
            far = x0 + d * step
            f_far = F(far)
            for _ in range(200):
                if f_far >= 0.0:
                    break
                step *= 2.0
                far = x0 + d * step
                f_far = F(far)
        return _monotone_root(F, x0, far, -tau, f_far)

    def sigma(self, s0, alpha, coef, tau, x0=0.0):
        """γ after time tau from anchor γ0 (never negative within a cell)."""
        x = self.position(x0, s0, alpha, coef, tau)
        value = s0 + alpha * (np.asarray(x) - x0)
        return np.maximum(value, 0.0) if np.ndim(value) else max(float(value), 0.0)

    def meeting_time(self, xa, sa, ca, xb, sb, cb, alpha, x_end=None):
        """Waiting time until front a (left) meets front b (right) in one cell, or inf.

        For fronts closing at a common γ, D(x) = T_a(x) - T_b(x) is strictly
        monotone along the common path, so the meeting point is a bracketed
        root. ``x_end`` is the cell end in the direction of a common motion;
        a meeting beyond it, or beyond a zero of g_Δ, returns inf (the corner
        or absorption comes first).
        """
        if not self.closing(ca, cb, sa if sa > 0.0 else sb):
            return math.inf
        if xb - xa <= 0.0:
            return 0.0
        da, db = ca.direction, cb.direction

        def gamma(x):
            value = sa + alpha * (x - xa)
            return value if value > 0.0 else 0.0

        def Ta(x):
            return self.travel_time(xa, sa, x, gamma(x), ca)

        def Tb(x):
            return self.travel_time(xb, sb, x, gamma(x), cb)

        if da == 0:
            return Tb(xa)
        if db == 0:
            return Ta(xb)
        if da > 0 > db:
            x = _monotone_root(lambda y: Ta(y) - Tb(y), xa, xb, -Tb(xa), Ta(xb))
            return 0.5 * (Ta(x) + Tb(x))
        # Common direction d: the leader is b for d > 0 and a for d < 0.
        d = da
        lead_x, lead_s = (xb, sb) if d > 0 else (xa, sa)
        lag = Ta if d > 0 else Tb
        lead = Tb if d > 0 else Ta

        def D(y):  # lead time minus lag time: negative at the leader, increasing ahead of it
            return lead(y) - lag(y)

        stop = self._zero_location(lead_x, lead_s, alpha, d)
        if x_end is not None:
            stop = x_end if stop is None else (min(stop, x_end) if d > 0 else max(stop, x_end))
        if stop is not None:
            d_stop = D(stop)
            if d_stop < 0.0:
                return math.inf
        else:
            step = abs(xb - xa)
            stop, d_stop = lead_x + d * step, D(lead_x + d * step)
            for _ in range(200):
                if d_stop >= 0.0:
                    break
                step *= 2.0
                stop = lead_x + d * step
                d_stop = D(stop)
            else:
                return math.inf
        x = _monotone_root(D, lead_x, stop, D(lead_x), d_stop)
        return 0.5 * (Ta(x) + Tb(x))

    def characteristic_coef(self, theta):
        return FrontCoef(theta, theta)

    def sample_path(self, x0, s0, alpha, coef, t0, t1, samples=24):
        """Points (x, t) along one piece, uniform in x, with t explicit (no root solves)."""
        x1 = self.position(x0, s0, alpha, coef, t1 - t0)
        xs = np.linspace(x0, x1, max(2, int(samples)))
        ts = [t0 + self.travel_time(x0, s0, x, max(s0 + alpha * (x - x0), 0.0), coef) for x in xs]
        return xs, np.minimum(np.asarray(ts, dtype=float), t1)


def _gl_unit():
    from .nonlinearity import _GL_X, _GL_W
    return _GL_X, _GL_W
