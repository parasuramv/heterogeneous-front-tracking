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
calls the module functions unchanged.
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
