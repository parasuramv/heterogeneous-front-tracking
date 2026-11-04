"""Multiplicative flux f(x,u) = g_Δ(x) h(u) with quantised flux levels.

h is |u|^p (default p = 2) or, with ``h=``, a general even convex
nonlinearity with a power-type Osgood tail (``nonlinearity``). For a general h
a level k has |u| = H(|k|δ/|g_Δ|) with H = h⁻¹, and fronts follow
``trajectory.GeneralLaw`` with anchor γ = |g_Δ|. Everything below the next
paragraph is the power-law case.

The discretised quantity is the flux-value function θ = G(x,u) = sgn(u) f(x,u),
not the flux itself. A level is an integer k with θ = kδ. On a cell,

    s(k) = sgn(k) (|k| δ)^{1/p},   u = sgn(g) s(k) / |g_Δ(x)|^{1/p},

and the flux value is f = sgn(g) |θ|. The sign of θ is the direction of the
characteristic speed f_u = |g|^{1/p} p sgn(θ)|θ|^{1-1/p}. Fronts are straight
in σ = |g_Δ|^{1-1/p} (``trajectory.PowerLaw``); for p = 2, s = sgn(k)sqrt(|k|δ)
and σ = sqrt|g_Δ|. Stationary profiles are the constant-level states; near a
zero of g they blow up like |x - z|^{-1/p} but stay integrable.
"""

import math

import numpy as np

from .heterogeneity import PiecewiseAffine
from .nonlinearity import Power, parse
from .trajectory import GeneralLaw, PowerLaw, divided_power
from ._validation import integer_levels, positive_count


class MultiplicativeFlux:
    """The flux g_Δ(x) h(u) together with the level spacing δ > 0.

    ``power=p`` gives h = |u|^p (p > 1, default 2). ``h`` gives a general
    nonlinearity: a ``nonlinearity.Nonlinearity``, or a string such as
    ``"u^2+u^4"`` or ``"|u|^3 + 0.5u^2"``. A pure power given through ``h``
    uses the exact power-law path. ``power`` is None for a general h, and
    ``nonlinearity`` is always set.
    """

    def __init__(self, g: PiecewiseAffine, delta: float, power: float = 2.0, h=None):
        if not isinstance(g, PiecewiseAffine):
            raise TypeError("g must be a PiecewiseAffine heterogeneity")
        delta = float(delta)
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be positive and finite")
        self.g, self.delta = g, delta
        nonlinearity = None
        if h is not None:
            if power != 2.0:
                raise ValueError("give either power or h, not both")
            nonlinearity = parse(h)
            if isinstance(nonlinearity, Power):
                power, nonlinearity = nonlinearity.p, None
        self.general = nonlinearity is not None
        if self.general:
            self.law = GeneralLaw(nonlinearity)
            self.nonlinearity, self.power = self.law.h, None
            self.sigma_left, self.sigma_right = g.absg_left, g.absg_right
            return
        self.law = PowerLaw(power)
        self.power = self.law.p
        self.nonlinearity = Power(self.power)
        # σ = |g_Δ|^q at the cell ends (nan at infinite ends, as for |g| itself).
        if self.law.quadratic:
            self.sigma_left, self.sigma_right = g.sigma_left, g.sigma_right
        else:
            self.sigma_left = self.law.sigma_of(g.absg_left)
            self.sigma_right = self.law.sigma_of(g.absg_right)
            self.sigma_left.flags.writeable = self.sigma_right.flags.writeable = False

    # ------------------------------------------------------------- level maps
    def theta(self, k):
        """Flux level value θ = kδ."""
        return np.asarray(k) * self.delta if np.ndim(k) else k * self.delta

    def s(self, k):
        """Signed root s = sgn(θ) |θ|^{1/p} (scalar or array); sqrt for p = 2. Power laws only."""
        if self.general:
            raise TypeError("s(θ) exists only for h = |u|^p; use u(k, x) or law.state")
        p = self.power
        if np.ndim(k) == 0:
            if not k:
                return 0.0
            magnitude = abs(k) * self.delta
            return math.copysign(math.sqrt(magnitude) if p == 2.0 else magnitude ** (1.0 / p), k)
        k = np.asarray(k, dtype=float)
        magnitude = np.abs(k) * self.delta
        return np.sign(k) * (np.sqrt(magnitude) if p == 2.0 else np.power(magnitude, 1.0 / p))

    def kappa(self, kL, kR):
        """Front coefficient K, with ẋ = |g|^{1/p} K: the secant (|θ_R| - |θ_L|)/(s_R - s_L).

        For p = 2 this is s(θ_L) + s(θ_R). For levels of one sign it is
        ±1/D, where D is the divided difference of y^{1/p} at |θ_L|, |θ_R|,
        evaluated without cancellation (fan steps have nearby levels).
        """
        if self.general:
            return self.law.coef(int(kL) * self.delta, int(kR) * self.delta)
        if self.law.quadratic:
            return self.s(kL) + self.s(kR)
        kL, kR = int(kL), int(kR)
        if kL == kR:
            return self.law.characteristic_kappa(kL * self.delta)
        if kL * kR >= 0:
            a, b = abs(kL) * self.delta, abs(kR) * self.delta
            sign = 1.0 if kL + kR > 0 else -1.0
            return sign / divided_power(a, b, 1.0 / self.power)
        return (abs(kR) - abs(kL)) * self.delta / (self.s(kR) - self.s(kL))

    def sigma(self, c, x):
        """σ = |g_Δ(x)|^q for x in cell c (scalar); sqrt|g_Δ| for p = 2; |g_Δ| for a general h."""
        if self.general:
            return self.g.abs_g(c, x)
        if self.law.quadratic:
            return self.g.sigma(c, x)
        return self.law.sigma_of(self.g.abs_g(c, x))

    def sigma_array(self, x, cells):
        """Vectorised σ for coordinates x already assigned to cells."""
        if self.general:
            return self.g.abs_g_array(x, cells)
        if self.law.quadratic:
            return self.g.sigma_array(x, cells)
        return self.law.sigma_of(self.g.abs_g_array(x, cells))

    def level(self, theta_values):
        """Nearest integer levels for flux-level values θ."""
        values = np.asarray(theta_values, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = np.rint(values / self.delta)
        levels = integer_levels(rounded.reshape(-1)).reshape(values.shape)
        return levels

    def theta_of(self, x, u):
        """θ = sgn(u) g_Δ(x) h(u) at coordinates x."""
        u = np.asarray(u, dtype=float)
        if self.general:
            return np.sign(u) * self.g(x) * self.nonlinearity.h(np.abs(np.atleast_1d(u))).reshape(u.shape)
        if self.power == 2.0:
            return np.sign(u) * self.g(x) * u * u
        return np.sign(u) * self.g(x) * np.abs(u) ** self.power

    # ----------------------------------------------------------- state values
    def u(self, k, x):
        """u of level(s) k at x; at zeros use the right-hand trace sign.

        A nonzero level diverges there. Choosing the right trace agrees with
        the tracker's right-continuous level sampling; it is not a finite
        pointwise solution value.
        """
        x = np.asarray(x, dtype=float)
        k = np.broadcast_to(np.asarray(k), x.shape)
        gx = self.g(x)
        if self.general:
            s = np.sign(k).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.abs(k) * self.delta / np.abs(gx)
            magnitude = self.nonlinearity.inverse_array(np.where(np.isfinite(y), y, 0.0))
            out = np.where(s == 0, 0.0, np.sign(gx) * s * magnitude)
        else:
            s = self.s(k)
            root = np.sqrt(np.abs(gx)) if self.power == 2.0 else np.abs(gx) ** (1.0 / self.power)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(s == 0, 0.0, np.sign(gx) * s / root)
        right_sign = self.g.sign[self.g.cell_of(x)]
        out = np.where((gx == 0) & (s != 0), np.copysign(np.inf, right_sign * s), out)
        return float(out) if out.ndim == 0 else out

    def flux_value(self, k, x):
        """f(x, u) = sgn(g(x)) |θ| for level(s) k at x."""
        return np.sign(self.g(x)) * np.abs(self.theta(k))

    def characteristic_speed(self, k, x):
        """f_u = |g|^{1/p} p sgn(θ)|θ|^{1-1/p} for level(s) k at x (2 sqrt|g| s for p = 2).

        For a general h, f_u = |g| h'(H(|θ|/|g|)) sgn θ.
        """
        if self.general:
            theta = np.asarray(self.theta(k), dtype=float)
            absg = np.abs(self.g(x))
            theta, absg = np.broadcast_arrays(theta, absg)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = self.nonlinearity.inverse_array(np.abs(theta) / absg)
            out = np.sign(theta) * absg * self.nonlinearity.dh(np.atleast_1d(u)).reshape(u.shape)
            return float(out) if out.ndim == 0 else out
        if self.power == 2.0:
            return 2.0 * np.sqrt(np.abs(self.g(x))) * self.s(k)
        theta = np.asarray(self.theta(k), dtype=float)
        return (np.abs(self.g(x)) ** (1.0 / self.power) * self.power
                * np.sign(theta) * np.abs(theta) ** self.law.q)

    def __repr__(self):
        if self.general:
            return f"MultiplicativeFlux(g={self.g!r}, delta={self.delta}, h={str(self.nonlinearity)!r})"
        power = "" if self.power == 2.0 else f", power={self.power:g}"
        return f"MultiplicativeFlux(g={self.g!r}, delta={self.delta}{power})"

    def characteristic(self, x0, theta, t_final, samples=32):
        """Exact genuine characteristic from x0 carrying flux level θ (any real value).

        It follows the closed-form law with K = h'(s(θ)) through every cell. It
        stops at a zero of g, where the characteristic enters the interface.
        Returns arrays (x, t).
        """
        law = self.law
        if not all(math.isfinite(v) for v in (x0, theta, t_final)) or t_final < 0:
            raise ValueError("x0, theta and t_final must be finite, with t_final >= 0")
        samples = positive_count(samples, "samples")
        g = self.g
        x0 = float(g.wrap(x0))
        kappa = law.characteristic_kappa(theta)
        d = law.direction(kappa)
        c = g.cell_of(x0, side="left" if d < 0 else "right")
        if g.periodic and d < 0 and x0 == g.origin:
            x0 = g.origin + g.period
        x, t = float(x0), 0.0
        sig = self.sigma(c, x)
        xs, ts = [x], [t]
        while t < t_final:
            if d == 0:
                xs.append(x)
                ts.append(t_final)
                break
            if d > 0:
                target, s_t, zero = g.right[c], self.sigma_right[c], g.zero_right[c]
            else:
                target, s_t, zero = g.left[c], self.sigma_left[c], g.zero_left[c]
            tau = law.travel_time(x, sig, float(target), float(s_t), kappa) if math.isfinite(target) else math.inf
            end = min(t + tau, t_final)
            tt = np.linspace(t, end, max(2, samples))
            xs.extend(law.position(x, sig, float(g.alpha[c]), kappa, tt[1:] - t))
            ts.extend(tt[1:])
            if end >= t_final or zero:
                break
            c = g.next_cell(c, d)
            x, t = float(target), end
            if g.periodic and d > 0 and c == 0:
                x = float(g.left[0])
            elif g.periodic and d < 0 and c == g.n_cells - 1:
                x = float(g.right[c])
            sig = float(s_t)
        return np.asarray(xs, dtype=float), np.asarray(ts, dtype=float)
