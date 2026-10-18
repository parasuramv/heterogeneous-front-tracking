"""Moving fronts with closed-form laws, and stationary interfaces at zeros of g.

A ``Front`` keeps the anchor (t0, x0, σ0) of its current affine cell; its
position at any later time in that cell is its ``law.position`` (the
quadratic ``trajectory.position`` unless h(u) = |u|^p with p ≠ 2). The
anchor changes only when the law changes (corner crossing), so positions do
not accumulate time-stepping error. An ``Interface`` sits at a zero of g_Δ,
never moves, and records the mass it absorbs.
"""

from dataclasses import dataclass, field
import math

from .trajectory import QUADRATIC


@dataclass(slots=True, eq=False)
class Front:
    """Jump from level kL to kR moving with ẋ = |g|^{1/p} κ inside ``cell`` (σκ for p = 2)."""

    kL: int
    kR: int
    kappa: float
    cell: int
    t0: float
    x0: float
    s0: float
    alpha: float
    # Cached absolute event times, maintained by the tracker.
    t_own: float = field(default=math.inf, repr=False)
    own_kind: str = field(default="", repr=False)
    t_pair: float = field(default=math.inf, repr=False)
    version: int = field(default=0, repr=False)  # invalidates queued events
    law: object = field(default=QUADRATIC, repr=False)

    def x(self, t):
        return self.law.position(self.x0, self.s0, self.alpha, self.kappa, t - self.t0)

    def sigma(self, t):
        return self.law.sigma(self.s0, self.alpha, self.kappa, t - self.t0)

    @property
    def direction(self):
        return (self.kappa > 0) - (self.kappa < 0)


@dataclass(slots=True, eq=False)
class Interface:
    """Stationary jump at a zero of g_Δ; fronts are absorbed, never cross.

    ``left_sign``/``right_sign`` are the signs of g on the adjacent cells.
    ``absorbed`` integrates f(z-) - f(z+), the net mass flux into the zero.
    """

    x: float
    kL: int
    kR: int
    left_sign: float
    right_sign: float
    left_cell: int
    right_cell: int
    last_time: float = 0.0
    absorbed: float = 0.0
    t_pair: float = field(default=math.inf, repr=False)
    t_own: float = field(default=math.inf, repr=False)
    own_kind: str = field(default="", repr=False)
    version: int = field(default=0, repr=False)

    def rate(self, delta):
        """Net mass flux into the zero: f(z-) - f(z+)."""
        return delta * (self.left_sign * abs(self.kL) - self.right_sign * abs(self.kR))

    def accrue(self, t, delta):
        """Integrate the absorption rate up to time t (levels constant since last_time)."""
        self.absorbed += self.rate(delta) * (t - self.last_time)
        self.last_time = t

    def absorbed_until(self, t, delta):
        return self.absorbed + self.rate(delta) * (t - self.last_time)

    def is_admissible(self):
        """Lax-type interface condition: θ ≥ 0 left of the zero, θ ≤ 0 right of it."""
        return self.kL >= 0 and self.kR <= 0
