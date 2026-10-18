"""Generalised Riemann problem between two stationary profiles.

States are integer flux levels k (θ = kδ). Because sgn θ = sgn f_u, one rule
covers both signs of g: θ_L > θ_R is an entropic shock (HetFT Theorem 2.3),
and θ_L < θ_R is a δ-fan of unit steps (HetFT Definition 1). Fan fronts have
κ_j = s_j + s_{j+1}, increasing in j, so they leave in spatial order.
"""


def solve_riemann(kL, kR):
    """Return the (left level, right level) pairs of the outgoing fronts, left to right."""
    kL, kR = int(kL), int(kR)
    if kL == kR:
        return []
    if kL > kR:
        return [(kL, kR)]
    return [(k, k + 1) for k in range(kL, kR)]


def is_shock(kL, kR):
    """True for an entropy-admissible (decreasing-level) jump."""
    return kL > kR
