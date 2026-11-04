> **Status (phases 1–3 done):** `MultiplicativeFlux(g, δ, h="u^2+u^4")` is public. Fronts follow `GeneralLaw`, cell integrals use the closed form P(θ,γ) = γQ(H(θ/γ)), and `flux.characteristic`, plotting, `main.py` output and `homogenisation.effective_flux_nodes` handle a general h. `PeriodicStraightening` rejects one. The Osgood primitive is factored, J = (u/h)·R, with R tabulated in log u and exact asymptotic tails. It matches 300-digit closed forms to 8e-16 for u ∈ [1e-30, 1e30]. Correction: the phase-1 exp-sinh quadrature was off by up to 4e-6 at small u for mixed powers, and is replaced. Power-law runs stay bit-identical. Speed is about 2–3 ms per event for u²+u⁴ (about 40× PowerLaw, 12× faster than phase 2). Open: phase 4 (CLI `--flux`, examples, convergence runs, derivation and README, notebooks) and phase 5 (asymmetric h).

# Plan: general convex multiplicative fluxes `g(x) h(u)`

## Context

The repo solves `u_t + (g(x)|u|^p)_x = 0` with `g` piecewise affine. Everything
rests on `PowerLaw` (`src/hetfront/trajectory.py`): in `σ = |g_Δ|^{1-1/p}` every
front is a straight line, so every event time comes from a *linear* equation.
The question was whether this needs `h = |u|^p`, or whether it extends to
convex `h` in general (`u²+u⁴`, `|u|³+u²`, …), the class the *Failure of
uniqueness* setting has in view.

**Answer (I checked the key identity against RK4 to 1e-15 for `h=u²+u⁴`):**

1. **The straight-line structure belongs to power laws only (no-go).** For
   one coordinate `σ(γ)`, `γ=|g_Δ|`, to straighten every characteristic, the
   speed `γ h'(H(θ/γ))` must factor as `a(θ)b(γ)`. That is Cauchy's
   multiplicative equation, so `h'(u) = c h(u)^r`. With `h(0)=0` this forces
   `h = c|u|^p`. Numerically, for `u²+u⁴` the ratio of the speeds for
   θ=1 and θ=4 is 0.339, 0.341, 0.366 at γ = 0.5, 1, 4, where a power law
   would give a constant. Polynomials do **not** "directly" generalise the
   σ-trick. The §5 conjugacy to homogeneous front tracking and the §9 exact
   homogenisation map also fail, because `U[θ](x) = ηH(θ/|g|)` is no longer
   `s(θ)·ρ(x)`.
2. **The method still works, with one closed form in place of linearity.**
   Let `H = h⁻¹` (odd branches) and let `J(u) = ∫_u^∞ dr/h(r)` be the
   "Osgood primitive". Define
   `P(θ,γ) = γH(θ/γ) + θJ(H(θ/γ))`. Then `∂_γP = H(θ/γ)` (the stationary state),
   `∂_θP = J(u)`, and `∂_θ∂_γP = 1/(γh'(u))` (characteristic slowness). On an affine cell:
   - **front** (θ_L,θ_R): `α[|θ|](t−t₀) = Λ(γ)−Λ(γ₀)`, `Λ = P(θ_R,·)−P(θ_L,·)`.
     This is closed form in time-of-arrival. A front's slowness is the
     θ-average of the characteristic slowness over the levels it spans.
   - **characteristic**: `α(t−t₀) = J(u₁) − J(u₀)`.
   - **mass on a cell**: `∫U[θ]dx = η(P(θ,γ₂)−P(θ,γ₁))/α`.
   - For `|u|^p`, `Λ ∝ σ`, which recovers `PowerLaw` exactly.
3. **Events.** Node, corner and absorption times stay *explicit*. Position at
   a given t, and collision times, become *monotone bracketed scalar roots*.
   For a fixed pair of levels, the order of the secant speeds at a common γ
   does not depend on γ, so `T_a(x) − T_b(x)` is strictly monotone. There
   is still no ODE solver.
4. **What carries over unchanged.** The θ shock/fan rule, "at most one
   outgoing front" (the proof becomes pointwise and combinatorial at the
   collision point, without §5's conjugacy), passive corners, the interface
   sign rule and mass accounting.
5. **The dividing line is `∫^∞ dr/h(r) < ∞`.** This one condition is
   equivalent to (a) integrable profiles at zeros, (b) finite-time absorption,
   (c) finite `J`, and (d) finite-time blow-up of `ṗ = −g'h(p)`, which is the
   manuscript's mechanism. If it fails (for example `h = u log(1+u)`, which is
   convex and superlinear), zeros become unreachable and the non-uniqueness
   picture changes qualitatively. Near zeros the **top** degree of a
   polynomial `h` governs (`U ~ |x−z|^{−1/p_max}`, local rate `Δ^{2−1/p_max}`).
   Small amplitudes are governed by the **bottom** degree.

## Hypotheses (new A1′, confirmed: Osgood-tail convex class; asymmetry deferred to phase 5)

`h ≥ 0`, `h(0)=0`, strictly convex, `C²` away from 0, and
`∫^∞ dr/h < ∞` on each side. `h` may be asymmetric (`h₊ ≠ h₋`); the branch is
chosen by `sgn u = η·sgn θ`. If `min h ≠ 0`, or `min h` sits away from 0,
the problem becomes a forced one (`m g'(x)` source). That case is rejected.

## Code changes

**New `src/hetfront/nonlinearity.py`**
- `Nonlinearity` base class, with `h`, `dh`, `inverse(y, sign)` (H),
  `osgood(u)` (J), `dd(a,b)` (a divided difference of h with no
  cancellation), and `validate()`, which checks convexity, `h(0)=0` and the
  finite tail on samples.
- `Power(p)` wraps the existing `PowerLaw`.
- `AbsPolynomial(coeffs, coeffs_minus=None)` is `Σ c_k|u|^{k}` with
  `c_k ≥ 0` and `k > 1` real. Its `dd` is computed exactly, termwise, with
  the existing `trajectory.divided_power`. `H` uses safeguarded Newton.
  `J` is a closed form for a monomial or binomial when one exists (for example
  `u²+u⁴: J = 1/u − (π/2 − atan u)`). Otherwise it uses numpy Gauss–Legendre
  after the substitution `r = u/s`. No SciPy: dependencies stay numpy-only.
- `Custom(h, dh, inverse=None, osgood=None)` uses numerical fallbacks.

**`src/hetfront/trajectory.py`: add `GeneralLaw(nonlinearity)`** with the
same interface as `PowerLaw`, and `sigma_of = identity`, so the anchor `s0`
is γ₀.
- `travel_time(x0,γ0,x1,γ1,coef)` is `(x1−x0)·S̄`. Here `S̄` is the mean
  slowness. It is a mixed divided difference of `P` and needs no division
  by α, so flat cells are handled. Use the closed form `Λ` when the
  relative separations in θ and γ exceed 1/2 (the `divided_power` rule).
  Otherwise use 8–16-point Gauss–Legendre on the smooth slowness
  `1/(γ·dd(u_L,u_R))`, which avoids cancellation on fan steps.
- `position(...,tau)` inverts the monotone `T(x)` by Illinois/Brent
  bisection, to a bracket of 2 ulp.
- `meeting_time` finds the root of the monotone `T_a(x)−T_b(x)`. It brackets
  the gap for fronts moving in opposite directions, and from the leader to the
  cell exit for fronts moving the same way. It returns `inf` if there is no
  sign change.
- `sample_path` is parametrised by x, with t explicit, so plotting needs no
  root solve.
- `characteristic_time` uses `J` directly.

**Front coefficient (`fronts.py`, `solver.py`)**: `Front.kappa` stops being
a scalar for general laws.
- Rename `Front.kappa` to an opaque law-specific `coef`. `PowerLaw` keeps the
  float `K`. `GeneralLaw` stores `(θ_L, θ_R, η-free sign data)`.
- Replace `f.kappa > 0` with `law.direction(coef)`, which is
  `sgn((|θ_R|−|θ_L|)/(θ_R−θ_L))` and constant in γ. Replace
  `a.kappa > b.kappa` with `law.closing(a,b)`. The affected sites are
  `solver.py:78–117, 130–134, 263–274, 668–682` and `visualization.py:182`.
- `Piece` keeps the field name `kappa` and the `PowerLaw` path stays
  bit-identical. `test_quadratic_law_is_the_unchanged_module_code` must still pass.

**`flux.py`**
- `MultiplicativeFlux(g, delta, power=2.0, h=None)`, with `h` and `power`
  mutually exclusive. `self.law` is `PowerLaw` or `GeneralLaw`.
- `s`, `u`, `theta_of`, `characteristic_speed` and `kappa` route through the
  law. `flux.power` becomes `None` for a general `h`. Audit every
  `flux.power` reader (`homogenisation.py`, `studies.py`, `examples.py`, tests).
- `characteristic()` uses `GeneralLaw.characteristic_*`.

**`solver.py` integrals**: `integral` and `primitive` (lines 483–495) use the
cell-mass closed form via `P`, with the same divided-difference guard.
`flux_through` needs no change (`f = η|θ|`).

**`homogenisation.py`**: `PeriodicStraightening` raises an error for a
non-power `h`, since there is no exact conjugacy. Add
`effective_flux_nodes(levels)`, which gives `v̄_k = ⟨ηH(kδ/|g|)⟩` in closed
form per cell, paired with `η|kδ|`. That is the polygonal homogenised flux,
reported as a comparison and not an identity. The O(ε) bound becomes an open
question.

**CLI/examples**: `--flux "u^2+u^4"` (a parsed `AbsPolynomial`) in `main.py`
and `convergence.py`. Add examples `poly-zeros` (`u²+u⁴`, three zeros) and
`poly-sine`, plus a playground selector in notebook 01 via
`tools/build_notebooks.py`.

**Docs**
- `docs/derivation.md`: §0 A1′, a new §3b "General convex h" (P, Λ,
  slowness-average identity, no-go proposition), §5's pointwise proof, §6
  Osgood, §9 limits.
- README: standing assumptions.

## Phasing

1. `nonlinearity.py` and `GeneralLaw`, with unit tests only. No solver change.
2. The `coef` refactor, with the `PowerLaw` path proven bit-identical (full
   suite plus `tools/stress_check.py`).
3. Wire up `MultiplicativeFlux(h=...)`, integrals and characteristic.
4. Examples, CLI, convergence runs for `u²+u⁴` and `|u|³+u²`, docs, notebooks.
5. Optional: asymmetric `h₊≠h₋`.

## Verification

- New `tests/test_general.py`:
  - `GeneralLaw` with `h=|u|^p` matches `PowerLaw` to 1e-13 for position,
    travel and meeting times.
  - Closed form against RK4 of `ẋ = [g h]/[u]` across many corners, for
    `u²+u⁴` and `|u|³+u²`. This reuses the RK4 oracle in
    `tests/test_trajectory.py` and `tests/_oracles.py`.
  - Fan steps with k≈10⁴ show no cancellation.
  - Meeting time against brute-force bisection on `x_a(t)−x_b(t)`.
  - Exact characteristic into a zero (`α t = J(u₁)−J(u₀)`).
  - Mass budget with absorption, TV and front count non-increasing,
    `emission == 0` after t=0, and L¹ contraction.
  - `validate()` rejects `u log(1+u)` (Osgood fails) and `u²+1` (min ≠ 0).
- `python -m unittest discover -s tests -t . -v`: the existing 92 tests are
  unchanged.
- `python tools/stress_check.py`, extended with general-h random cases.
- `python main.py --example poly-zeros`, plus the convergence script with
  `--flux "u^2+u^4"`. Report the measured orders as observations.
- `python tools/check_notebooks.py --write`.
