# Derivation: front tracking with piecewise-affine heterogeneity

This note records the mathematics the code implements. It builds on the
heterogeneous front tracking of Venkatesh (*Front tracking for scalar
conservation laws with spatially heterogeneous flux*, Indian J. Pure Appl.
Math., 2026; "HetFT" below) and on the *Failure of uniqueness* manuscript.
Statements marked **observed** are numerical observations from this
repository, not theorems.

## 0. Standing assumptions

We solve

$$
u_t + \big(g(x)\,h(u)\big)_x = 0,\qquad h(u) = |u|^p,\quad p>1 ,
$$

on the line or on a circle $\mathbb R/L\mathbb Z$.

* **(A1) Power-law $h$.** $h(u)=|u|^p$ with $p>1$; the default is $p=2$,
  $h(u)=u^2$. §§2–6 are written for $p=2$, which is the case of HetFT and the
  manuscript. §3 ends with the general-$p$ laws (`MultiplicativeFlux(...,
  power=p)`), and §§5–6 and §9 note what changes. The $p=2$ code path is the
  original quadratic one, unchanged.
* **(A2) Transverse zeros.** $g$ is Lipschitz with finitely many zeros on
  compact sets, and at each zero $z$, $|g'(z\pm)| \ge \beta > 0$. Tangential
  zeros (for example $g = x^2$ or $x^3$), when detected, are rejected by
  `PiecewiseAffine.from_function`. Nodal interpolation would quietly turn a
  zero of order $\eta>1$ into a simple zero. The sharpness results of the
  manuscript (§ *Heterogeneity with vanishing derivative*) depend on $\eta$,
  so this case needs separate treatment. Finite sampling cannot certify all
  zeros of an arbitrary callable. The root finder rejects detected tangencies
  and minima unresolved at floating-point scale, and requires a mesh resolving
  the coefficient's oscillations.
* **Well-signed versus sign-changing.** If $|g|\ge g_{\min}>0$ there are no
  interfaces: HetFT applies directly, $\theta$ is bounded, and mass is
  conserved exactly. This is the easier case. When $g$ changes sign, the zeros
  become absorbing interfaces, $u$ is unbounded but integrable, and mass may
  vanish into, or emerge from, the zeros.
* **Convergence** as $\delta,\Delta\to0$ is studied empirically
  (`convergence.py`). The expected route to a proof is uniform approximation of
  $g$ by smooth heterogeneities. HetFT assumes $g\in C^2$, whereas $g_\Delta$ is
  only Lipschitz.

## 1. Discretising the derivative of $g$

On a mesh $x_0<\dots<x_{M}$ set $g'_\Delta$ equal to the cell average of $g'$:

$$
g'_\Delta\big|_{(x_j,x_{j+1})} = \frac{g(x_{j+1})-g(x_j)}{x_{j+1}-x_j}.
$$

Then $g_\Delta(x) = g(x_0)+\int_{x_0}^x g'_\Delta$ is the continuous nodal
interpolant, $g_\Delta = a_j x + b_j$ on cell $j$. The true zeros of $g$ are
detected by sampling, located by bisection, and inserted as nodes, so $g_\Delta$ has one sign on each
open cell. The flux $f_\Delta(x,u) = g_\Delta(x)u^2$ is piecewise affine in $x$.

## 2. Flux levels and stationary profiles

As in HetFT, the quantity that is discretised is the flux-value function

$$
\theta = G(x,u) = \mathrm{sgn}(u)\, f(x,u),
$$

not the flux itself. On each open sign interval, stationary solutions are the
profiles with constant $\theta$. Extending one level through a zero need not
satisfy the selected interface rule. Writing $\sigma(x) = \sqrt{|g_\Delta(x)|}$ and
$s(\theta) = \mathrm{sgn}(\theta)\sqrt{|\theta|}$,

$$
u = U[\theta](x) = \mathrm{sgn}(g)\,\frac{s(\theta)}{\sigma(x)},\qquad
f = \mathrm{sgn}(g)\,|\theta|,\qquad
f_u = 2\sigma s .
$$

So **$\mathrm{sgn}\,\theta$ is the direction of the characteristics**, on
either sign of $g$. Near a zero $z$, $U[\theta]\sim|x-z|^{-1/2}$: unbounded but
integrable. Levels are quantised, $\theta = k\delta$ with $k\in\mathbb Z$.

For $h=|u|^p$ the same holds with $s(\theta)=\mathrm{sgn}(\theta)|\theta|^{1/p}$:

$$
U[\theta](x) = \mathrm{sgn}(g)\,\frac{s(\theta)}{|g_\Delta(x)|^{1/p}},\qquad
f = \mathrm{sgn}(g)\,|\theta|,\qquad
f_u = |g_\Delta|^{1/p}\,p\,\mathrm{sgn}(\theta)|\theta|^{1-1/p},
$$

so $U[\theta]\sim|x-z|^{-1/p}$ near a zero, still integrable since $p>1$.

## 3. The xu² mechanism, cell by cell

The characteristic system is $\dot q = g(q)h'(p)$, $\dot p = -g'(q)h(p)$. On a
cell $g' \equiv a_j$, so the $p$-equation $\dot p = -a_j p^2$ decouples:

$$
p(t) = \frac{p_0}{1+a_j p_0 t},\qquad g(q(t)) = g(q_0)\,(1+a_jp_0t)^2 .
$$

This is the computation the manuscript does for $g(x)=x$, now on every cell.

**Straightening variables.** Set $w = \sigma u$, which equals
$\pm\sqrt{|\theta|}$, so stationary states are constant in $w$. On a cell with
$g_\Delta>0$, using $\sigma_x = a/(2\sigma)$,

$$
u_t + (g u^2)_x = 0 \iff w_t + \tfrac{a}{2}\,(w^2)_\sigma = 0 ,
$$

and the same form holds when $g_\Delta<0$. In $(\sigma,t)$ each cell carries
Burgers' equation when $a\ne0$, so **fronts and characteristics are straight
lines in $\sigma$**. For $a=0$, sigma is constant and is not a spatial
coordinate; the original x-coordinate gives constant-speed motion. The front
formulas below cover this case without a change of variables.

**Front law.** A front joining levels $\theta_L,\theta_R$ moves at the
Rankine–Hugoniot speed $\dot x = g(u_L+u_R) = \sigma\kappa$ with
$\kappa = s(\theta_L)+s(\theta_R)$. Let $\alpha = \mathrm d|g_\Delta|/\mathrm dx$
on the cell. Then $\dot\sigma = \alpha\kappa/2$ is constant, and

$$
\sigma(\tau) = \sigma_0 + \tfrac12\alpha\kappa\tau,\qquad
x(\tau) = x_0 + \kappa\tau\big(\sigma_0 + \tfrac14\alpha\kappa\tau\big).
$$

A characteristic carrying level $\theta$ is the same curve with $\kappa = 2s(\theta)$.

*Check:* for $g=x$ with $\theta_L=0$, $\theta_R=-1$ and $x_0 = 1$, this gives
$x(t) = (2-t)^2/4$. This is the shock of the manuscript's unbounded-solution
example (HetFT §4.1); it reaches the zero at $t = 2$.

**Event times are roots of linear equations.** Because $|g_\Delta|$ is affine,
$\sigma_2-\sigma_1 = \alpha(x_2-x_1)/(\sigma_1+\sigma_2)$. This gives forms with
no division by $\alpha$ (so flat cells need no special case):

| event | waiting time |
| --- | --- |
| front reaches node $x_n$ | $\tau = 2(x_n-x_0)\,/\,\big(\kappa(\sigma_0+\sigma_n)\big)$ |
| fronts 1 < 2 in one cell meet | $\tau = 2(x_2-x_1)\,/\,\big((\sigma_1+\sigma_2)(\kappa_1-\kappa_2)\big)$, if $\kappa_1>\kappa_2$ |

No ODE solver is used anywhere; `tests/test_trajectory.py` checks the closed
forms against RK4 integration of the Rankine–Hugoniot ODE across many corners.

### General powers

Write $q=1-1/p$ and $m=p/(p-1)$, so $qm=1$. With $U[\theta]$ as in §2, the
Rankine–Hugoniot speed of a front joining $\theta_L\ne\theta_R$ is

$$
\dot x = \frac{[f]}{[u]} = |g_\Delta|^{1/p}\,K,\qquad
K = \frac{|\theta_R|-|\theta_L|}{s(\theta_R)-s(\theta_L)} ,
$$

the secant slope of $r\mapsto|r|^p$ between $s(\theta_L)$ and $s(\theta_R)$.
For $p=2$, $K=\kappa=s_L+s_R$. Set $\sigma=|g_\Delta|^{q}$. On a cell,
$\dot\sigma = q|g|^{q-1}\alpha\dot x = q\alpha K$ is constant, so **fronts are
straight in $\sigma$** for every $p$. A characteristic carrying $\theta$ is the
same curve with $K=h'(s(\theta))=p\,\mathrm{sgn}(\theta)|\theta|^{q}$.

Because $|g_\Delta|=\sigma^m$ is affine, $\alpha(x_1-x_0)=\sigma_1^m-\sigma_0^m$.
Every law then uses one two-point mean of the speed factor $|g|^{1/p}$,

$$
M(\sigma_0,\sigma_1)=q\,\frac{\sigma_1^m-\sigma_0^m}{\sigma_1-\sigma_0}
=\frac{x_1-x_0}{K\tau},\qquad M(\sigma,\sigma)=\sigma^{m-1}=|g|^{1/p},
$$

which is $(\sigma_0+\sigma_1)/2$ when $p=2$. None of the laws divides by $\alpha$:

| quantity | general $p$ | $p=2$ |
| --- | --- | --- |
| position | $x(\tau)=x_0+K\tau\,M(\sigma_0,\sigma(\tau))$, $\sigma(\tau)=\sigma_0+q\alpha K\tau$ | $x_0+\kappa\tau(\sigma_0+\tfrac14\alpha\kappa\tau)$ |
| front reaches node $x_n$ | $\tau=(x_n-x_0)/\big(K\,M(\sigma_0,\sigma_n)\big)$ | as above |
| fronts 1 < 2 meet | $\tau=(x_2-x_1)/\big(M(\sigma_1,\sigma_2)(K_1-K_2)\big)$, if $K_1>K_2$ | as above |
| $\int_{x_1}^{x_2}\lvert g_\Delta\rvert^{-1/p}dx$ | $(x_2-x_1)/M(\sigma_1,\sigma_2)$ | $2(x_2-x_1)/(\sigma_1+\sigma_2)$ |

The meeting time holds because $\sigma$ is monotone in $x$ on a cell with
$\alpha\ne0$, and both fronts are linear in $\sigma$. On a flat cell both
sides reduce to constant speeds $|g|^{1/p}K$. `trajectory.PowerLaw` evaluates
$M$ as a divided difference $\sigma_0^m\,\mathrm{expm1}\big(m\log1p((\sigma_1-\sigma_0)/\sigma_0)\big)/(\sigma_1-\sigma_0)$
when the arguments are close, so it does not cancel. For fan steps $K$ is
computed in the same way, as $\pm1/D$ with $D$ the divided difference of
$y^{1/p}$ at $|\theta_L|,|\theta_R|$.

*Check:* for $g=x$, $\theta_L=0$, $\theta_R=-1$ and $x_0=1$: $K=-1$,
$\sigma=x^q=1-qt$, so $x(t)=(1-qt)^m$, which reaches the zero at $t=p/(p-1)$.
For $p=2$ this is $(2-t)^2/4$ and $t=2$.

These laws are elementary consequences of the Rankine–Hugoniot condition
(**proved** above). `tests/test_power.py` checks them against RK4 integration
of the physical ODE $\dot x=[g|u|^p]/[u]$ across many corners, against the
exact solution just given, and, for constant $g$, against `fronttrack` with
the flux $b|u|^p$. For a general convex $h$, the front ODE
$\dot x = F(g(x))$ on an affine cell still reduces to the quadrature
$\mathrm dt = \mathrm dg/(aF(g))$, but it is no longer linear in a power of $g$.

## 4. Corners are passive

At a node where $g_\Delta\ne0$ but $g'_\Delta$ jumps, $U[\theta]$ depends on
$x$ only through $g_\Delta$, so it is continuous. The front speed $\sigma\kappa$
is continuous as well. A front crossing a corner therefore creates no waves: it
only switches from $(a_j,b_j)$ to $(a_{j+1},b_{j+1})$, with its anchor
$(t, x_n, \sigma_n)$ reset. Continuity of the profile and its flux is the
reason no stationary jump must be resolved. The local sigma-coordinate may
reverse orientation at a corner or degenerate on a flat cell, so it must not
be treated as one global coordinate. This is the advantage over piecewise-constant $g$ (Sylla's scheme;
discontinuous-flux theory), where every node is an interface with its own
Riemann solver.

## 5. Riemann problems and interactions

$G$ is increasing in $u$ where $g>0$ and decreasing where $g<0$. Together with
convexity (respectively concavity), this gives **one rule in $\theta$ for both
signs of $g$**:

* $\theta_L > \theta_R$: a single entropic shock (HetFT Thm 2.3);
* $\theta_L < \theta_R$: a δ-fan of unit level steps (HetFT Def. 1). Fan
  fronts have $\kappa_j = s_j+s_{j+1}$, increasing in $j$, so they separate.

At a collision the whole cluster is replaced by the Riemann solution between
its two exterior levels. With levels on $\delta\mathbb Z$ this is at most one
front (zero if the exterior levels agree) (HetFT Thm 2.4), and the code counts any exception as
`multi_front_collision`; none has been seen. Each front keeps the direction
$\mathrm{sgn}\,\kappa$ throughout its life, so on the line it crosses each
node at most once. Starting from $N$ fronts there are at most $N-1$ collisions
and at most $2N-1$ fronts ever exist, so there are at most $(2N-1)M$ corner
events. On a circle, fronts keep circulating, so the number of events is finite
on every bounded time interval.

*General $p$.* HetFT assumes $f(x,\cdot)$ uniformly convex, which $|u|^p$ lacks
globally for $p\ne2$: $f_{uu}$ degenerates at $u=0$ when $p>2$ and as $|u|\to\infty$ when
$p<2$. The discrete rule does not need HetFT. On a sign interval,
$X=\int|g_\Delta|^{-1/p}\,dx$ is an increasing $C^1$ coordinate, and by §3 every
front moves with $\dot X = K$, the secant speed of $\eta|r|^p$ ($\eta=\mathrm{sgn}\,g$)
between its states $r=s(\theta)$. So in $(X,t)$ the scheme is exactly
homogeneous Dafermos front tracking for the polygonal interpolant of $\eta|r|^p$
on the nodes $s(k\delta)$, with levels unchanged. That polygon is convex for
$\eta=1$ and concave for $\eta=-1$, so the shock/fan rule and "at most one
outgoing front" follow as in the homogeneous theory (**proved** for the
discrete scheme). The tests check this conjugacy on the circle for
$p\in\{3/2,3\}$ (via §9's map, which is $X$ rescaled). Convergence of the
scheme to an entropy solution for $p\ne2$ is **not proved** here; numerical
refinement studies are reported in §8.

## 6. Zeros of $g$: absorbing interfaces

As a front approaches a zero, $\sigma\to0$ and its speed $\sigma\kappa\to0$.
Yet $\dot\sigma$ is constant, so the zero is reached **in finite time**, and the
front is absorbed into a stationary interface. The same holds for every
$p>1$, with $\sigma=|g|^{1-1/p}$ and speed $|g|^{1/p}K$. The interface
condition, the interface flux $f(z^\pm)=\mathrm{sgn}(g_\pm)|\theta_\pm|$ and the
limiting entropy flux $\theta$ below do not depend on $p$. The manuscript's Lax-type
interface condition becomes a sign condition in $\theta$, valid for both
signs of $g$:

$$
\theta(z^-) \ge 0 ,\qquad \theta(z^+) \le 0 .
$$

That is, characteristics enter the zero and never leave it. Fronts never cross
a zero, so each sign interval evolves independently.

* *Initial data.* The manuscript sets $\theta^\delta = 0$ on an
  $\varepsilon(\delta)$-neighbourhood of each zero (`interface_layer=ε`). The
  default $\varepsilon = 0$ uses the limit of that construction: an
  inadmissible level next to a zero emits a δ-fan from the zero, treating the
  interface as level 0.
* *Absorption keeps admissibility.* Suppose a front arrives from the right with
  $\kappa<0$, while the level beside the zero is $k_L\le0$. If its right level
  were $k_R>0$, the front would be a rarefaction-type jump of at least two
  levels. By Thm 2.4 that cannot occur, so $k_R \le 0$. The regression tests check this:
  `counts["emission"]` stays 0 after $t=0$ in every run so far.
* *Mass.* Entropy solutions here need not be weak solutions. Each interface
  accrues the net flux $f(z^-)-f(z^+) = \mathrm{sgn}(g_L)|\theta_L| -
  \mathrm{sgn}(g_R)|\theta_R|$. Then $\int u + \text{absorbed}$ (minus
  window inflow) is conserved to roundoff. The inflow at a fixed edge $x$ is
  integrated exactly: the level there is the left level plus the jumps of all
  items left of $x$, so it changes only when a front crosses $x$, at the
  closed-form `travel_time` of its piece. This uses no mass integral, so the
  budget is a check rather than an identity.
* *Non-selected solutions.* `enforce_interface=False` disables emission; it
  does not enforce entropy for arbitrary traces. At a transverse zero the
  limiting Kružkov entropy flux for any finite comparison state is $\theta$,
  so its jump must satisfy $\theta_R-\theta_L\le0$. The selected sign rule
  implies this inequality, but the converse fails. Balanced traces
  $\theta_L=\theta_R=1$ in notebook 06 give the non-selected family in the
  vanishing-seed limit. A seed of width $r>0$ has mass $2\sqrt r$ and shock
  $(\sqrt r+(t-\lambda)/2)^2$, rather than exactly $(t-\lambda)^2/4$.

Here “entropy solution” follows the manuscript's unbounded locally integrable
class satisfying the Kružkov inequalities, without separately requiring the
weak conservation law. Across a zero the distributional conservation-law
residual is $(f(z^+)-f(z^-))\delta_z$. Interface accounting records its
negative integral. Standard bounded entropy solutions satisfy weak
conservation; these two classes must not be conflated.

## 7. The circle

$g$ is $L$-periodic and $g_\Delta$ its periodic interpolant. Following
`fronttrack.periodic`, items are kept in cyclic order on the chart
$[n_0, n_0+L)$. The seam is a node, so a front crosses it through an ordinary
corner event, followed by a rotation of the list. Cluster detection uses the
lift $x(\text{item}_k) + \lfloor k/n\rfloor L$, so that a collision covering
the whole circle keeps the correct exterior level. A well-signed $g$ gives
genuine wraparound with exact mass conservation. Zeros split the circle into
independent sign arcs, one of which may cross the seam.

## 8. Observed convergence

`python convergence.py` uses initial levels that are multiples of $0.1$, which
are exact for every $\delta = 0.1\cdot2^{-k}$. It measures errors against
a fine numerical reference: $\|\theta-\theta_{\rm ref}\|_{L^1}$ and an 801-point sampled maximum of the
primitive of $u-u_{\rm ref}$. Each primitive value is exact up to roundoff;
the maximum is not a certified continuum supremum. **Observed** orders:

| problem | δ alone | Δ alone | both |
| --- | --- | --- | --- |
| line, $g = 2+\sin 3x$ | 1.01 (1.00) | 2.00 (1.98) | 1.06 (1.26) |
| line, $g = x(1-x^2)$, 3 zeros | 1.00 (1.00) | 1.46 (1.44) | 1.03 (1.24) |
| circle, $g = 2+\sin 2\pi x$ | 1.00 (2.00) | 2.01 (2.02) | 1.03 (1.94) |
| circle, $g = \sin 2\pi x + 0.3$, 2 zeros | 0.96 (0.91) | 1.37 (1.36) | 0.96 (0.92) |

Entries are fitted orders for the θ-L¹ error, with the primitive error in
parentheses (five refinement levels; reference two halvings finer), rerun on
2026-10-09 after the line mesh-width correction.

These runs suggest first-order delta behaviour and approximately second-order
mesh behaviour away from zeros. Near transverse zeros the measured mesh rate
is about 3/2. A possible local explanation, assuming one-sided $C^2$ regularity
and an exactly inserted zero, is $|g_\Delta-g|=O(\Delta|x-z|)$ in the adjacent
cell. Since $|g|\asymp|x-z|$, the relative coefficient error is $O(\Delta)$,
and integrating the resulting $O(\Delta)|x-z|^{-1/2}$ profile error over that
cell gives $O(\Delta^{3/2})$. This local calculation does not prove a global
dynamic rate. Line mesh widths use span/(nodes-1), not span/nodes.

The table above uses $h=u^2$. Five-level experiments for $h=|u|^p$ with
$p=3/2,3$, including finer spatial-reference and primitive-sampling checks,
are recorded in [the general-power report](convergence-2026-10-15.md).
The local interpolation argument suggests $O(\Delta^{2-1/p})$ beside a zero,
from integrating $O(\Delta)|x-z|^{-1/p}$ over one cell. This remains a local
heuristic, not a global dynamic rate: measured slopes depend on the problem,
metric, resolution range and numerical reference.

## 9. Homogenisation and the exact coordinate map

This section is written for $h=u^2$; here $p$ denotes the mean state, not the
exponent. For $h=|u|^{\bar p}$ (exponent $\bar p$), replace $|g|^{-1/2}$ by
$|g|^{-1/\bar p}$ throughout: $m=\langle|g|^{-1/\bar p}\rangle$,
$\rho=1/(m|g|^{1/\bar p})$ and $c=\eta m^{-\bar p}$, with effective flux
$c|v|^{\bar p}$. The substitution $u=\rho\,v\circ z$ again leaves the levels
unchanged, since $\mathrm{sgn}(u)g|u|^{\bar p}=\mathrm{sgn}(v)c|v|^{\bar p}$.
`PeriodicStraightening(g, power=p̄)` implements this.

Let $g$ have fixed sign $\eta$ and be periodic with $|g|\ge g_{\min}>0$.
Define normalised averages and the positive stationary corrector by

$$m=\langle |g|^{-1/2}\rangle,\qquad c=\eta m^{-2},\qquad
\rho(x)=\frac{1}{m\sqrt{|g(x)|}},\qquad z(x)=\int_0^x\rho(\xi)\,d\xi.$$

The mean of $U[\theta]$ is $p=\eta m s(\theta)$; its flux is
$\eta|\theta|=cp^2$. Thus the signed effective flux is $\bar F(p)=cp^2$.
For positive $g$, $F_{uu}=2g$, so
$\bar F''=2/m^2=(\langle F_{uu}^{-1/2}\rangle)^{-2}$. This repository uses
$u^2$, whereas the quadratic section of the neighbouring draft uses $u^2/2$.
There is no missing factor of two once that convention is respected.

The substitution $u(x,t)=\rho(x)v(z(x),t)$ gives

$$u_t+(gu^2)_x=\rho\,[v_t+(cv^2)_z].$$

It also maps weak formulations and front speeds: $z'\dot x=\kappa/m$,
which equals $c(v_L+v_R)$ for either sign. Initial breakpoints must be mapped
by $z$ while integer flux levels stay unchanged. This is an exact conjugacy
of the quantised front trackers, distinct from the homogenised comparison,
whose initial macroscopic breakpoints stay at their original x-coordinates.
The tests compare these representations through collisions and seams.

For $\varepsilon$-periodic $g$, $z(x)-x$ is periodic because
$\langle\rho\rangle=1$, and $d=\|z-\mathrm{id}\|_\infty=O(\varepsilon)$.
The inverse has the same displacement bound. For any periodic BV function
$q$ and increasing map $\psi$ with $|\psi(x)-x|\le d$,

$$\int|q(\psi(x))-q(x)|\,dx\le2d\mathrm{TV}(q).$$

Indeed, bound the integrand by $|Dq|([x-d,x+d])$ on the periodic lift and
integrate. Every point of the variation measure is counted over a set of
x-length $2d$. For prepared data $u_0=\rho p_0$, the transformed data are
$v_0=p_0\circ z^{-1}$. Homogeneous contraction and TVD therefore give

$$\|u(t)-\rho\bar u(t)\|_{L^1(dx)}
=\|v(t)-\bar u(t)\circ z^{-1}\|_{L^1(dz)}
\le 4d\mathrm{TV}(p_0).$$

Use the triangle inequality through $\bar u(t)$: contraction bounds the first
term by $\|p_0\circ z^{-1}-p_0\|_1\le2d\mathrm{TV}(p_0)$, and the
second is at most $2d\mathrm{TV}(\bar u(t))$.
For the finite-level scheme the same argument applies to the common
homogeneous polygonal flux; notebook 08 checks this bound for quantised
data. This does not bound the separate error between a finite-level run and
the continuum entropy solution.

For constant nonzero $p$, the stationary solution $u^\varepsilon=\rho p$ has
$\|u^\varepsilon-p\|_1=|p|\langle|\rho-1|\rangle>0$ for nonconstant g on a
unit circle containing an integer number of periods. Its primitive error is
$O(\varepsilon)$. This supplies an exact obstruction to uncorrected strong
convergence, alongside a vanishing stationary-corrector error.
The zero datum remains an exception to a blanket failure claim.

`PeriodicStraightening` computes $m$, $z$, displacement extrema, and corrector
errors using closed cell integrals of the actual interpolant. Notebook 08
reports the initial conserved mass mismatch, sampled primitive maxima, exact
corrected L1 errors, and separate sensitivity to delta, data sampling,
coefficient resolution, and observation density. Fixed nodes per microscopic
period hold the interpolated shape fixed: their effective coefficient does
not approach the smooth coefficient just by reducing $\varepsilon$.
