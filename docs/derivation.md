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
* **(A1′) General convex $h$** (`MultiplicativeFlux(..., h="u^2+u^4")`, §3b).
  $h$ is even, $h(0)=0$, strictly convex on $[0,\infty)$, and of power type
  at infinity: $h(r)\gtrsim r^k$ for some $k>1$. Power-type growth is slightly
  stronger than the Osgood condition $\int^\infty dr/h<\infty$, which is the
  real dividing line for the zeros (§6). A nonzero minimum of $h$ is excluded:
  it would add the source $m\,g'(x)$. An asymmetric $h$, equal to $h_+(u)$ for $u\ge0$ and
  $h_-(-u)$ for $u<0$, is allowed when each branch satisfies these conditions
  (`h="u^2; u^2+u^4"`, §3b).
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
the flux $b|u|^p$. For a general convex $h$ the front ODE on an affine cell
still integrates in closed form, but no longer linearly in any power of $g$;
see §3b.

## 3b. General convex $h$

Under (A1′) write $H=h^{-1}$ on $[0,\infty)$, $\gamma=|g_\Delta|$ (affine on a cell,
slope $\alpha$), $\eta=\mathrm{sgn}\,g$. The stationary profile of level $\theta$ is

$$U[\theta](x)=\eta\,\mathrm{sgn}(\theta)\,H\big(|\theta|/\gamma(x)\big),\qquad f=\eta|\theta|,$$

so $\mathrm{sgn}\,\theta$ is still the direction of the characteristic speed
$f_u=\gamma\,h'(H(|\theta|/\gamma))\,\mathrm{sgn}\,\theta$, on either sign of $g$.

**Proposition (no straightening coordinate).** Suppose a $C^1$ coordinate
$\sigma(\gamma)$ makes every characteristic a straight line in $(\sigma,t)$ on every
affine cell. Then $h=c|u|^p$.

*Proof.* Along a characteristic $\dot\sigma=\sigma'(\gamma)\,\alpha\,\gamma h'(H(\theta/\gamma))$
must be constant while $\gamma$ varies, for each $\theta>0$. Hence
$\gamma h'(H(\theta/\gamma))=a(\theta)/\sigma'(\gamma)$. Put $\psi=h'\circ H$ and
$y=\theta/\gamma$: then $\psi(y)=a(y\gamma)\,b(\gamma)$ for all $y,\gamma>0$, with $b=1/(\gamma\sigma')$.
Taking $\gamma=1$ gives $a=\psi/b(1)$, so $\psi(y\gamma)=\psi(y)\,b(1)/b(\gamma)$. With $y=1$,
$b(1)/b(\gamma)=\psi(\gamma)/\psi(1)$, and $\phi=\psi/\psi(1)$ solves Cauchy's multiplicative
equation $\phi(y\gamma)=\phi(y)\phi(\gamma)$. Its continuous solutions are $\phi(y)=y^r$.
Then $h'(u)=C\,h(u)^r$. If $r=1$, $h=Ae^{Cu}$, which cannot vanish at 0. Otherwise
$h^{1-r}$ is affine in $u$, and $h(0)=0$ forces $h=c\,u^{1/(1-r)}$. $\square$

Numerically, for $u^2+u^4$ the speed ratio of the levels 1 and 4 is $0.339, 0.341, 0.366$
at $\gamma=0.5,1,4$; for a power it is constant. The σ-straightening of §3, the
conjugacy of §5 and the exact homogenisation map of §9 are therefore all
specific to power laws.

**Primitives.** Let $J(u)=\int_u^\infty dr/h(r)$ (finite by (A1′)) and
$Q(u)=u+h(u)J(u)$. Define

$$P(\theta,\gamma)=\gamma\,Q\big(H(\theta/\gamma)\big)\quad(\theta,\gamma>0),\qquad P(\theta,0)=0,$$

extended oddly in $\theta$. With $u=H(\theta/\gamma)$, so that $\gamma h(u)=\theta$, direct
differentiation gives

$$\partial_\gamma P = u,\qquad \partial_\theta P = J(u),\qquad \partial_\theta\partial_\gamma P=\frac{1}{\gamma h'(u)} .$$

The first is the stationary state, the second the Osgood primitive, and the
third the characteristic slowness. All three are **proved**.

**Front law.** A front $(\theta_L,\theta_R)$ has Rankine–Hugoniot speed
$\dot x=[f]/[u]=[|\theta|]/[H(\theta/\gamma)]$, independent of $\eta$. Its slowness
$S=dt/dx$ is therefore $[H(\theta/\gamma)]/[|\theta|]$, and with $d\gamma=\alpha\,dx$,

$$\alpha\,[|\theta|]\,(t_1-t_0)=\Big[P(\theta_R,\gamma)-P(\theta_L,\gamma)\Big]_{\gamma_0}^{\gamma_1}.$$

For levels of one sign, $S$ is the $\theta$-average of $\partial_\theta\partial_\gamma P$ over the
jump: *a front's slowness is the mean characteristic slowness of the levels it
spans.* A characteristic carrying $\theta$ satisfies
$\alpha\,(t_1-t_0)=\mathrm{sgn}\,\theta\,\big(J(u_1)-J(u_0)\big)$. For $h=|u|^p$, $Q=mu$ and
$P\propto s(\theta)\sigma$, which recovers §3: $P(\theta_R,\cdot)-P(\theta_L,\cdot)$ is linear in σ.
The cell mass of a profile is
$\int U[\theta]\,dx=\eta\,\mathrm{sgn}\,\theta\,\big(P(|\theta|,\gamma_2)-P(|\theta|,\gamma_1)\big)/\alpha$.

**Events.** Arrival at a node, a corner or a zero is explicit. A front's
position at a given time and the meeting time of two fronts are roots of
monotone scalar functions:

* **Lemma (speed order is independent of position).** For fronts $a=(\theta_1,\theta_2)$
  and $b=(\theta_2,\theta_3)$, the sign of $\dot x_a-\dot x_b$ at a common $\gamma$ does not
  depend on $\gamma$. Proof: at fixed $\gamma$ both speeds are secant slopes of the
  convex (or, for $\eta<0$, concave) function $\eta\gamma h$ between states
  $u_i=\eta\,\mathrm{sgn}\,\theta_i\,H(|\theta_i|/\gamma)$. The order of the $u_i$ is the order of the
  $\theta_i$ (reversed when $\eta<0$) for every $\gamma$, and secant slopes of a convex
  function are ordered by the order of their endpoints. $\square$
* Hence if $a$ closes on $b$ at one point of a cell, it does so everywhere on
  it. The time-to-reach functions $T_a(x)$, $T_b(x)$ satisfy $T_a'=S_a$ and $T_b'=S_b$, so
  $T_a-T_b$ is strictly monotone along the common path, and the meeting point is
  a bracketed root. The code uses Newton with the exact derivative
  $S_a-S_b$, safeguarded by the bracket. There is still no ODE solver.

**Asymmetric $h$.** Let $h=h_+$ on $u\ge0$ and $h(u)=h_-(-u)$ on $u<0$, each branch as in
(A1′). The glued $h$ is convex on $\mathbb R$ with minimum $0$ at $u=0$, so the
θ shock/fan rule and the lemma below are unchanged. The profile of level
$\theta$ on a cell of sign $\eta$ has $\mathrm{sgn}\,u=\eta\,\mathrm{sgn}\,\theta$, so it uses the branch
$h_{\eta\,\mathrm{sgn}\theta}$. Every formula above holds with that branch's $H$, $J$ and $Q$, level by
level. A front's coefficient therefore carries $\eta$, which is constant along
the front because fronts never cross zeros of $g$. Two consequences:

* **Single-branch traces at zeros.** The admissible traces $\theta(z^-)\ge0\ge\theta(z^+)$
  give $\mathrm{sgn}\,u=\eta_-$ on the left and $-\eta_+$ on the right. At a transverse
  zero $\eta_+=-\eta_-$, so both traces have the sign of $\eta_-=-\mathrm{sgn}\,g'(z)$. Where
  $g'(z)<0$ the selected solution beside the zero lives on $h_+$, and where
  $g'(z)>0$ it lives on $h_-$. The blow-up rate at each zero is that branch's top
  degree (**proved**; tested on both signs of $g'$).
* **Reflection.** If $u$ solves the problem with $(g,h_+,h_-)$, then $v=-u$ solves it with
  $(-g,h_-,h_+)$, and the θ levels coincide. The tests and the stress check use
  this as an exact oracle: identical levels, fronts and event counts, with $u\mapsto-u$.

**Numerics** (`trajectory.GeneralLaw`, `nonlinearity`).

* Times are $(x_1-x_0)$ times a mean slowness, so no formula divides by $\alpha$.
  When $\gamma_0,\gamma_1$ are far apart this mean comes from the closed form above.
  Otherwise it is Gauss–Legendre on the smooth $S(\gamma)$, with the number of points
  chosen from the Bernstein ellipse through $\gamma=0$. Fan steps use a
  cancellation-free divided difference of $h$.
* $J$ is evaluated in factored form, $J=(u/h)\,R$ with $R=hJ/u=Q/u-1$. For a power,
  $R\equiv1/(p-1)$. For $\sum c_k|u|^k$, $R$ is bounded and runs from
  $1/(k_{\min}-1)$ to $1/(k_{\max}-1)$. So $u/h$ carries both power laws of $J$ exactly:
  divergence at $0$, decay at $\infty$. $R$ is tabulated once per $h$ as piecewise
  Chebyshev series in $\log u$, on the window where all lower (respectively
  upper) terms of $h$ exceed $1/20$ of the leading one. Outside it, $J$ is its exact
  asymptotic series. At infinity this is a series in $u^{-(k_{\max}-k_i)}$. At zero it
  is the local series plus a finite-part constant ($\pi/2$ for $u^2+u^4$), computed
  once. Log terms appear where exponents resonate (for $u^2+|u|^3$, $1/r$ integrates
  to $\log u$). Against 300-digit closed forms the factored $J$ is accurate to
  $8\cdot10^{-16}$ for $u\in[10^{-30},10^{30}]$ (**observed**).
* Root solves stop once a Newton step is below $10^{-9}$ of the smaller of two
  lengths: the displacement from the anchor, and the distance to the zero of
  $g_\Delta$, where the slowness is singular. Bisection is geometric in the
  displacement. A front just emitted from, or about to be absorbed into, a
  zero is then located to full *relative* precision in its distance to the zero,
  as `PowerLaw`'s closed forms are. There $|u|$ can be $10^4$–$10^6$, so this precision
  is what keeps mass budgets at roundoff.
* Cost: about 1 ms per event at the fine resolutions of the convergence studies,
  and 2–3 ms on coarse meshes, where cells are long and more Gauss points are
  needed. `PowerLaw` costs about 0.05–0.08 ms per event.

`tests/test_general.py` checks the following:

* the generic numerics against `PowerLaw` (via $\sum c_k|u|^k$ with a single term);
* positions against RK4 of $\dot x=[gh]/[u]$;
* meeting times against bisection;
* the characteristic's arrival at a zero at exactly $J(H(|\theta|))/\alpha$;
* mass budgets through absorption, $L^1$ contraction, exact cell integrals, and the interaction invariants.

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

*General convex $h$.* The conjugacy is unavailable (§3b), but the discrete rule
needs only the pointwise speed order of §3b's lemma.

* **Riemann rule.** At a point $x$ with flux frozen at $\eta\gamma(x)h$, a decreasing jump in
  $\theta$ is a decreasing jump in $u$ for $\eta>0$, an admissible shock for a convex
  flux, and an increasing jump in $u$ for $\eta<0$, admissible for a concave one.
  Fan steps have speeds ordered by the lemma at every $\gamma$, so they separate
  for all time and across corners.
* **At most one outgoing front.** Two adjacent fan steps never meet, so in any
  colliding cluster every unit increase of the level is followed by a shock
  that brings it back to at most its earlier value. By induction
  $k_{\text{last}}\le k_{\text{first}}+1$. The Riemann solution between the exterior
  levels is therefore at most one front. This argument is combinatorial and
  holds for any convex $h$ (**proved** for the discrete scheme).

Convergence for a general $h$ is **not proved**. For well-signed $g$ and
uniformly convex $h$, such as $u^2+u^4$ or $u^2+|u|^3$ ($h''\ge2$), HetFT's uniform-convexity
hypothesis holds, unlike for $|u|^p$ with $p\ne2$. That leaves only the smoothness of
$g_\Delta$ as the gap. Measured rates are in [the general-flux report](convergence-2026-11-05.md).

## 6. Zeros of $g$: absorbing interfaces

As a front approaches a zero, $\sigma\to0$ and its speed $\sigma\kappa\to0$.
Yet $\dot\sigma$ is constant, so the zero is reached **in finite time**, and the
front is absorbed into a stationary interface. The same holds for every
$p>1$, with $\sigma=|g|^{1-1/p}$ and speed $|g|^{1/p}K$.

For a general $h$ (§3b) one condition, $\int^\infty dr/h<\infty$, controls the
behaviour at a zero. It is equivalent to each of the following:

* the profiles $H(|\theta|/\gamma)$ are integrable at the zero;
* fronts and characteristics reach it in finite time, since $P(\theta,0)=0$ is finite;
* the manuscript's $\dot p=-g'h(p)$ blows up in finite time.

Where it fails, for example for the convex, superlinear $u\log(1+u)$, zeros
become unreachable and the absorption and non-uniqueness mechanism is lost.
(A1′) asks slightly more, power-type growth, so that the arrival times stay
comparable with powers of the distance. For a polynomial, the top degree
$k_{\max}$ governs: $U\sim|x-z|^{-1/k_{\max}}$. The bottom degree governs small
amplitudes. The interface condition and the mass accounting below are
unchanged. The interface
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

For a general $h$ (§3b) the same argument uses the exponent that governs $U$
near the zero, the top degree $k_{\max}$ of a polynomial. On the line,
[the general-flux report](convergence-2026-11-05.md) measures 1.60 for $u^2+|u|^3$,
level by level equal to $|u|^3$, and 1.70 for $u^2+u^4$ (**observed**).

*Zero alignment.* Inserting a zero as a node splits its cell at a fraction
$f$ that, under dyadic refinement, stays fixed only when the zero is a dyadic
number. Otherwise it wanders from level to level. The near-zero error
constant depends on $f$, so local mesh slopes then fluctuate rather than settle.
On the line problems $f$ is the same at every level, $(0.75,0.5,0.25)$, which is why
their slopes are clean. On the sign-changing circle it wanders
($0.22,0.45,0.90,0.79,0.59$). A translated pair of otherwise identical problems
shows the effect directly: with the zeros on nodes the slopes rise monotonically to
$1.67\approx5/3$; shifted by $0.1137$ they swing between $0.6$ and $2.0$. A
fixed-δ floor is ruled out: the mesh errors there are δ-independent to 0.3%
([report](convergence-2026-11-05.md#zero-alignment-and-the-circle-slopes)).
Fitted orders on misaligned meshes summarise this fluctuation. They are not
asymptotic rates (**observed**).

Meshes are therefore zero-aligned by default (`PiecewiseAffine.aligned`): every
zero is a node, and the cells between zeros are equal and bisected at each
level. [The aligned-mesh report](convergence-2026-11-06.md) fits
$C(h^r-h_{\rm ref}^r)$, which accounts for the reference being only four times
finer. With that correction the line orders are $2-1/k_{\max}$ to within 0.04
for $|u|^{3/2}$, $u^2$, $|u|^3$, $u^2+|u|^3$ and $u^2+u^4$, with log residuals of
at most 0.5%. On the circle the slopes rise steadily and are 0.04–0.08 short at
1024 cells (**observed**).

The heuristic is sharper as a sum. A cell of length $\ell$ at distance $d$
from the zero contributes about $\ell^3d^{-1-1/k_{\max}}$, so a uniform mesh gives
$h^{2-1/k_{\max}}\sum_i i^{-1-1/k_{\max}}$. This is not the touching cell
alone. Alignment fixes the constant's dependence on $f$, not the rate, and costs
accuracy: the uniform mesh's inserted node beside each zero lowers the constant
1.3–1.6-fold on the line. Power grading, $\ell(d)\approx\Delta d^\beta$ with
$\beta>1/(2k_{\max})$, should restore $O(\Delta^2)$ at $O(\Delta^{-1})$ cells
(**untested**).

## 9. Homogenisation and the exact coordinate map

This section is written for $h=u^2$; here $p$ denotes the mean state, not the
exponent. For $h=|u|^{\bar p}$ (exponent $\bar p$), replace $|g|^{-1/2}$ by
$|g|^{-1/\bar p}$ throughout: $m=\langle|g|^{-1/\bar p}\rangle$,
$\rho=1/(m|g|^{1/\bar p})$ and $c=\eta m^{-\bar p}$, with effective flux
$c|v|^{\bar p}$. The substitution $u=\rho\,v\circ z$ again leaves the levels
unchanged, since $\mathrm{sgn}(u)g|u|^{\bar p}=\mathrm{sgn}(v)c|v|^{\bar p}$.
`PeriodicStraightening(g, power=p̄)` implements this.

For a general convex $h$ there is no such map: $U[\theta]$ does not factor as
$\rho(x)\,V(\theta)$ (§3b), and `PeriodicStraightening` rejects one. The
homogenised flux is still explicit at the levels.
`homogenisation.effective_flux_nodes` returns
$\bar v_k=\eta\,\mathrm{sgn}(k)\langle H(|k|\delta/|g|)\rangle$, from closed-form cell means via $P$,
with $\bar F_k=\eta|k|\delta$. For $|u|^p$ these nodes lie on $c|v|^p$. For a general $h$ they define a
homogenised *comparison*, and the $O(\varepsilon)$ corrector bound below is not proved.

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
