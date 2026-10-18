"""Matplotlib views of heterogeneous front tracking (ipywidgets only for ``dashboard``).

Two fields are shown. θ = sgn(u) f(x,u) is the discretised, bounded flux
level, piecewise constant between fronts; its sign is the direction of the
characteristics. u is the conserved quantity, a curved stationary profile
between fronts that blows up like |x - z|^{-1/2} at a zero z of g. Fronts are
drawn from their exact closed-form pieces, so curved paths are not polylines
from time stepping. Zeros of g are dashed vertical lines.
"""

from dataclasses import dataclass, field

import numpy as np

from .trajectory import PowerLaw

POSITIVE, NEGATIVE = "#dbe9f6", "#fbe3d0"   # sign-interval shading (g > 0, g < 0)
ZERO_COLOR, FRONT_COLOR = "#b2182b", "0.12"


def _axis(values, name, minimum=2):
    a = np.array(values, dtype=float, copy=True)
    if a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
        raise ValueError(f"{name} must contain at least {minimum} finite, strictly increasing values")
    return a


@dataclass
class History:
    """Observations of one run: sampled fields, exact pieces and diagnostics."""

    x: np.ndarray
    times: np.ndarray
    levels: np.ndarray            # integer levels, shape (len(times), len(x))
    theta: np.ndarray
    u: np.ndarray                 # ±inf replaced by nan
    steps: list                   # per time: (edges, levels) exact θ step profile
    pieces: list                  # closed-form Piece records up to times[-1]
    g: object                     # the PiecewiseAffine heterogeneity
    delta: float
    diagnostics: dict = field(default_factory=dict)
    periodic: bool = False
    power: float = 2.0            # h(u) = |u|^p; needed to redraw fronts and profiles


def _step_profile(tracker, a, b):
    pos = tracker.positions()
    if tracker.cyclic:
        pos = tracker.g.wrap(pos)
    inner = np.sort(pos[(pos > a) & (pos < b)])
    edges = np.unique(np.r_[a, inner, b])
    return edges, np.asarray(tracker.sample_levels(edges[:-1]))


def observe_tracker(tracker, x, times, *, mass_window=None):
    """Advance ``tracker`` through ``times`` and record samples and diagnostics.

    This consumes the tracker up to times[-1]; build a fresh tracker to restart.
    ``mass_window`` (line only) sets the interval for the mass diagnostic; the
    default is the observation window. On a circle the whole circle is used.
    """
    x, times = _axis(x, "x"), _axis(times, "times", 1)
    if times[0] < tracker.time:
        raise ValueError("observation times precede the current tracker time")
    flux = tracker.flux
    window = (x[0], x[-1]) if mass_window is None else mass_window
    if not tracker.cyclic and any(z == window[0] or z == window[1] for z in tracker.g.zeros):
        raise ValueError("mass_window endpoints must not be zeros of g; use one-sided windows")
    rows, steps = [], []
    diag = {k: [] for k in ("fronts", "total_variation", "mass", "absorbed", "events", "edge_flux")}
    edges = np.array(window, dtype=float)
    for t in times:
        tracker.run(float(t))
        rows.append(tracker.sample_levels(x))
        # An endpoint-free whole-circle sampling grid still represents the
        # closing arc; retain it in exact profile plots and exports.
        end = x[-1]
        if (tracker.cyclic and np.isclose(x[0], tracker.g.origin)
                and np.isclose(x[-1] + x[1] - x[0], tracker.g.origin + tracker.g.period)):
            end = tracker.g.origin + tracker.g.period
        steps.append(_step_profile(tracker, x[0], end))
        diag["fronts"].append(tracker.front_count())
        diag["total_variation"].append(tracker.total_variation())
        diag["mass"].append(tracker.integral() if tracker.cyclic else tracker.integral(*window))
        # Only zeros inside a line observation window belong to its budget.
        # Circle diagnostics always cover the entire circle.
        diag["absorbed"].append(sum(iface.absorbed_until(tracker.time, flux.delta)
                                    for iface in tracker.interfaces
                                    if tracker.cyclic or window[0] < iface.x < window[1]))
        diag["events"].append(tracker.event_count)
        if tracker.cyclic:
            diag["edge_flux"].append(0.0)
        else:
            f_a, f_b = flux.flux_value(tracker.sample_levels(edges), edges)
            diag["edge_flux"].append(float(f_a - f_b))
    diag = {k: np.asarray(v) for k, v in diag.items()}
    # Net inflow through the window edges, integrated exactly over the front
    # crossings of each edge (not sampled at the frames). The circle has no edges.
    if tracker.cyclic:
        diag["inflow"] = np.zeros(len(times))
    else:
        diag["inflow"] = tracker.flux_through(window[0], times) - tracker.flux_through(window[1], times)
    levels = np.asarray(rows, dtype=int)
    with np.errstate(invalid="ignore"):
        u = np.asarray(flux.u(levels, np.broadcast_to(x, levels.shape)), dtype=float)
    u[~np.isfinite(u)] = np.nan
    return History(x, times, levels, flux.delta * levels, u, steps,
                   tracker.pieces_until(times[-1]), tracker.g, flux.delta, diag, tracker.cyclic,
                   flux.power)


# ------------------------------------------------------------------ helpers
def theta_norm(history, limits=None):
    """Zero-centred colour scale for θ (its sign is the characteristic direction)."""
    from matplotlib.colors import TwoSlopeNorm, Normalize
    if limits is None:
        m = float(np.max(np.abs(history.theta))) or 1.0
        limits = (-m, m)
    lo, hi = limits
    if lo < 0 < hi:
        return TwoSlopeNorm(0.0, lo, hi)
    return Normalize(lo, hi)


def u_norm(history, limits=None, percentile=99.0):
    """Robust zero-centred scale for u; values blowing up at zeros saturate."""
    from matplotlib.colors import TwoSlopeNorm
    if limits is None:
        finite = np.abs(history.u[np.isfinite(history.u)])
        m = float(np.percentile(finite, percentile)) if finite.size else 1.0
        limits = (-(m or 1.0), m or 1.0)
    return TwoSlopeNorm(0.0, *limits)


def _shade_signs(ax, g, xlim, alpha=0.35):
    for a, b, sign in g.sign_intervals():
        spans = [(a, b)]
        if g.periodic:
            spans = [(a, b), (a - g.period, b - g.period)]
        for lo, hi in spans:
            lo, hi = max(lo, xlim[0]), min(hi, xlim[1])
            if lo < hi:
                ax.axvspan(lo, hi, color=POSITIVE if sign > 0 else NEGATIVE, alpha=alpha, lw=0, zorder=0)


def _zeros_in(g, xlim):
    zs = list(g.zeros)
    if g.periodic:
        zs = [z + k * g.period for z in zs for k in (-1, 0, 1)]
    return [z for z in zs if xlim[0] <= z <= xlim[1]]


def _xlim(history):
    """Observation window; a whole periodic chart is shown closed, [n_0, n_0 + L]."""
    g = history.g
    if g.periodic and np.isclose(history.x[0], g.origin) and \
            np.isclose(history.x[-1] + (history.x[1] - history.x[0]), g.origin + g.period):
        return (g.origin, g.origin + g.period)
    return (history.x[0], history.x[-1])


def thin_pieces(pieces, max_fronts=80):
    """Keep whole front paths for at most about ``max_fronts`` left levels.

    A front keeps its levels across corners, so filtering by level keeps
    complete paths. This only declutters plots; nothing is recomputed.
    """
    levels = sorted({p.kL for p in pieces})
    if len(levels) <= max_fronts:
        return list(pieces)
    stride = -(-len(levels) // max_fronts)
    return [p for p in pieces if p.kL % stride == 0]


def front_lines(pieces, samples=24, power=2.0):
    """Polylines (list of (n, 2) arrays in (x, t)) from closed-form pieces of h = |u|^power."""
    law = PowerLaw(power)
    lines = []
    for p in pieces:
        if p.t1 <= p.t0:
            continue
        n = 2 if p.alpha == 0.0 or p.kappa == 0.0 else samples
        xs, ts = law.sample_path(p.x0, p.s0, p.alpha, p.kappa, p.t0, p.t1, n)
        lines.append(np.column_stack((xs, ts)))
    return lines


# -------------------------------------------------------------------- plots
def plot_heterogeneity(g, window=None, *, ax=None, exact=None, samples=801):
    """g_Δ with corners (dots), zeros (red) and sign intervals shaded; optional exact g."""
    import matplotlib.pyplot as plt
    ax = plt.subplots(figsize=(7, 2.6))[1] if ax is None else ax
    if window is None:
        window = (g.origin, g.origin + g.period) if g.periodic else (
            g.nodes[0] - 0.1 * (g.nodes[-1] - g.nodes[0]), g.nodes[-1] + 0.1 * (g.nodes[-1] - g.nodes[0]))
    xs = np.linspace(*window, samples)
    _shade_signs(ax, g, window)
    if exact is not None:
        ax.plot(xs, np.asarray(exact(xs)) * np.ones_like(xs), color="0.6", lw=1, ls="--", label="g")
    ax.plot(xs, g(xs), color="0.1", lw=1.4, label=r"$g_\Delta$")
    nodes = np.asarray(g.nodes)
    if g.periodic:
        nodes = np.concatenate([nodes + k * g.period for k in (-1, 0, 1)])
    nodes = nodes[(nodes >= window[0]) & (nodes <= window[1])]
    ax.plot(nodes, g(nodes), "o", ms=2.5, color="0.25", label="corners")
    zs = _zeros_in(g, window)
    if zs:
        ax.plot(zs, np.zeros(len(zs)), "o", ms=6, mfc="white", mec=ZERO_COLOR, mew=1.6, label="zeros")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set(xlim=window, xlabel="x", ylabel="g", title="Heterogeneity (sign intervals shaded)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return ax


def plot_spacetime(history, field="theta", *, ax=None, cmap="RdBu_r", norm=None,
                   fronts=True, zeros=True, colorbar=True, characteristics=None, max_fronts=80):
    """Colour field of θ or u in (x, t) with exact front paths and zeros overlaid.

    ``characteristics`` may be a list of (x, t) arrays (e.g. from
    ``MultiplicativeFlux.characteristic``) drawn as thin dotted curves.
    ``max_fronts`` thins the drawn paths by level (None draws every front).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    if len(history.times) < 2:
        raise ValueError("space-time rendering needs at least two times")
    ax = plt.subplots(figsize=(7, 4.5))[1] if ax is None else ax
    values = history.theta if field == "theta" else history.u
    if norm is None:
        norm = theta_norm(history) if field == "theta" else u_norm(history)
    mesh = ax.pcolormesh(history.x, history.times, values, shading="nearest",
                         cmap=cmap, norm=norm, rasterized=True)
    xlim = _xlim(history)
    if fronts and history.pieces:
        pieces = history.pieces if max_fronts is None else thin_pieces(history.pieces, max_fronts)
        lines = front_lines(pieces, power=history.power)
        if history.periodic:
            lines = [np.column_stack((l[:, 0] + k * history.g.period, l[:, 1]))
                     for l in lines for k in (-1, 0, 1)]
        dense = len(pieces) > 600
        ax.add_collection(LineCollection(lines, colors=FRONT_COLOR, linewidths=0.25 if dense else 0.5,
                                         alpha=0.45 if dense else 0.75))
    if characteristics:
        for xs, ts in characteristics:
            ax.plot(xs, ts, color="0.3", lw=0.6, ls=":")
    if zeros:
        for z in _zeros_in(history.g, xlim):
            ax.axvline(z, color=ZERO_COLOR, lw=1.0, ls="--")
    label = r"$\theta = \mathrm{sgn}(u)\,f(x,u)$" if field == "theta" else "u"
    ax.set(xlim=xlim, ylim=(history.times[0], history.times[-1]), xlabel="x", ylabel="t",
           title=f"Space–time: {'flux level θ' if field == 'theta' else 'conserved quantity u'}")
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=label, extend="neither" if field == "theta" else "both")
    return ax


def plot_profile(history, index=-1, field="theta", *, ax=None, norm=None, dense=2001):
    """Snapshot of θ (exact steps) or u (curved stationary profiles, clipped near zeros)."""
    import matplotlib.pyplot as plt
    ax = plt.subplots(figsize=(7, 3.2))[1] if ax is None else ax
    xlim = _xlim(history)
    _shade_signs(ax, history.g, xlim, alpha=0.25)
    edges, levels = history.steps[index]
    t = history.times[index]
    if field == "theta":
        ax.stairs(history.delta * levels, edges, color="0.1", lw=1.4)
        m = float(np.max(np.abs(history.theta))) or 1.0
        ax.set_ylim(-1.1 * m, 1.1 * m)
        ax.set_ylabel("θ")
    else:
        from .flux import MultiplicativeFlux
        flux = MultiplicativeFlux(history.g, history.delta, history.power)
        for a, b, k in zip(edges[:-1], edges[1:], levels):
            xs = np.linspace(a, b, max(3, int(dense * (b - a) / (xlim[1] - xlim[0]))))
            with np.errstate(invalid="ignore"):
                us = np.asarray(flux.u(np.full(xs.shape, k), xs), dtype=float)
            us[~np.isfinite(us)] = np.nan
            ax.plot(xs, us, color="0.1", lw=1.3)
        norm = u_norm(history) if norm is None else norm
        pad = 0.15 * (norm.vmax - norm.vmin)
        ax.set_ylim(norm.vmin - pad, norm.vmax + pad)
        ax.set_ylabel("u")
    for z in _zeros_in(history.g, xlim):
        ax.axvline(z, color=ZERO_COLOR, lw=1.0, ls="--")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set(xlim=xlim, xlabel="x", title=f"{'θ' if field == 'theta' else 'u'} at t = {t:g}")
    return ax


def plot_diagnostics(history, *, axes=None):
    """Front count, total variation of θ, and the mass budget over time."""
    import matplotlib.pyplot as plt
    if axes is None:
        axes = plt.subplots(1, 3, figsize=(11, 2.8), constrained_layout=True)[1]
    d, t = history.diagnostics, history.times
    axes[0].step(t, d["fronts"], where="post", color="0.15")
    axes[0].set(xlabel="t", ylabel="fronts", title="Front count (non-increasing)")
    axes[1].plot(t, d["total_variation"], color="0.15")
    axes[1].set(xlabel="t", ylabel="TV(θ)", title="Total variation of θ")
    axes[2].plot(t, d["mass"], color="#2166ac", label="∫u")
    if np.any(d["absorbed"]) or np.any(d["inflow"]):
        axes[2].plot(t, d["mass"] + d["absorbed"] - d["inflow"], color="0.15", ls="--",
                     label="∫u + absorbed − inflow")
    axes[2].set(xlabel="t", title="Mass budget (window or circle)")
    axes[2].ticklabel_format(axis="y", useOffset=False)
    lo, hi = axes[2].get_ylim()
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    floor = 1e-3 * max(1.0, abs(centre))
    if half < floor:
        axes[2].set_ylim(centre - floor, centre + floor)
    axes[2].legend(fontsize=8, frameon=False)
    return axes


def plot_circle(history, index=-1, *, ax=None, norm=None, cmap="RdBu_r"):
    """θ on the circle: a coloured ring with the profile drawn radially."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    g = history.g
    if not g.periodic:
        raise ValueError("plot_circle needs a periodic run")
    ax = plt.subplots(figsize=(4.2, 4.2))[1] if ax is None else ax
    norm = theta_norm(history) if norm is None else norm
    angle = 2 * np.pi * (history.x - g.origin) / g.period
    theta = history.theta[index]
    m = float(np.max(np.abs(history.theta))) or 1.0
    radius = 1.0 + 0.35 * theta / m
    pts = np.column_stack((np.cos(angle), np.sin(angle)))
    pts = np.vstack((pts, pts[0]))  # include the final arc through the seam
    segs = np.stack((pts[:-1], pts[1:]), axis=1)
    ring = LineCollection(segs, cmap=cmap, norm=norm, linewidths=7)
    ring.set_array(theta)
    ax.add_collection(ring)
    curve = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    curve = np.vstack((curve, curve[0]))
    ax.plot(curve[:, 0], curve[:, 1], color="0.1", lw=1.0)
    for z in g.zeros:
        a = 2 * np.pi * (z - g.origin) / g.period
        ax.plot([0.8 * np.cos(a), 1.4 * np.cos(a)], [0.8 * np.sin(a), 1.4 * np.sin(a)],
                color=ZERO_COLOR, lw=1.2, ls="--")
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal",
           title=f"θ on the circle, t = {history.times[index]:g}")
    ax.axis("off")
    return ax


def overview(history, *, exact_g=None, characteristics=None, title=None):
    """Four panels: g_Δ, θ and u in space–time, and diagnostics; returns the figure."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 2.2, 1.1])
    plot_heterogeneity(history.g, _xlim(history), ax=fig.add_subplot(grid[0, :]),
                       exact=exact_g)
    plot_spacetime(history, "theta", ax=fig.add_subplot(grid[1, :2]), characteristics=characteristics)
    plot_profile(history, -1, "u", ax=fig.add_subplot(grid[1, 2]))
    plot_diagnostics(history, axes=[fig.add_subplot(grid[2, k]) for k in range(3)])
    if title:
        import textwrap
        fig.suptitle(textwrap.fill(title, 120))
    return fig


def dashboard(history, *, fronts=True):
    """Time slider with Play, field selector and front toggle over cached frames.

    Moving the slider never reruns the solver.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    n = len(history.times)
    frame = widgets.IntSlider(min=0, max=n - 1, value=0, description="Frame", continuous_update=False)
    play = widgets.Play(min=0, max=n - 1, interval=150)
    link = widgets.jslink((play, "value"), (frame, "value"))
    chooser = widgets.ToggleButtons(options=[("θ", "theta"), ("u", "u")], value="theta",
                                    description="Field")
    paths = widgets.Checkbox(value=fronts, description="Front paths")
    output = widgets.Output()

    def draw(change=None):
        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
            plot_profile(history, frame.value, chooser.value, ax=axes[0])
            if n > 1:
                plot_spacetime(history, chooser.value, ax=axes[1], fronts=paths.value)
                axes[1].axhline(history.times[frame.value], color="k", lw=1)
            else:
                axes[1].set_visible(False)
            display(fig)
            plt.close(fig)

    for widget in (frame, chooser, paths):
        widget.observe(draw, names="value")
    panel = widgets.VBox([widgets.HBox([play, frame]), widgets.HBox([chooser, paths]), output])
    panel._hetfront_link = link  # keep the widget link alive with the panel
    draw()
    return panel
