"""Visualization tools for DTQW."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------- 
# Layout

def layout(graph, kind="auto"):
    """Return a dict {node: (x, y)} for drawing."""
    nodes = graph.nodes
    if kind == "auto":
        # Heuristic based on the shape of the ids.
        if all(isinstance(n, tuple) and len(n) == 2 for n in nodes):
            kind = "grid"
        else:
            kind = "circular"
    if kind == "line":
        return {n: (i, 0.0) for i, n in enumerate(nodes)}
    if kind == "grid":
        return {n: (n[1], -n[0]) for n in nodes}  # (col, -row): row 0 on top
    if kind == "circular":
        m = len(nodes)
        ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
        return {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}
    raise ValueError(f"Unknown layout: {kind!r}")


# ---------------------------------------------------------------------- 
# Plots

def plot_distribution(graph, probabilities, ax=None, **bar_kwargs):
    """Pllot of the per-node probability."""
    
    p = np.asarray(probabilities, dtype=float)
    if ax is None:
        _, ax = plt.subplots()
    x = np.arange(graph.n_nodes)
    ax.bar(x, p, **bar_kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in graph.nodes], rotation=45, fontsize=8)
    ax.set_xlabel("Node")
    ax.set_ylabel("Probability")
    ax.set_title("Walker distribution")
    return ax


def plot_graph(graph, probabilities=None, kind="auto", ax=None, node_size=300, cmap="viridis"):
    """Draw the graph as a node-link diagram, the color of the nodes is given by probability."""

    pos = layout(graph, kind)
    if ax is None:
        _, ax = plt.subplots()

    # Edges.
    for e in graph.edges:
        x0, y0 = pos[e.u]
        x1, y1 = pos[e.v]
        ax.plot([x0, x1], [y0, y1], color="0.7", zorder=1, linewidth=1)

    # Nodes.
    xs = [pos[n][0] for n in graph.nodes]
    ys = [pos[n][1] for n in graph.nodes]
    if probabilities is None:
        colors = "steelblue"
    else:
        colors = np.asarray(probabilities, dtype=float)
    sc = ax.scatter(xs, ys, c=colors, s=node_size, cmap=cmap,
                    zorder=2, edgecolors="black")
    
    if probabilities is not None:
        plt.colorbar(sc, ax=ax, label="probability")
    
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Graph")
    
    return ax


def plot_evolution(walk, states, ax=None, cmap="viridis"):
    """Time x node heatmap of a list of states."""

    probs = np.array([walk.probabilities(s) for s in states])
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(probs, aspect="auto", origin="lower",
                   cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="probability")
    ax.set_xlabel("Node")
    ax.set_ylabel("Step (t)")
    ax.set_title("Time evolution")
    return ax


def animate(walk, states, save_path=None, kind="line", fps=10,
            cmap=None, node_size=300, interval=None, smooth=0.15):
    """Animate the per-node probability over time.

    Args:
        walk: a DiscreteTimeWalk.
        states: list of Walker (e.g. from ``walk.run``).
        save_path: if given, save the animation there (GIF from a '.gif' path).
        kind: 'line' (continuous curve, good for line/cycle), 'bar' (bar chart
              per node) or 'graph' (node-link diagram colored by probability).
        fps: frames per second, used for saving and default playback speed.
        cmap: colormap for the 'graph' kind (defaults to a cohesive indigo one).
        node_size: marker size for the 'graph' kind.
        interval: delay between frames in ms (defaults to 1000/fps).
        smooth: 'line' only. Smoothing of the adaptive y-axis, in (0, 1]. Small
              values track the shrinking peak slowly (roughly normalized height
              without fully rescaling every frame); 1 rescales each frame.
    Returns:
        A matplotlib FuncAnimation.
    """
    import matplotlib.pyplot as plt

    probs = np.array([walk.probabilities(s) for s in states])
    # Scale to the *spread* dynamics: the localized t=0 spike (prob 1 at one
    # node) would otherwise flatten every later frame. It just saturates.
    disp = probs[1:] if len(probs) > 1 else probs
    vmax = float(disp.max()) if disp.size else 1.0
    if interval is None:
        interval = 1000.0 / fps

    if kind == "line":
        anim = _animate_line(walk, probs, interval, smooth)
    elif kind == "bar":
        anim = _animate_bar(walk, probs, vmax, interval)
    elif kind == "graph":
        anim = _animate_graph(walk, probs, interval, cmap, node_size, smooth)
    else:
        raise ValueError(f"Unknown kind: {kind!r} (use 'line', 'bar' or 'graph')")

    if save_path is not None:
        _save_anim(anim, save_path, fps)
    return anim


def _save_anim(anim, save_path, fps):
    """Save an animation (GIF via Pillow when the path ends in '.gif')."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=fps) if str(save_path).endswith(".gif") else None
    anim.save(save_path, writer=writer)
    plt.close(anim._fig)


def _smoothed_ymax(probs, smooth):
    """Per-frame y-limit that follows the peak with an exponential lag.

    Returns an array ``ymax[t]``. The t=0 localized spike is ignored as a
    starting point so the axis reflects the spread dynamics.
    """
    frame_max = probs.max(axis=1)
    ymax = np.empty_like(frame_max)
    acc = frame_max[min(1, len(frame_max) - 1)]  # skip the t=0 delta
    for t, m in enumerate(frame_max):
        acc = smooth * m + (1.0 - smooth) * acc
        ymax[t] = acc
    return ymax


# Light palette built around an indigo/violet accent.
_PALETTE = {
    "bg": "#ffffff",       # background
    "fg": "#2b2d42",       # text / ticks
    "line": "#5b5bd6",     # curve + shaded area
    "bars": "#b8c0ff",     # bars
}


def _qwalk_cmap():
    """Cohesive colormap (light lavender -> indigo -> violet) for node fills."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "qwalk", ["#f5f5ff", "#b8c0ff", "#5b5bd6", "#7c3aed"]
    )


def _style_axes(ax):
    """Apply the light palette (background, spines, ticks, grid) to an axis."""
    p = _PALETTE
    ax.set_facecolor(p["bg"])
    for spine in ax.spines.values():
        spine.set_color(p["fg"])
        spine.set_alpha(0.3)
    ax.tick_params(colors=p["fg"])
    ax.grid(True, color=p["fg"], alpha=0.12)
    ax.set_xlabel("node", color=p["fg"])


def _line_panel(ax, walk, probs, smooth, eps=1e-9):
    """Draw bars + smooth curve + shaded area on ``ax``; return ``update(t)``.

    The curve is the envelope (near-zero nodes are dropped each frame, so it is
    not a comb on bipartite graphs like the line), over a matching bar chart
    with a low-alpha fill. The y-limit follows the shrinking peak (see
    :func:`_smoothed_ymax`) so the walk stays visible instead of fading.
    """
    p = _PALETTE
    x = np.arange(walk.graph.n_nodes)
    ymax = _smoothed_ymax(probs, smooth)

    _style_axes(ax)
    ax.set_xlim(0, walk.graph.n_nodes - 1)
    bars = ax.bar(x, probs[0], color=p["bars"], alpha=0.5, width=0.9)
    (line,) = ax.plot(x, probs[0], color=p["line"], lw=2.4, zorder=3)
    fill = [ax.fill_between(x, probs[0], color=p["line"], alpha=0.2, zorder=2)]

    def update(t):
        prob = probs[t]
        for bar, h in zip(bars, prob):
            bar.set_height(h)
        mask = prob > eps
        line.set_data(x[mask], prob[mask])
        fill[0].remove()
        fill[0] = ax.fill_between(x[mask], prob[mask], color=p["line"],
                                  alpha=0.2, zorder=2)
        ax.set_ylim(0, ymax[t] * 1.1)

    return update


def _animate_line(walk, probs, interval, smooth):
    """Single-panel line animation (see :func:`_line_panel`)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    p = _PALETTE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(p["bg"])
    ax.set_ylabel("probability", color=p["fg"])
    panel = _line_panel(ax, walk, probs, smooth)
    title = ax.set_title("t = 0", color=p["fg"], fontweight="bold")

    def update(t):
        panel(t)
        title.set_text(f"t = {t}")

    anim = FuncAnimation(fig, update, frames=len(probs),
                         interval=interval, blit=False)
    anim._fig = fig
    return anim


def animate_compare(walks, states_list, labels=None, save_path=None,
                    fps=10, smooth=0.15, interval=None):
    """Animate several line walks side by side, sharing a time counter.

    Args:
        walks: list of DiscreteTimeWalk.
        states_list: list of state lists (one per walk), e.g. from ``walk.run``.
        labels: optional per-panel titles.
        save_path: if given, save the animation (GIF from a '.gif' path).
        fps, smooth, interval: as in :func:`animate`.
    Returns:
        A matplotlib FuncAnimation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    p = _PALETTE
    m = len(walks)
    probs_list = [np.array([w.probabilities(s) for s in states])
                  for w, states in zip(walks, states_list)]
    n_frames = min(len(pr) for pr in probs_list)
    if interval is None:
        interval = 1000.0 / fps
    if labels is None:
        labels = [None] * m

    fig, axes = plt.subplots(1, m, figsize=(6 * m, 4.5), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(p["bg"])
    axes[0].set_ylabel("probability", color=p["fg"])

    panels = []
    for ax, walk, probs, label in zip(axes, walks, probs_list, labels):
        panels.append(_line_panel(ax, walk, probs, smooth))
        if label:
            ax.set_title(label, color=p["fg"], fontweight="bold")
    clock = fig.suptitle("t = 0", color=p["fg"], fontweight="bold", fontsize=13)

    def update(t):
        for panel in panels:
            panel(t)
        clock.set_text(f"t = {t}")

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False)
    anim._fig = fig
    if save_path is not None:
        _save_anim(anim, save_path, fps)
    return anim


def _animate_bar(walk, probs, vmax, interval):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    graph = walk.graph
    x = np.arange(graph.n_nodes)
    fig, ax = plt.subplots()
    bars = ax.bar(x, probs[0])
    ax.set_ylim(0, vmax * 1.05)
    ax.set_xlabel("node")
    ax.set_ylabel("probability")
    title = ax.set_title("t = 0")

    def update(t):
        for bar, h in zip(bars, probs[t]):
            bar.set_height(h)
        title.set_text(f"t = {t}")
        return (*bars, title)

    anim = FuncAnimation(fig, update, frames=len(probs),
                         interval=interval, blit=False)
    anim._fig = fig  # keep a handle so we can close it after saving
    return anim


def _animate_graph(walk, probs, interval, cmap, node_size, smooth):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    p = _PALETTE
    if cmap is None:
        cmap = _qwalk_cmap()
    graph = walk.graph
    pos = layout(graph)
    # Adaptive color scale, so populated nodes stay vivid even as the walk
    # spreads (a fixed scale washes out most frames on small cycles that revive).
    ymax = _smoothed_ymax(probs, smooth)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(p["bg"])
    ax.set_facecolor(p["bg"])

    for e in graph.edges:
        x0, y0 = pos[e.u]
        x1, y1 = pos[e.v]
        ax.plot([x0, x1], [y0, y1], color=p["line"], alpha=0.25,
                zorder=1, linewidth=1.5)

    xs = [pos[n][0] for n in graph.nodes]
    ys = [pos[n][1] for n in graph.nodes]
    sc = ax.scatter(xs, ys, c=probs[0], s=node_size, cmap=cmap,
                    vmin=0, vmax=ymax[0], zorder=2,
                    edgecolors=p["fg"], linewidths=1.2)
    cbar = fig.colorbar(sc, ax=ax, label="probability", shrink=0.8)
    cbar.ax.yaxis.label.set_color(p["fg"])
    cbar.ax.tick_params(colors=p["fg"])
    ax.set_aspect("equal")
    ax.axis("off")
    title = ax.set_title("t = 0", color=p["fg"], fontweight="bold")

    def update(t):
        sc.set_array(probs[t])
        sc.set_clim(0, ymax[t] * 1.1)
        title.set_text(f"t = {t}")
        return (sc, title)

    anim = FuncAnimation(fig, update, frames=len(probs),
                         interval=interval, blit=False)
    anim._fig = fig
    return anim
