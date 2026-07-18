"""Visualization tools for DTQW."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Layout

def layout(graph, kind="auto"):
    """Node positions {node: (x, y)} for drawing.

    Args:
        graph: the graph (Graph).
        kind: layout, one of "auto", "line", "grid", "circular".

    Returns:
        A dict {node: (x, y)}.
    """
    nodes = graph.nodes
    if kind == "auto":
        # heuristic from the id shape
        if all(isinstance(n, tuple) and len(n) == 2 for n in nodes):
            kind = "grid"
        else:
            kind = "circular"
    if kind == "line":
        return {n: (i, 0.0) for i, n in enumerate(nodes)}
    if kind == "grid":
        return {n: (n[1], -n[0]) for n in nodes}  # row 0 on top
    if kind == "circular":
        m = len(nodes)
        ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
        return {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}
    raise ValueError(f"Unknown layout: {kind!r}")


# ----------------------------------------------------------------------
# Plots

def plot_distribution(graph, probabilities, ax=None, **bar_kwargs):
    """Bar plot of the per-node probability.

    Args:
        graph: the graph (Graph).
        probabilities: per-node probability array.
        ax: matplotlib axis to draw on.
        **bar_kwargs: forwarded to ax.bar.

    Returns:
        The matplotlib axis.
    """
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
    """Draw the graph as a node-link diagram colored by probability.

    Args:
        graph: the graph (Graph).
        probabilities: per-node probability array, or None.
        kind: layout kind.
        ax: matplotlib axis to draw on.
        node_size: marker size.
        cmap: colormap name.

    Returns:
        The matplotlib axis.
    """
    pos = layout(graph, kind)
    if ax is None:
        _, ax = plt.subplots()

    # edges
    for e in graph.edges:
        x0, y0 = pos[e.u]
        x1, y1 = pos[e.v]
        ax.plot([x0, x1], [y0, y1], color="0.7", zorder=1, linewidth=1)

    # nodes
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
    """Time-by-node heatmap of a trajectory of states.

    Args:
        walk: the walk (DiscreteTimeWalk).
        states: trajectory from run.
        ax: matplotlib axis to draw on.
        cmap: colormap name.

    Returns:
        The matplotlib axis.
    """
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
            cmap=None, node_size=300, interval=None, smooth=0.15, substeps=4):
    """Animate the per-node probability over time.

    Args:
        walk: the walk (DiscreteTimeWalk).
        states: trajectory from run.
        save_path: output path, saving a GIF for a '.gif' path.
        kind: 'line', 'bar' or 'graph'.
        fps: frames per second.
        cmap: colormap for the 'graph' kind.
        node_size: marker size for the 'graph' kind.
        interval: delay between frames in ms (defaults to 1000/fps).
        smooth: 'line' only, adaptive y-axis smoothing in (0, 1].
        substeps: 'graph' only, interpolated sub-frames between steps (int).

    Returns:
        A matplotlib FuncAnimation.
    """
    import matplotlib.pyplot as plt

    probs = np.array([walk.probabilities(s) for s in states])
    # ignore the localized t=0 spike so later frames stay visible
    disp = probs[1:] if len(probs) > 1 else probs
    vmax = float(disp.max()) if disp.size else 1.0
    if interval is None:
        interval = 1000.0 / fps

    if kind == "line":
        anim = _animate_line(walk, probs, interval, smooth)
    elif kind == "bar":
        anim = _animate_bar(walk, probs, vmax, interval)
    elif kind == "graph":
        anim = _animate_graph(walk, probs, interval, cmap, node_size, vmax, substeps)
    else:
        raise ValueError(f"Unknown kind: {kind!r} (use 'line', 'bar' or 'graph')")

    if save_path is not None:
        _save_anim(anim, save_path, fps)
    return anim


def _save_anim(anim, save_path, fps):
    """Save an animation, using Pillow for a '.gif' path.

    Args:
        anim: the animation.
        save_path: output path.
        fps: frames per second (overridden by anim._save_fps when set).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    fps = getattr(anim, "_save_fps", fps)
    writer = PillowWriter(fps=fps) if str(save_path).endswith(".gif") else None
    anim.save(save_path, writer=writer)
    plt.close(anim._fig)


def _smoothed_ymax(probs, smooth):
    """Per-frame y-limit that follows the peak with an exponential lag.

    Args:
        probs: per-frame probability array.
        smooth: smoothing factor in (0, 1].

    Returns:
        An array of per-frame y-limits.
    """
    frame_max = probs.max(axis=1)
    ymax = np.empty_like(frame_max)
    acc = frame_max[min(1, len(frame_max) - 1)]  # skip the t=0 delta
    for t, m in enumerate(frame_max):
        acc = smooth * m + (1.0 - smooth) * acc
        ymax[t] = acc
    return ymax


# light palette built around an indigo/violet accent
_PALETTE = {
    "bg": "#ffffff",       # background
    "fg": "#2b2d42",       # text / ticks
    "line": "#5b5bd6",     # curve + shaded area
    "bars": "#b8c0ff",     # bars
}


def _qwalk_cmap():
    """Cohesive lavender-to-violet colormap for node fills.

    Returns:
        A matplotlib colormap.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "qwalk", ["#f5f5ff", "#b8c0ff", "#5b5bd6", "#7c3aed"]
    )


def _style_axes(ax):
    """Apply the light palette to a matplotlib axis.

    Args:
        ax: the matplotlib axis.
    """
    p = _PALETTE
    ax.set_facecolor(p["bg"])
    for spine in ax.spines.values():
        spine.set_color(p["fg"])
        spine.set_alpha(0.3)
    ax.tick_params(colors=p["fg"])
    ax.grid(True, color=p["fg"], alpha=0.12)
    ax.set_xlabel("node", color=p["fg"])


def _line_panel(ax, walk, probs, smooth, eps=1e-9):
    """Draw bars, a smooth curve and a shaded area on an axis.

    Args:
        ax: the matplotlib axis.
        walk: the walk (DiscreteTimeWalk).
        probs: per-frame probability array.
        smooth: smoothing factor in (0, 1].
        eps: threshold below which nodes are dropped from the curve.

    Returns:
        An update(t) callback that redraws frame t.
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
    """Single-panel line animation.

    Args:
        walk: the walk (DiscreteTimeWalk).
        probs: per-frame probability array.
        interval: delay between frames in ms.
        smooth: smoothing factor in (0, 1].

    Returns:
        A matplotlib FuncAnimation.
    """
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
        states_list: list of trajectories, one per walk.
        labels: optional per-panel titles.
        save_path: output path, saving a GIF for a '.gif' path.
        fps: frames per second.
        smooth: adaptive y-axis smoothing in (0, 1].
        interval: delay between frames in ms (defaults to 1000/fps).

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
    """Bar-chart animation of the per-node probability.

    Args:
        walk: the walk (DiscreteTimeWalk).
        probs: per-frame probability array.
        vmax: fixed y-limit.
        interval: delay between frames in ms.

    Returns:
        A matplotlib FuncAnimation.
    """
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
    anim._fig = fig  # handle for closing after save
    return anim


def _interpolate_frames(probs, substeps):
    """Linearly interpolate sub-frames between consecutive steps.

    Args:
        probs: per-step probability array.
        substeps: sub-frames inserted between steps (int).

    Returns:
        The interpolated frames and their fractional step indices.
    """
    n = len(probs)
    if substeps <= 1 or n < 2:
        return probs, np.arange(n, dtype=float)
    t_src = np.arange(n)
    t_dst = np.linspace(0, n - 1, (n - 1) * substeps + 1)
    interp = np.empty((len(t_dst), probs.shape[1]))
    for j in range(probs.shape[1]):
        interp[:, j] = np.interp(t_dst, t_src, probs[:, j])
    return interp, t_dst


def _animate_graph(walk, probs, interval, cmap, node_size, vmax, substeps):
    """Node-link animation with nodes colored by probability.

    Args:
        walk: the walk (DiscreteTimeWalk).
        probs: per-step probability array.
        interval: delay between steps in ms.
        cmap: colormap, or None for the default.
        node_size: marker size.
        vmax: fixed color-scale maximum.
        substeps: interpolated sub-frames between steps (int).

    Returns:
        A matplotlib FuncAnimation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    p = _PALETTE
    if cmap is None:
        cmap = _qwalk_cmap()
    graph = walk.graph
    pos = layout(graph)

    probs_i, times = _interpolate_frames(probs, substeps)
    frame_interval = interval / substeps

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
    # fixed color scale across frames
    sc = ax.scatter(xs, ys, c=probs_i[0], s=node_size, cmap=cmap,
                    vmin=0, vmax=vmax, zorder=2,
                    edgecolors=p["fg"], linewidths=1.2)
    cbar = fig.colorbar(sc, ax=ax, label="probability", shrink=0.8)
    cbar.ax.yaxis.label.set_color(p["fg"])
    cbar.ax.tick_params(colors=p["fg"])
    ax.set_aspect("equal")
    ax.margins(0.15)  # so large markers aren't clipped
    ax.axis("off")
    title = ax.set_title("t = 0", color=p["fg"], fontweight="bold")

    def update(i):
        sc.set_array(probs_i[i])
        title.set_text(f"t = {int(np.floor(times[i]))}")
        return (sc, title)

    anim = FuncAnimation(fig, update, frames=len(probs_i),
                         interval=frame_interval, blit=False)
    anim._fig = fig
    anim._save_fps = 1000.0 / frame_interval
    return anim
