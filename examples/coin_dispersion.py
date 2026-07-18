"""Tuning the coin: the rotation coin and group velocity.

The one-parameter rotation coin ``C(theta) = [[cos, sin], [sin, -cos]]`` is the
knob that sets how fast a walk on the line spreads. ``theta = pi/4`` is the
Hadamard coin (widest two-horn); ``theta -> 0`` makes the coin nearly the
identity so the walker races outward as two thin ballistic peaks; ``theta ->
pi/2`` freezes it near the origin. The peaks always sit at ``+/- cos(theta) t``,
so the spread is linear in time with a slope you dial in.

Run::

    python examples/coin_dispersion.py

Saves the figure to 'coin_dispersion.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk, rotation


def main():
    """Run the coin-dispersion example and save the figure."""
    n = 301
    center = n // 2
    steps = 100
    g = Graph.line(n)
    w = Walker.at_node(g, center, coin_state=[1, 1j])
    pos = np.arange(n) - center

    thetas = [0.15 * np.pi, 0.25 * np.pi, 0.35 * np.pi, 0.45 * np.pi]
    fig, ax = plt.subplots(figsize=(9, 5))

    for theta in thetas:
        walk = DiscreteTimeWalk(g, coin=rotation(theta))
        final = walk.step(w, steps)
        p = walk.probabilities(final)
        # keep only the reachable parity, for a clean curve
        mask = (np.arange(n) % 2) == (center + steps) % 2
        ax.plot(pos[mask], p[mask],
                label=fr"$\theta$ = {theta/np.pi:.2f}$\pi$, "
                      fr"$v = \cos\theta$ = {np.cos(theta):.2f}")
        # predicted ballistic peak position +/- cos(theta) * t
        ax.axvline(np.cos(theta) * steps, color="0.85", lw=0.8, zorder=0)
        ax.axvline(-np.cos(theta) * steps, color="0.85", lw=0.8, zorder=0)

    ax.set_xlabel("position")
    ax.set_ylabel("probability")
    ax.set_title(f"Rotation coin C($\\theta$) on the line, t = {steps}\n"
                 "(grey lines: predicted peaks at $\\pm\\cos\\theta \\cdot t$)")
    ax.legend()

    fig.tight_layout()
    fig.savefig("coin_dispersion.png", dpi=120)
    print("Figure saved to coin_dispersion.png")


if __name__ == "__main__":
    main()
