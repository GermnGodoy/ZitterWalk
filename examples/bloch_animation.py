"""Animated Bloch oscillation of an electric quantum walk on the line.

A constant field makes the walker "breathe": it spreads out and then refocuses
back to the origin every Bloch period, instead of spreading forever. Run::

    python examples/bloch_animation.py

Saves the animation to 'bloch_walk.gif'.
"""

import numpy as np

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk import viz


def main():
    """Run the Bloch animation and save the GIF."""
    n = 161
    center = n // 2
    t_max = 96        # 2 Bloch periods (period ~ 48 steps)

    g = Graph.line(n)
    # A weaker field gives a wider Bloch oscillation (amplitude ~ 1/E), so the
    # breathing is easy to see. E/2pi = 1/48 stays rational -> exact revivals.
    E = 2 * np.pi / 48
    walk = DiscreteTimeWalk(g, coin="hadamard", field=lambda x: E * (x - center))

    w = Walker.at_node(g, center, coin_state=[1, 1j])
    states = walk.run(w, t_max)

    # smooth=1 rescales each frame so the breathing stays framed
    viz.animate(walk, states, save_path="bloch_walk.gif", kind="line",
                fps=10, smooth=1.0)
    print("Animation saved to bloch_walk.gif")


if __name__ == "__main__":
    main()
