"""Side-by-side electric quantum walk: rational vs irrational field.

Two line walks under a constant field animated together:

* rational   E/2pi = 1/48  -> the walker breathes and refocuses exactly
  (Bloch oscillations, periodic revivals),
* irrational E/2pi = phi-1  -> it stays localized, never refocusing cleanly
  (dynamical localization).

Run::

    python examples/bloch_compare.py

Saves the animation to 'bloch_compare.gif'.
"""

import numpy as np

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk import viz


def main():
    n = 161
    center = n // 2
    t_max = 96

    g = Graph.line(n)
    golden = (1 + np.sqrt(5)) / 2

    rational = DiscreteTimeWalk(
        g, coin="hadamard", field=lambda x: (2 * np.pi / 48) * (x - center))
    irrational = DiscreteTimeWalk(
        g, coin="hadamard", field=lambda x: (2 * np.pi * (golden - 1)) * (x - center))

    walks = [rational, irrational]
    labels = ["rational  E/2π = 1/48  (Bloch)", "irrational  E/2π = φ-1  (localized)"]
    states = [w.run(Walker.at_node(g, center, coin_state=[1, 1j]), t_max)
              for w in walks]

    # smooth=1: rescale each frame so both panels stay framed as they evolve.
    viz.animate_compare(walks, states, labels=labels,
                        save_path="bloch_compare.gif", fps=10, smooth=1.0)
    print("Animation saved to bloch_compare.gif")


if __name__ == "__main__":
    main()
