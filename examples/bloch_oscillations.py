"""Electric quantum walk on the line: Bloch oscillations and localization.

Compares three walks with a time x position heatmap:

* free walk (no field)      -> ballistic spreading (light cone),
* rational field E/2pi=1/16 -> periodic revivals (Bloch oscillations),
* irrational field          -> dynamical localization.

Run::

    python examples/bloch_oscillations.py

Saves the figure to 'bloch_oscillations.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from quantum_walks import Graph, Walker, DiscreteTimeWalk
from quantum_walks import viz


def main():
    n = 161
    center = n // 2
    steps = 64

    g = Graph.line(n)

    golden = (1 + np.sqrt(5)) / 2
    walks = {
        "free (no field)": DiscreteTimeWalk(g, coin="hadamard"),
        "rational  E/2π = 1/16": DiscreteTimeWalk(
            g, coin="hadamard", field=lambda x: (2 * np.pi / 16) * (x - center)),
        "irrational  E/2π = φ-1": DiscreteTimeWalk(
            g, coin="hadamard", field=lambda x: (2 * np.pi * (golden - 1)) * (x - center)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, walk) in zip(axes, walks.items()):
        w = Walker.at_node(g, center, coin_state=[1, 1j])
        states = walk.run(w, steps)
        viz.plot_evolution(walk, states, ax=ax)
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig("bloch_oscillations.png", dpi=120)
    print("Figure saved to bloch_oscillations.png")


if __name__ == "__main__":
    main()
