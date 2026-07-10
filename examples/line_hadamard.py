"""Canonical example: Hadamard walk on the line.

Reproduces the "two-horn" distribution characteristic of the DTQW and compares
it visually with the diffuse bell curve of a classical walk. Run::

    python examples/line_hadamard.py

Saves the figure to 'line_hadamard.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk import viz


def main():
    n = 201                 # nodes (long line so we don't hit the boundaries)
    center = n // 2
    steps = 80

    g = Graph.line(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")

    # Symmetric initial state: (|left> + i|right>) / sqrt(2).
    w = Walker.at_node(g, center, coin_state=[1, 1j])

    states = walk.run(w, steps)
    p_final = walk.probabilities(states[-1])

    # --- Figure with 3 panels ---------------------------------------- #
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    positions = np.arange(n) - center

    # 1) Final distribution (only nodes with the same parity as 'steps' have
    #    nonzero probability; we drop the rest so it looks clean).
    parity = steps % 2
    mask = (np.arange(n) % 2) == (center + parity) % 2
    ax1.plot(positions[mask], p_final[mask], color="crimson")
    ax1.set_title(f"DTQW Hadamard, t = {steps}")
    ax1.set_xlabel("position")
    ax1.set_ylabel("probability")

    # 2) Time evolution (ballistic front).
    viz.plot_evolution(walk, states, ax=ax2)

    # 3) Comparison with classical diffusion (same theoretical sigma ~ t/sqrt(2)).
    sigma = steps / np.sqrt(2)
    gauss = np.exp(-positions ** 2 / (2 * sigma ** 2))
    gauss /= gauss.sum()
    ax3.plot(positions[mask], p_final[mask], color="crimson", label="quantum")
    ax3.plot(positions, gauss, color="navy", ls="--", label="classical (diffusion)")
    ax3.set_title("Quantum vs classical")
    ax3.set_xlabel("position")
    ax3.legend()

    fig.tight_layout()
    fig.savefig("line_hadamard.png", dpi=120)
    print("Figure saved to line_hadamard.png")

    # A couple of numbers to verify the ballistic spreading.
    mean = (positions * p_final).sum()
    std = np.sqrt(((positions - mean) ** 2 * p_final).sum())
    print(f"standard deviation after {steps} steps: {std:.1f}  (~ ballistic, O(t))")


if __name__ == "__main__":
    main()
