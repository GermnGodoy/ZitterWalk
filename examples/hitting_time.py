"""Absorbing walks and hitting times.

Put an *absorbing* node on the line: every step, any amplitude that has reached
it is removed (a continuous "is the walker here yet?" measurement). The
surviving probability decays, and its step-to-step drop is the first-passage
(hitting-time) distribution -- when the walker is absorbed for the first time.

Because a quantum walk travels ballistically, it reaches a distant trap far
sooner than a classical diffusive walk would. We start the walker next to a
trap and watch it get caught.

Run::

    python examples/hitting_time.py

Saves the figure to 'hitting_time.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk


def main():
    """Run the hitting-time example and save the figure."""
    n = 201
    start_node = 100
    trap = 60           # absorbing vertex, 40 sites to the left
    steps = 220

    g = Graph.line(n)
    w = Walker.at_node(g, start_node, coin_state=[1, 1j])
    walk = DiscreteTimeWalk(g, coin="hadamard", absorbing=trap)

    states = walk.run(w, steps)
    survival = np.array([walk.survival_probability(s) for s in states])
    first_passage = -np.diff(survival)          # prob. of absorption at step t
    t = np.arange(1, steps + 1)

    absorbed = 1.0 - survival[-1]
    # Mean hitting time, conditioned on having been absorbed by the end.
    mean_ht = (t * first_passage).sum() / absorbed if absorbed > 0 else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(np.arange(steps + 1), survival, color="navy")
    ax1.set_xlabel("step t")
    ax1.set_ylabel("surviving probability")
    ax1.set_title(f"Survival (trap {start_node - trap} sites away)")
    ax1.set_ylim(0, 1.02)

    ax2.plot(t, first_passage, color="crimson")
    ax2.axvline(mean_ht, color="0.5", ls="--",
                label=f"mean hitting time ~ {mean_ht:.0f}")
    ax2.set_xlabel("step t")
    ax2.set_ylabel("first-passage probability")
    ax2.set_title("Hitting-time distribution")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("hitting_time.png", dpi=120)
    print("Figure saved to hitting_time.png")
    print(f"absorbed by t={steps}: {absorbed:.3f}   mean hitting time: {mean_ht:.1f}")


if __name__ == "__main__":
    main()
