"""Quantum spatial search: the O(sqrt(N)) speedup.

A coined quantum walk can find a marked vertex quadratically faster than a
classical random walk. The recipe (Shenvi-Kempe-Whaley) is simple with
inhomogeneous coins: drive the walk with the Grover coin everywhere, but flip
the sign of the coin (``-I``) on the marked vertex. Starting from the uniform
superposition over all arcs, amplitude concentrates on the marked vertex after
about ``sqrt(N)`` steps, where a classical search needs ``~N``.

We run it on two classic substrates -- the complete graph and the hypercube --
and plot the success probability (probability on the marked vertex) versus
time, marking the ``sqrt(N)`` scale.

Run::

    python examples/quantum_search.py

Saves the figure to 'quantum_search.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk


def success_curve(graph, marked, steps):
    """Success probability over time for a search on a graph.

    Args:
        graph: the graph to search (Graph).
        marked: marked vertex id, or a list of them.
        steps: number of steps (int).

    Returns:
        An array of success probabilities, one per step.
    """
    walk = DiscreteTimeWalk.search(graph, marked=marked)
    start = Walker.uniform(graph)
    states = walk.run(start, steps)
    return np.array([walk.success_probability(s) for s in states])


def main():
    """Run the quantum-search example and save the figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Complete graph K_N, for a few sizes. (The step operator is dense and
    # scales as O(n_arcs^2) = O(N^4) here, so we keep N modest -- see the note
    # on performance in docs/guide.md.)
    for N in (16, 36, 64):
        steps = int(3 * np.sqrt(N))
        sp = success_curve(Graph.complete(N), marked=0, steps=steps)
        t = np.arange(len(sp))
        axes[0].plot(t / np.sqrt(N), sp, label=f"K_{N}")
    axes[0].axhline(1.0, color="0.7", lw=0.8, ls=":")
    axes[0].set_xlabel(r"time / $\sqrt{N}$")
    axes[0].set_ylabel("success probability")
    axes[0].set_title("Complete graph: peak near t ~ $\\sqrt{N}$")
    axes[0].legend()

    # Hypercube Q_d (N = 2**d).
    for d in (4, 6, 8):
        N = 1 << d
        steps = int(2.5 * np.sqrt(N))
        sp = success_curve(Graph.hypercube(d), marked=0, steps=steps)
        axes[1].plot(np.arange(len(sp)), sp, label=f"Q_{d}  (N={N})")
    axes[1].set_xlabel("step t")
    axes[1].set_ylabel("success probability")
    axes[1].set_title("Hypercube search")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("quantum_search.png", dpi=120)
    print("Figure saved to quantum_search.png")

    for N in (16, 36, 64):
        sp = success_curve(Graph.complete(N), marked=0, steps=int(3 * np.sqrt(N)))
        print(f"K_{N:>4}: peak success {sp.max():.3f} at t = {sp.argmax()}  "
              f"(sqrt(N) = {np.sqrt(N):.1f})")


if __name__ == "__main__":
    main()
