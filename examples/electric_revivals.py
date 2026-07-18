"""Electric quantum walks: revivals, ballistic escape, and localization.

Reproduces the phenomenology of

    C. Cedzich, T. Rybar, A. H. Werner, A. Alberti, M. Genske, R. F. Werner,
    "Propagation of Quantum Walks in Electric Fields", Phys. Rev. Lett. 111,
    160601 (2013).

The electric walk ``W = e^{i x Phi} C S`` (moving shift, SU(2) coin) has a
long-time behaviour that depends dramatically on the field ``Phi``:

* rational ``Phi/2pi = n/m``   -> exact revivals: for odd ``m`` the initial
  state is reproduced at ``t = 2m`` (revival theorem), after which the walk
  expands ballistically;
* very irrational ``Phi``      -> Anderson / dynamical localization: the walk
  stays put, the return probability never decays to zero.

We plot the RMS spread ``sigma(t)`` and the return probability ``p(t)`` for a
rational field ``Phi = 2pi/5`` (revival at ``t = 10``) and the golden-ratio
field ``Phi = 2pi(phi-1)`` (localized).

Run::

    python examples/electric_revivals.py

Saves the figure to 'electric_revivals.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk


def main():
    """Run the electric-revivals example and save the figure."""
    n = 241
    center = n // 2
    steps = 60
    g = Graph.line(n)
    golden = (1 + np.sqrt(5)) / 2

    walks = {
        "rational  Phi=2pi/5  (revival at t=10)":
            2 * np.pi / 5,
        "golden  Phi=2pi(phi-1)  (localized)":
            2 * np.pi * (golden - 1),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, Phi in walks.items():
        walk = DiscreteTimeWalk(g, coin="hadamard", shift="moving",
                                field=lambda x, Phi=Phi: Phi * (x - center))
        w = Walker.at_node(g, center, coin_state=[1, 0])
        states = walk.run(w, steps)
        sigma = walk.std_evolution(states)
        p_ret = walk.return_probability_evolution(states, center)
        ax1.plot(np.arange(steps + 1), sigma, lw=2, label=label)
        ax2.plot(np.arange(steps + 1), p_ret, lw=2, label=label)

    for m in (10, 20, 30, 40, 50):     # revival times 2m for m=5,10,...
        ax2.axvline(m, color="0.8", lw=0.8, zorder=0)

    ax1.set_xlabel("step t"); ax1.set_ylabel("sigma(t)  (RMS spread)")
    ax1.set_title("Spread: ballistic-with-revivals vs localized")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("step t"); ax2.set_ylabel("p(t)  (return probability)")
    ax2.set_title("Return probability: sharp revivals at t = 2m")
    ax2.legend(fontsize=8)

    fig.suptitle("Propagation of quantum walks in electric fields")
    fig.tight_layout()
    fig.savefig("electric_revivals.png", dpi=120)
    print("Figure saved to electric_revivals.png")


if __name__ == "__main__":
    main()
