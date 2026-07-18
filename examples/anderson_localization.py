"""Static coin disorder on the line: Anderson localization.

A clean Hadamard walk spreads ballistically (the two-horn distribution). If we
instead give each site its *own* random coin (static disorder), interference
turns destructive on average and the walker stops spreading: it stays
exponentially localized around the origin. This is the discrete-time cousin of
Anderson localization.

We show it two ways:

  * the final distribution, clean vs. disordered (log scale, so the
    exponential localization shows up as a straight-line tent), and
  * the participation ratio over time -- how many sites are occupied -- which
    keeps growing for the clean walk but saturates under disorder.

Run::

    python examples/anderson_localization.py

Saves the figure to 'anderson_localization.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk, random_coins


def main():
    """Run the Anderson-localization example and save the figure."""
    n = 401
    center = n // 2
    steps = 150
    g = Graph.line(n)
    w = Walker.at_node(g, center, coin_state=[1, 1j])

    clean = DiscreteTimeWalk(g, coin="hadamard")
    # average a few disorder realizations so the profile is smooth
    realizations = 30

    clean_states = clean.run(w, steps)
    clean_pr = np.array([clean.participation_ratio(s) for s in clean_states])
    clean_p = clean.probabilities(clean_states[-1])

    dis_p = np.zeros(n)
    dis_pr = np.zeros(steps + 1)
    for r in range(realizations):
        walk = DiscreteTimeWalk(g, coin=random_coins(g, seed=r))
        states = walk.run(w, steps)
        dis_p += walk.probabilities(states[-1])
        dis_pr += [walk.participation_ratio(s) for s in states]
    dis_p /= realizations
    dis_pr /= realizations

    pos = np.arange(n) - center
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.semilogy(pos, np.maximum(clean_p, 1e-12), color="crimson",
                 label="clean Hadamard (ballistic)")
    ax1.semilogy(pos, np.maximum(dis_p, 1e-12), color="navy",
                 label=f"disordered, avg of {realizations}")
    ax1.set_xlim(-steps, steps)
    ax1.set_ylim(1e-8, 1)
    ax1.set_xlabel("position")
    ax1.set_ylabel("probability (log)")
    ax1.set_title(f"Final distribution, t = {steps}")
    ax1.legend()

    ax2.plot(clean_pr, color="crimson", label="clean (grows ~ t)")
    ax2.plot(dis_pr, color="navy", label="disordered (saturates)")
    ax2.set_xlabel("step t")
    ax2.set_ylabel("participation ratio")
    ax2.set_title("How many sites are occupied")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("anderson_localization.png", dpi=120)
    print("Figure saved to anderson_localization.png")
    print(f"final participation ratio  clean={clean_pr[-1]:.0f}  "
          f"disordered={dis_pr[-1]:.0f}")


if __name__ == "__main__":
    main()
