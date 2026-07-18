"""Klein's paradox in a DTQW: transmission through a high potential step.

Reproduces the Klein-paradox analysis of

    P. Kurzynski, "Relativistic effects in quantum walks: Klein's paradox and
    zitterbewegung", Phys. Lett. A 372 (2008) 6125 (Fig. 3).

A uniform potential over the region ``x >= a`` is a per-step phase ``e^{-iV}``
there -- exactly what the electric ``field`` argument builds (here a *step*
rather than a ramp). A right-moving Dirac wave packet hitting the step shows
three regimes as the step height ``V`` grows from 0 to ~pi:

* small ``V``           -> ordinary transmission (same energy band),
* intermediate ``V``    -> (near) total reflection: the packet's energies land
                           in the forbidden gap on the far side,
* ``V`` near pi         -> transmission *re-emerges* -- the packet tunnels into
                           the negative-energy (antiparticle) band. This
                           re-entrant transmission is Klein's paradox.

Run::

    python examples/klein_paradox.py

Saves the figure to 'klein_paradox.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk.coin import rotation


def transmitted_probability(g, a_step, V, theta, start, steps):
    """Total probability past the step after the given number of steps.

    Args:
        g: the graph (Graph).
        a_step: first site of the potential step (int).
        V: step height (potential phase).
        theta: coin angle (radians).
        start: initial packet centre (int).
        steps: number of steps (int).

    Returns:
        The transmitted probability, the walk, and the initial state.
    """
    walk = DiscreteTimeWalk(g, coin=rotation(theta), shift="moving",
                            field=lambda x: 0.0 if x < a_step else V)
    # coin_state [0, 1] is a right-mover (see examples/zitterbewegung.py)
    w = Walker.gaussian(g, center=start, width=18.0,
                        coin_state=[0, 1], momentum=0.6)
    final = walk.step(w, steps)
    p = walk.probabilities(final)
    xs = np.arange(g.n_nodes)
    return float(p[xs >= a_step].sum()), walk, w


def main():
    """Run the Klein-paradox example and save the figure."""
    n = 1601
    a_step = 900
    start = a_step - 80
    theta = np.pi / 4          # Hadamard-type coin
    steps = 150
    g = Graph.line(n)

    # (a) transmission vs step height V
    Vs = np.linspace(0.0, np.pi, 25)
    trans = [transmitted_probability(g, a_step, V, theta, start, steps)[0]
             for V in Vs]

    # (b) a snapshot in the Klein (re-entrant) regime
    V_klein = 3.0
    t_frac, walk, w0 = transmitted_probability(g, a_step, V_klein, theta,
                                               start, steps)
    states = walk.run(w0, steps)
    P = np.array([walk.probabilities(s) for s in states])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(Vs, trans, "-o", ms=3)
    ax1.axvspan(0, 1.2, color="C2", alpha=0.12)
    ax1.axvspan(1.2, 2.6, color="C3", alpha=0.12)
    ax1.axvspan(2.6, np.pi, color="C0", alpha=0.12)
    ax1.set_xlabel("step height V (potential phase)")
    ax1.set_ylabel("transmitted probability")
    ax1.set_title("transmit / reflect / Klein-transmit")

    xs = np.arange(n)
    win = slice(a_step - 260, a_step + 260)
    ax2.imshow(P[:, win], aspect="auto", origin="lower",
               extent=[a_step - 260, a_step + 260, 0, steps], cmap="magma")
    ax2.axvline(a_step, color="w", lw=1, ls="--")
    ax2.set_xlabel("lattice site x")
    ax2.set_ylabel("step t")
    ax2.set_title(f"packet vs step (V={V_klein}): {t_frac:.0%} transmitted")

    fig.suptitle("Klein's paradox in a discrete-time quantum walk")
    fig.tight_layout()
    fig.savefig("klein_paradox.png", dpi=120)
    print(f"Klein regime V={V_klein}: transmitted {t_frac:.1%}")
    print("Figure saved to klein_paradox.png")


if __name__ == "__main__":
    main()
