"""Semiclassical Bloch oscillations of a Dirac wave packet in an electric DTQW.

Reproduces the central result of

    P. Arnault, B. Pepper, A. Perez, "Quantum walks in weak electric fields and
    Bloch oscillations", Phys. Rev. A 101, 062324 (2020).

The electric discrete-time quantum walk on the line is ``W = e^{i x phi} S C``
with the **moving shift** ``S`` and the coin ``C = [[cos t, sin t],
[sin t, -cos t]]`` (``rotation(theta)``). For a *weak* field ``phi`` and a
*wide* initial packet, the walk behaves semiclassically: the mean position
oscillates as a localized particle with the Bloch period

    T_Bloch = 2 pi / phi,

rather than spreading ballistically. This is the discrete-walk analogue of
Bloch oscillations of an electron in a tight-binding band under a static field.

Run::

    python examples/dirac_bloch.py

Saves the figure to 'dirac_bloch.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk.coin import rotation


def main():
    """Run the Bloch-oscillation example and save the figure."""
    n = 401
    center = n // 2
    theta = np.pi / 4
    g = Graph.line(n)

    fields = [2 * np.pi / 30, 2 * np.pi / 45, 2 * np.pi / 60]
    steps = 150

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for phi in fields:
        walk = DiscreteTimeWalk(g, coin=rotation(theta), shift="moving",
                                field=lambda x, phi=phi: phi * (x - center))
        # wide packet gives smooth semiclassical oscillations
        w = Walker.gaussian(g, center=center, width=14.0,
                            coin_state=[1, -1], momentum=0.0)
        states = walk.run(w, steps)
        xt = walk.mean_position_evolution(states) - center
        label = f"phi=2pi/{round(2*np.pi/phi):d}  (T_Bloch={2*np.pi/phi:.0f})"
        ax1.plot(np.arange(steps + 1), xt, lw=2, label=label)

    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("step t")
    ax1.set_ylabel("<x>(t)")
    ax1.set_title("Semiclassical Bloch oscillations: period 2pi/phi")
    ax1.legend(fontsize=8)

    # heatmap for the weakest field, a localized particle oscillating in place
    phi = fields[-1]
    walk = DiscreteTimeWalk(g, coin=rotation(theta), shift="moving",
                            field=lambda x: phi * (x - center))
    w = Walker.gaussian(g, center=center, width=14.0,
                        coin_state=[1, -1], momentum=0.0)
    states = walk.run(w, steps)
    P = np.array([walk.probabilities(s) for s in states])
    win = slice(center - 60, center + 60)
    ax2.imshow(P[:, win], aspect="auto", origin="lower",
               extent=[center - 60, center + 60, 0, steps], cmap="viridis")
    ax2.set_xlabel("lattice site x")
    ax2.set_ylabel("step t")
    ax2.set_title(f"P(x, t) for phi=2pi/{round(2*np.pi/phi):d}")

    fig.suptitle("Electric DTQW: Bloch oscillations of a Dirac packet")
    fig.tight_layout()
    fig.savefig("dirac_bloch.png", dpi=120)
    print("Figure saved to dirac_bloch.png")


if __name__ == "__main__":
    main()
