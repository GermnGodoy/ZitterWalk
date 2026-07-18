"""Zitterbewegung: the trembling motion of a Dirac particle, in a DTQW.

Reproduces the central effect of

    P. Kurzynski, "Relativistic effects in quantum walks: Klein's paradox and
    zitterbewegung", Phys. Lett. A 372 (2008) 6125.

A discrete-time quantum walk with the **moving shift** ``S|x,±> = |x±1,±>`` and
a two-dimensional coin is a discretization of the 1-D free Dirac equation. A
wave packet built from both energy bands (particle + antiparticle) shows
*zitterbewegung*: its mean position ``<x>(t)`` trembles at the angular frequency
of the band gap, ``omega_ZB = E+(k0) - E-(k0)``, on top of its uniform drift.

At the band centre ``k0 = 0`` the gap equals the eigenphase splitting of the
coin ``C`` itself. With the genuine SU(2) coin ``su2(theta) = exp(i theta/2
sigma_x)`` this splitting is exactly ``theta``, so ``omega_ZB = theta`` -- the
coin angle literally plays the role of the particle mass (Kurzynski).

Run::

    python examples/zitterbewegung.py

Saves the figure to 'zitterbewegung.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk.coin import su2


def zb_frequency(states, walk):
    """Angular frequency of the residual oscillation of <x>(t), drift removed.

    Args:
        states: trajectory from run.
        walk: the walk (DiscreteTimeWalk).

    Returns:
        The dominant angular frequency and the <x>(t) curve.
    """
    xt = walk.mean_position_evolution(states)
    t = np.arange(len(xt))
    detrended = xt - np.poly1d(np.polyfit(t, xt, 1))(t)
    amp = np.abs(np.fft.rfft(detrended))
    freq = np.fft.rfftfreq(len(detrended))
    return 2 * np.pi * freq[1:][np.argmax(amp[1:])], xt


def main():
    """Run the zitterbewegung example and save the figure."""
    n = 801
    center = n // 2
    steps = 90
    g = Graph.line(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, theta in zip(axes, (np.pi / 2, np.pi / 4)):
        # Dirac particle of "mass" theta, su2 coin eigenphase gap = theta
        walk = DiscreteTimeWalk(g, coin=su2(theta, axis="x"), shift="moving")
        # wide standing packet at k0 = 0 overlapping both bands
        w = Walker.gaussian(g, center=center, width=25.0,
                            coin_state=[1, 0], momentum=0.0)
        states = walk.run(w, steps)
        omega, xt = zb_frequency(states, walk)

        ax.plot(np.arange(steps + 1), xt - center, lw=2)
        ax.set_title(f"theta={theta:.3f}:  omega_ZB={omega:.2f}  (theory {theta:.2f})")
        ax.set_xlabel("step t")
        ax.set_ylabel("<x>(t)")
        ax.axhline(0, color="k", lw=0.5)
        print(f"theta={theta:.4f}  measured omega_ZB={omega:.3f}  "
              f"theory (=theta)={theta:.3f}")

    fig.suptitle("Zitterbewegung: <x>(t) trembles at the band-gap frequency")
    fig.tight_layout()
    fig.savefig("zitterbewegung.png", dpi=120)
    print("Figure saved to zitterbewegung.png")


if __name__ == "__main__":
    main()
