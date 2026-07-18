ZitterWalk
==========

.. rst-class:: lead

   A small, dependency-light package to simulate **Discrete-Time Quantum Walks
   (DTQW)** — built for studying and experimenting.

**ZitterWalk** turns a graph into a quantum-walk simulator in a handful of
lines. You pick a topology (a line, a cycle, a grid, a hypercube, or one of your
own), drop a *coin* on it, prepare a *walker*, and let a unitary step operator
evolve the state. From there you can read off distributions, spreading
exponents, entanglement entropy, search success probabilities, or animate the
whole thing.

.. code-block:: python

   from zitterwalk import Graph, Walker, DiscreteTimeWalk

   g = Graph.line(201)                              # line of 201 nodes
   w = Walker.at_node(g, 100, coin_state=[1, 1j])   # symmetric start at the center
   walk = DiscreteTimeWalk(g, coin="hadamard")

   final = walk.step(w, times=80)                   # evolve 80 steps
   p = walk.probabilities(final)                    # per-node distribution
   walk.std(final)                                  # spread ~ t (ballistic)

The signature Hadamard-walk distribution on a line, produced directly from the
code above:

.. plot::
   :caption: Hadamard walk on a line after 80 steps: probability piles up at the
             ballistic fronts near :math:`\pm t/\sqrt{2}`, nothing like a
             classical bell curve.

   import numpy as np
   from zitterwalk import Graph, Walker, DiscreteTimeWalk

   n, center, steps = 201, 100, 80
   g = Graph.line(n)
   walk = DiscreteTimeWalk(g, coin="hadamard")
   w = Walker.at_node(g, center, coin_state=[1, 1j])
   p = walk.probabilities(walk.step(w, steps))

   pos = np.arange(n) - center
   mask = (np.arange(n) % 2) == (center + steps) % 2   # keep the reachable parity
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=(7, 3.2))
   ax.fill_between(pos[mask], p[mask], color="#5b5bd6", alpha=0.25)
   ax.plot(pos[mask], p[mask], color="#5b5bd6", lw=2)
   ax.set_xlabel("position")
   ax.set_ylabel("probability")
   ax.set_title("DTQW Hadamard walk, t = 80")


Why quantum walks?
------------------

A classical random walk diffuses: after :math:`t` steps its standard deviation
grows like :math:`\sigma(t) \sim \sqrt{t}`. A quantum walk replaces the coin
flip and the move by *unitary* operators acting on a superposition, so the
walker is a wave of complex amplitudes that **interferes** with itself. The net
effect is *ballistic* spreading, :math:`\sigma(t) \sim t` — quadratically faster.
That speed-up is what powers quantum-walk search algorithms, and the same
dynamics reproduce the Dirac and Schrödinger equations in suitable limits.

Read :doc:`theory` for the one-page version of the math, or jump straight into
the :doc:`tutorial`.


Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   tutorial
   theory

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/zitterwalk


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
