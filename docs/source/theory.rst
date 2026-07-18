Theory in one page
==================

This page collects the minimum of theory needed to read the API with physical
intuition. It is not a textbook — see the references at the bottom for that.


The state lives on the arcs
---------------------------

A discrete-time quantum walk (DTQW) is defined on the **arcs** (directed edges)
of a graph, not on its vertices. Each undirected edge :math:`\{u, v\}` splits
into two arcs, :math:`(u, v)` and :math:`(v, u)`, and the arc :math:`(u, v)` is
read as "the walker is at :math:`u` about to move towards :math:`v`". The
quantum state is one complex amplitude per arc,

.. math::

   |\psi\rangle = \sum_{(u,v)} \psi_{(u,v)}\, |u, v\rangle,
   \qquad \sum_{(u,v)} |\psi_{(u,v)}|^2 = 1 .

In ZitterWalk this vector is exactly ``Walker.psi``, a length-``n_arcs`` complex
array. The number of arcs is easy to see for the built-in graphs:

.. doctest::

   >>> Graph.line(101).n_arcs        # 2*(n-1)
   200
   >>> Graph.cycle(8).n_arcs         # 2*n
   16


One step: coin then shift
-------------------------

A single step is a product of two unitaries,

.. math::

   U = S\, C .

The **coin** :math:`C` acts locally: at every node it applies a small unitary
that mixes the amplitudes of the arcs *leaving* that node. Because arcs leaving
the same node are stored contiguously, :math:`C` is block-diagonal, one block
per node. The **shift** :math:`S` is a permutation of the arcs that moves
amplitude between nodes. The default ``flip_flop`` shift swaps each arc with its
reverse, :math:`(u, v) \mapsto (v, u)`; the ``moving`` shift instead keeps the
walker going "straight".

Since both factors are unitary, :math:`U` is unitary and probability is
conserved for all time — which is the numerical invariant the test suite leans
on:

.. doctest::

   >>> import numpy as np
   >>> g = Graph.cycle(7)
   >>> U = DiscreteTimeWalk(g, coin="grover").U
   >>> bool(np.allclose(U.conj().T @ U, np.eye(g.n_arcs)))
   True

The eigenphases of :math:`U` are the walk's **quasi-energies**, available
through :meth:`~zitterwalk.walk.DiscreteTimeWalk.spectrum`; they live in
:math:`(-\pi, \pi]` with one per arc.


Ballistic vs diffusive spreading
--------------------------------

The headline fact about quantum walks is *how fast they spread*. A classical
random walker diffuses, with a standard deviation that grows like the square
root of time,

.. math::

   \sigma_\text{classical}(t) \sim \sqrt{t}.

Interference changes the exponent. The quantum walker spreads **ballistically**,

.. math::

   \sigma_\text{quantum}(t) \sim t,

quadratically faster. This quadratic gap is the engine behind quantum-walk
search algorithms, and it is easy to see directly from
:meth:`~zitterwalk.walk.DiscreteTimeWalk.std`:

.. plot::
   :caption: Quantum (ballistic) vs classical (diffusive) spreading from the
             same starting point. The quantum curve is a straight line.

   import numpy as np
   import matplotlib.pyplot as plt
   from zitterwalk import Graph, Walker, DiscreteTimeWalk

   g = Graph.line(321)
   walk = DiscreteTimeWalk(g, coin="hadamard")
   w = Walker.at_node(g, 160, coin_state=[1, 1j])
   sigma = walk.std_evolution(walk.run(w, 140))

   t = np.arange(len(sigma))
   fig, ax = plt.subplots(figsize=(6.5, 3.6))
   ax.plot(t, sigma, color="#5b5bd6", lw=2.2, label=r"quantum $\sim t$")
   ax.plot(t, 0.9 * np.sqrt(t), color="#b8c0ff", lw=2, ls="--",
           label=r"classical $\sim \sqrt{t}$")
   ax.set_xlabel("step  t"); ax.set_ylabel(r"$\sigma(t)$"); ax.legend()


Zitterbewegung — the name of the package
-----------------------------------------

With an :func:`~zitterwalk.coin.su2` coin and the ``moving`` shift, the DTQW
becomes a discretization of the **Dirac equation**. A Gaussian packet started in
a single coin component then shows *Zitterbewegung* ("trembling motion"): its
mean position oscillates as it drifts, at a frequency set by the coin angle
:math:`\theta` (the Dirac "mass"). That trembling is where **ZitterWalk** gets
its name.

.. plot::
   :caption: Zitterbewegung: the mean position trembles around its drift at
             angular frequency :math:`\omega_{ZB} = \theta`.

   import numpy as np
   import matplotlib.pyplot as plt
   from zitterwalk import Graph, Walker, DiscreteTimeWalk, su2

   n, center, theta = 601, 300, np.pi / 2
   g = Graph.line(n)
   walk = DiscreteTimeWalk(g, coin=su2(theta, "x"), shift="moving")
   w = Walker.gaussian(g, center=center, width=22.0, coin_state=[1, 0])
   xt = walk.mean_position_evolution(walk.run(w, 80)) - center

   fig, ax = plt.subplots(figsize=(6.5, 3.4))
   ax.plot(np.arange(len(xt)), xt, color="#7c3aed", lw=2)
   ax.set_xlabel("step  t"); ax.set_ylabel(r"$\langle x \rangle - x_0$")
   ax.set_title(r"Dirac walk, $\theta = \pi/2$")


Further reading
---------------

* J. Kempe, *Quantum random walks — an introductory overview*,
  Contemp. Phys. **44**, 307 (2003).
* S. E. Venegas-Andraca, *Quantum walks: a comprehensive review*,
  Quantum Inf. Process. **11**, 1015 (2012).
* A. Ambainis, *Quantum walk algorithm for element distinctness*,
  SIAM J. Comput. **37**, 210 (2007).
