Installation
============

Requirements
------------

ZitterWalk targets **Python 3.10+** and depends only on the scientific-Python
staples:

* `NumPy <https://numpy.org/>`_ (``>=1.24``) — the linear algebra behind every
  step operator.
* `Matplotlib <https://matplotlib.org/>`_ (``>=3.6``) — used by the
  :mod:`zitterwalk.viz` plotting and animation helpers.

Nothing else is required for the core simulator.


Install from source
--------------------

The package is not on PyPI yet, so install it from a clone in editable mode:

.. code-block:: bash

   git clone https://github.com/GermnGodoy/ZitterWalk.git
   cd ZitterWalk
   pip install -e .

Editable (``-e``) means your working copy *is* the installed package, so any
change you make to the source is picked up immediately — handy while
experimenting.

To also pull in the test dependencies (``pytest``):

.. code-block:: bash

   pip install -e ".[test]"


Check the installation
----------------------

A quick smoke test that also doubles as a first taste of the API. The step
operator of a quantum walk is unitary, so the total probability must stay at
exactly ``1`` no matter how long you evolve:

.. doctest::

   >>> from zitterwalk import Graph, Walker, DiscreteTimeWalk
   >>> import zitterwalk
   >>> zitterwalk.__version__
   '0.3.0'
   >>> g = Graph.line(41)
   >>> walk = DiscreteTimeWalk(g, coin="hadamard")
   >>> w = Walker.at_node(g, 20, coin_state=[1, 1j])
   >>> final = walk.step(w, times=15)
   >>> p = walk.probabilities(final)
   >>> bool(abs(p.sum() - 1.0) < 1e-12)
   True

If that prints ``True`` you are ready to go.


Run the tests
-------------

From the repository root:

.. code-block:: bash

   python -m pytest

The suite checks the physics directly — coin unitarity, norm conservation,
ballistic spreading, Anderson localization under disorder, Bloch oscillations,
zitterbewegung frequency, and Grover search — so a green run means the simulator
reproduces the expected quantum-walk phenomenology.


Building the documentation
---------------------------

The documentation you are reading is built with `Sphinx
<https://www.sphinx-doc.org/>`_. To build it locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   make html          # output in docs/build/html/index.html
   make doctest       # run every example in these pages
