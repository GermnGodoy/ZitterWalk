"""zitterwalk: a simple package to simulate Discrete-Time Quantum Walks.

Main pieces
-----------
* :class:`~zitterwalk.graph.Graph` (+ :class:`Node`, :class:`Edge`):
  the topology, with arc representation.
* :mod:`~zitterwalk.coin`: coins (Hadamard, Grover, Fourier).
* :class:`~zitterwalk.walker.Walker`: the initial quantum state.
* :class:`~zitterwalk.walk.DiscreteTimeWalk`: the evolution engine.
* :mod:`~zitterwalk.viz`: visualization with matplotlib.

Quick example
-------------
>>> from zitterwalk import Graph, Walker, DiscreteTimeWalk
>>> g = Graph.line(101)
>>> w = Walker.at_node(g, 50, coin_state=[1, 1j])   # symmetric start
>>> walk = DiscreteTimeWalk(g, coin="hadamard")
>>> final = walk.step(w, times=40)
>>> p = walk.probabilities(final)   # distribution over the nodes
"""

from .graph import Graph, Node, Edge
from .coin import hadamard, grover, fourier, build_coin_operator, COINS
from .walker import Walker
from .walk import DiscreteTimeWalk
from . import viz

__all__ = [
    "Graph",
    "Node",
    "Edge",
    "Walker",
    "DiscreteTimeWalk",
    "hadamard",
    "grover",
    "fourier",
    "build_coin_operator",
    "COINS",
    "viz",
]

__version__ = "0.1.0"
