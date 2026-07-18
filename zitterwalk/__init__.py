"""zitterwalk, a simple package to simulate Discrete-Time Quantum Walks.

Main pieces
    - Graph (with Node and Edge), the topology with its arc representation.
    - coin, the coins (Hadamard, Grover, Fourier).
    - Walker, the initial quantum state.
    - DiscreteTimeWalk, the evolution engine.
    - viz, visualization with matplotlib.

Quick example
    >>> from zitterwalk import Graph, Walker, DiscreteTimeWalk
    >>> g = Graph.line(101)
    >>> w = Walker.at_node(g, 50, coin_state=[1, 1j])   # symmetric start
    >>> walk = DiscreteTimeWalk(g, coin="hadamard")
    >>> final = walk.step(w, times=40)
    >>> p = walk.probabilities(final)   # distribution over the nodes
"""

from .graph import Graph, Node, Edge
from .coin import (
    hadamard,
    grover,
    fourier,
    rotation,
    su2,
    build_coin_operator,
    random_coins,
    marked_coins,
    COINS,
)
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
    "rotation",
    "su2",
    "build_coin_operator",
    "random_coins",
    "marked_coins",
    "COINS",
    "viz",
]

__version__ = "0.3.0"
