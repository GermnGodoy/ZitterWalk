"""The walker, the quantum state living on the arcs of a graph."""

from __future__ import annotations

import numpy as np


class Walker:
    """Quantum state of a walker on a graph.

    Attributes:
        graph: the graph the state lives on (Graph).
        psi: complex amplitude per arc (length n_arcs).
    """

    def __init__(self, graph, psi):
        graph._ensure_arcs()
        psi = np.asarray(psi, dtype=complex)
        if psi.shape != (graph.n_arcs,):
            raise ValueError(
                f"The state has shape {psi.shape}, expected ({graph.n_arcs},)."
            )
        self.graph = graph
        self.psi = psi

    # ------------------------------------------------------------------
    # Constructors of initial states

    @classmethod
    def at_node(cls, graph, node, coin_state=None):
        """Walker localized on a single node.

        Args:
            graph: the graph (Graph).
            node: node id where the walker sits.
            coin_state: amplitudes on the node's arcs (defaults to uniform).

        Returns:
            The Walker.
        """
        graph._ensure_arcs()
        psi = np.zeros(graph.n_arcs, dtype=complex)
        idxs = graph.node_arcs[node]
        d = len(idxs)
        if d == 0:
            raise ValueError(f"Node {node!r} is isolated; nowhere to go.")
        if coin_state is None:
            coin_state = np.ones(d, dtype=complex)
        else:
            coin_state = np.asarray(coin_state, dtype=complex)
            if coin_state.shape != (d,):
                raise ValueError(
                    f"coin_state must have length {d} (degree of {node!r}), "
                    f"not {coin_state.shape}."
                )
        psi[idxs] = coin_state
        return cls(graph, _normalize(psi))

    @classmethod
    def uniform(cls, graph):
        """Equal superposition over all arcs.

        Args:
            graph: the graph (Graph).

        Returns:
            The Walker.
        """
        graph._ensure_arcs()
        psi = np.ones(graph.n_arcs, dtype=complex)
        return cls(graph, _normalize(psi))

    @classmethod
    def superposition(cls, graph, amplitudes):
        """Walker from a dict {arc (u, v): amplitude}.

        Args:
            graph: the graph (Graph).
            amplitudes: dict mapping arcs (u, v) to complex amplitudes.

        Returns:
            The Walker.
        """
        graph._ensure_arcs()
        psi = np.zeros(graph.n_arcs, dtype=complex)
        for (u, v), amp in amplitudes.items():
            psi[graph.arc_index[(u, v)]] = amp
        return cls(graph, _normalize(psi))

    @classmethod
    def gaussian(cls, graph, center, width, coin_state=None, momentum=0.0):
        """Gaussian wave packet on the line or cycle.

        Args:
            graph: the graph (Graph).
            center: packet centre, in coordinate units.
            width: Gaussian standard deviation, in coordinate units.
            coin_state: amplitudes on each node's arcs (defaults to uniform).
            momentum: initial quasimomentum.

        Returns:
            The Walker.
        """
        graph._ensure_arcs()
        psi = np.zeros(graph.n_arcs, dtype=complex)
        cs = None if coin_state is None else np.asarray(coin_state, dtype=complex)
        for nid in graph.nodes:
            idxs = graph.node_arcs[nid]
            d = len(idxs)
            if d == 0:
                continue
            if cs is not None and len(cs) != d:
                continue
            x = graph.node_coord(nid)[0]
            envelope = np.exp(-((x - center) ** 2) / (4.0 * width ** 2)) \
                * np.exp(1j * momentum * x)
            psi[idxs] = envelope * (np.ones(d, dtype=complex) if cs is None else cs)
        return cls(graph, _normalize(psi))

    # ------------------------------------------------------------------
    # Utilities

    @property
    def norm(self):
        """Euclidean norm of the state."""
        return float(np.linalg.norm(self.psi))

    def copy(self):
        """Return an independent copy of the walker.

        Returns:
            A new Walker with a copied state.
        """
        return Walker(self.graph, self.psi.copy())

    def __repr__(self):
        return f"Walker(n_arcs={self.psi.size}, norm={self.norm:.6f})"


def _normalize(psi):
    """Normalize a state vector to unit norm.

    Args:
        psi: amplitude array.

    Returns:
        The normalized array.
    """
    n = np.linalg.norm(psi)
    if n == 0:
        raise ValueError("The initial state is zero.")
    return psi / n
