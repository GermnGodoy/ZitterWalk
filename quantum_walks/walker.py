"""The walker, represents a quantum state."""

from __future__ import annotations

import numpy as np


class Walker:
    """State (possibly in superposition) of a walker on a graph."""

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
        """Walker localized at node.

        Args:
            graph: Graph instance.
            node: Node ID where the walker is localized.
            coin_state: Optional array of amplitudes for the outgoing arcs.
        Returns:
            Walker instance.
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
    def superposition(cls, graph, amplitudes):
        """Walker from a dict {arc (u, v): amplitude}.

        Useful to prepare arbitrary states over specific arcs.

        Args:
            graph: Graph instance.
            amplitudes: Dict mapping arcs (u, v) to complex amplitudes.
        Returns:
            Walker instance.
        """
        graph._ensure_arcs()
        psi = np.zeros(graph.n_arcs, dtype=complex)
        for (u, v), amp in amplitudes.items():
            psi[graph.arc_index[(u, v)]] = amp
        return cls(graph, _normalize(psi))

    # ------------------------------------------------------------------ 
    # Utilities 
    @property
    def norm(self):
        return float(np.linalg.norm(self.psi))

    def copy(self):
        return Walker(self.graph, self.psi.copy())

    def __repr__(self):
        return f"Walker(n_arcs={self.psi.size}, norm={self.norm:.6f})"


def _normalize(psi):
    n = np.linalg.norm(psi)
    if n == 0:
        raise ValueError("The initial state is zero; cannot normalize.")
    return psi / n
