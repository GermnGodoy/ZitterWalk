"""The discrete-time quantum walk (DTQW) engine."""

from __future__ import annotations

import numpy as np

from .coin import build_coin_operator
from .walker import Walker


class DiscreteTimeWalk:
    """Discrete-time quantum walk on a graph.

    Attributes:
        - graph: Graph instance that defines the topology of the walk.
        - coin: Coin operator to use at each node.
        - field: static electric field (a phase potential), or None.
    """

    # ------------------------------------------------------------------
    # Construction

    def __init__(self, graph, coin="grover", field=None):
        graph._ensure_arcs()
        self.graph = graph
        self.coin = coin
        self.field = field
        self.C = build_coin_operator(graph, coin)
        self.S = self._build_shift(graph)
        U = self.S @ self.C
        # Static electric field: a per-step phase e^{-i phi(x)} applied after
        # the shift (position = tail of the arc). Diagonal, so fold it into U.
        self.phases = self._build_field_phases(graph, field)
        if self.phases is not None:
            U = self.phases[:, None] * U
        self.U = U

    @staticmethod
    def _build_shift(graph):
        """Permutation matrix of the flip-flop shift."""
        A = graph.n_arcs
        S = np.zeros((A, A), dtype=complex)
        for i, j in enumerate(graph.flip):
            S[j, i] = 1.0
        return S

    @staticmethod
    def _build_field_phases(graph, field):
        """Per-arc phases e^{-i phi(x)} of a static electric field.

        Args:
            field: None (no field); a number (constant field, potential
                   field * node, for numeric node ids); or a callable
                   node -> potential (arbitrary scalar potential).
        Returns:
            A length-n_arcs complex array, or None if there is no field.
        """
        if field is None:
            return None
        potential = field if callable(field) else (lambda node: field * node)
        return np.array([np.exp(-1j * potential(u)) for (u, _) in graph.arcs])

    # ------------------------------------------------------------------ 
    # Evolution
     
    def step(self, walker, times=1):
        """Apply times steps and return a new Walker."""
        psi = walker.psi
        for _ in range(times):
            psi = self.U @ psi # operator
        return Walker(self.graph, psi)

    def run(self, walker, steps):
        """Run steps steps and return the list of states [t=0..steps].

        Includes the initial state at index 0, so the list has steps + 1.
        Basically step but preserving all intermediate states.
        """
        states = [walker.copy()]
        current = walker
        for _ in range(steps):
            current = self.step(current)
            states.append(current)
        return states

    # ------------------------------------------------------------------ 
    # Observables                                                        
     
    def probabilities(self, walker):
        """Probability distribution over the nodes."""
        graph = self.graph
        p = np.zeros(graph.n_nodes)
        prob_arcs = np.abs(walker.psi) ** 2
        for nid in graph.nodes:
            idxs = graph.node_arcs[nid]
            p[graph.node_position(nid)] = prob_arcs[idxs].sum()
        return p

    def probability_dict(self, walker):
        """Probabilities but as a dict."""
        p = self.probabilities(walker)
        return {nid: p[self.graph.node_position(nid)] for nid in self.graph.nodes}

    def __repr__(self):
        return (
            f"DiscreteTimeWalk(coin={self.coin!r}, field={self.field!r}, "
            f"n_arcs={self.graph.n_arcs})"
        )
