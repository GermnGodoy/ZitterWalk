"""Coins for DTQW.

Available standard coins are:
    - grover
    - fourier
    - hadamard
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Coin definitions

def hadamard(d=2):
    """Hadamard coin of dimension d (must be a power of 2).

    For d = 2 it is the standard Hadamard matrix, for d = 2**k it is the
    tensor product of k Hadamards.
    """

    assert 2 ** int(np.log2(d)) == d, "Hadamard coin requires dimension to be a power of 2."

    h1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    h = np.array([[1]], dtype=complex)

    for _ in range(int(np.log2(d))):
        h = np.kron(h, h1)
    return h


def grover(d):
    """Grover coin of dimension d"""
    G = np.full((d, d), 2.0 / d, dtype=complex)
    G[np.diag_indices(d)] -= 1.0
    return G


def fourier(d):
    """Fourier coin of dimension d"""
    j, k = np.meshgrid(np.arange(d), np.arange(d))
    omega = np.exp(2j * np.pi / d)
    return omega ** (j * k) / np.sqrt(d)

COINS = {
    "hadamard": hadamard,
    "grover": grover,
    "fourier": fourier,
}

# -----------------------------------------------------------------------------
# Coin operator

def _resolve_coin(coin, d):
    """Return the coin matrix for a node of degree d.

    Args:
        coin: either name of a standard coin, a callable, or a fixed matrix of size (d, d).
    Returns:
        A (d, d) coin matrix.
    """
    if isinstance(coin, str):
        try:
            factory = COINS[coin]
        except KeyError:
            raise ValueError(
                f"Unknown coin {coin!r}. Options: {sorted(COINS)}"
            )
        return factory(d)
    
    if callable(coin):
        return np.asarray(coin(d), dtype=complex)
    
    matrix = np.asarray(coin, dtype=complex)
    if matrix.shape != (d, d):
        raise ValueError(
            f"The coin matrix has shape {matrix.shape}, but this node has "
            f"degree {d}. Use a name or a callable for mixed degrees."
        )
    
    return matrix


def build_coin_operator(graph, coin="grover"):
    """Assemble the graph's global coin operator.

    Args:
        graph: a Graph object
        coin: either name of a standard coin, a callable, or a fixed matrix of size (d, d) 
              for all nodes of degree d.
    Returns:
        A (A, A) coin operator, where A is the number of arcs in the graph.
    """
    graph._ensure_arcs()
    A = graph.n_arcs
    C = np.zeros((A, A), dtype=complex)
    for nid in graph.nodes:
        idxs = graph.node_arcs[nid]
        d = len(idxs)
        if d == 0:
            continue  # isolated node
        block = _resolve_coin(coin, d)
        C[np.ix_(idxs, idxs)] = block
    return C
