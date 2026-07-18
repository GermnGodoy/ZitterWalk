"""Coins for DTQW.

Available standard coins are grover, fourier, hadamard and rotation(theta).

A coin specification is anything the walk knows how to turn into a per-node
coin matrix: a name from the available coins, a callable d -> (d, d) matrix, a
fixed (d, d) matrix, or a dict {node: spec}.
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Coin definitions

def hadamard(d=2):
    """Hadamard coin of dimension d, the tensor product of log2(d) qubit Hadamards.

    Args:
        d: coin dimension, a power of 2 (int).

    Returns:
        The (d, d) Hadamard matrix.
    """
    if 2 ** int(np.log2(d)) != d:
        raise ValueError("Hadamard coin requires dimension to be a power of 2")

    h1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    h = np.array([[1]], dtype=complex)

    for _ in range(int(np.log2(d))):
        h = np.kron(h, h1)
    return h


def grover(d):
    """Grover diffusion coin of dimension d.

    Args:
        d: coin dimension (int).

    Returns:
        The (d, d) Grover matrix.
    """
    G = np.full((d, d), 2.0 / d, dtype=complex)
    G[np.diag_indices(d)] -= 1.0
    return G


def fourier(d):
    """Fourier (discrete Fourier transform) coin of dimension d.

    Args:
        d: coin dimension (int).

    Returns:
        The (d, d) Fourier matrix.
    """
    j, k = np.meshgrid(np.arange(d), np.arange(d))
    omega = np.exp(2j * np.pi / d)
    return omega ** (j * k) / np.sqrt(d)


def rotation(theta):
    """Tunable degree-2 coin of angle theta.

    Args:
        theta: rotation angle (radians).

    Returns:
        A factory d -> (d, d) coin matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    mat = np.array([[c, s], [s, -c]], dtype=complex)

    def factory(d):
        if d == 1:
            return np.array([[1.0]], dtype=complex)
        if d != 2:
            raise ValueError(
                f"Rotation coin is only defined for degree 2 (got {d}) "
            )
        return mat

    return factory


def su2(theta, axis="x"):
    """SU(2) rotation coin of angle theta about a Pauli axis.

    Args:
        theta: rotation angle, the Dirac "mass" (radians).
        axis: Pauli axis, one of "x", "y", "z".

    Returns:
        A factory d -> (d, d) coin matrix.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    mats = {
        "x": np.array([[c, 1j * s], [1j * s, c]], dtype=complex),
        "y": np.array([[c, s], [-s, c]], dtype=complex),
        "z": np.array([[c + 1j * s, 0], [0, c - 1j * s]], dtype=complex),
    }
    try:
        mat = mats[axis]
    except KeyError:
        raise ValueError(f"axis must be one of 'x', 'y', 'z' (got {axis!r}).")

    def factory(d):
        if d == 1:
            return np.array([[1.0]], dtype=complex)
        if d != 2:
            raise ValueError(
                f"su2 coin is only defined for degree 2 (got {d})"
            )
        return mat

    return factory


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
        coin: coin name, a callable, or a fixed (d, d) matrix.
        d: node degree (int).

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


def build_coin_operator(graph, coin="grover", default="grover"):
    """Assemble the graph's global coin operator as a block-diagonal matrix.

    Args:
        graph: the graph (Graph).
        coin: coin specification.
        default: coin for nodes missing from a coin dict.

    Returns:
        The (A, A) coin operator, A the number of arcs.
    """
    graph._ensure_arcs()
    A = graph.n_arcs
    C = np.zeros((A, A), dtype=complex)
    per_node = isinstance(coin, dict)
    for nid in graph.nodes:
        idxs = graph.node_arcs[nid]
        d = len(idxs)
        if d == 0:
            continue  # isolated
        spec = coin.get(nid, default) if per_node else coin
        block = _resolve_coin(spec, d)
        C[np.ix_(idxs, idxs)] = block
    return C


# -----------------------------------------------------------------------------
# Inhomogeneous-coin helpers

def random_coins(graph, seed=None, rng=None, low=0.0, high=2 * np.pi, base="grover"):
    """Per-node random coins as a dict, for static coin disorder.

    Args:
        graph: the graph (Graph).
        seed: seed for a fresh generator (int).
        rng: explicit numpy Generator.
        low: lowest rotation angle (radians).
        high: highest rotation angle (radians).
        base: base coin for nodes of degree other than 2.

    Returns:
        A dict {node: coin}.
    """
    graph._ensure_arcs()
    if rng is None:
        rng = np.random.default_rng(seed)
    coins = {}
    for nid in graph.nodes:
        d = len(graph.node_arcs[nid])
        if d == 0:
            continue
        if d == 2:
            theta = rng.uniform(low, high)
            c, s = np.cos(theta), np.sin(theta)
            coins[nid] = np.array([[c, s], [s, -c]], dtype=complex)
        else:
            phases = np.exp(1j * rng.uniform(0.0, 2 * np.pi, size=d))
            coins[nid] = phases[:, None] * _resolve_coin(base, d)
    return coins


def marked_coins(graph, marked, marker=None, default="grover"):
    """Coin dict for quantum search, with a special coin on the marked nodes.

    Args:
        graph: the graph being searched (Graph).
        marked: a marked node id, or a list of them.
        marker: coin for the marked nodes (defaults to -I).
        default: coin for the unmarked nodes.

    Returns:
        A dict {node: coin}, unmarked nodes served by default.
    """
    if marker is None:
        marker = lambda d: -np.identity(d, dtype=complex)
    return {m: marker for m in _as_node_list(marked)}


def _as_node_list(marked):
    """Normalize marked into a list of node ids.

    Args:
        marked: a node id or a collection of them.

    Returns:
        A list of node ids.
    """
    if isinstance(marked, (list, set, frozenset, np.ndarray)):
        return list(marked)
    return [marked]
