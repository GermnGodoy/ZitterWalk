"""Graph topology.

Graph describes the structure (nodes and edges) and derives from it the arc
(directed-edge) representation that the walk dynamics needs. It knows nothing
about amplitudes or coins, those are the responsibility of other modules.

The quantum state of a DTQW lives on the arcs of the graph. Each undirected
edge {u, v} gives rise to two directed arcs, (u, v) and (v, u). An arc (u, v)
is read as "the walker is at u about to move towards v".
"""

from __future__ import annotations
from itertools import product

import numpy as np


class Node:
    """A graph node holding its id and its neighbor ids.

    Attributes:
        id: the node identifier.
        neighbors: neighbor ids, in insertion order (list).
    """

    __slots__ = ("id", "neighbors")

    def __init__(self, node_id):
        self.id = node_id
        self.neighbors = []  # insertion order

    @property
    def degree(self):
        """Number of neighbors of the node."""
        return len(self.neighbors)

    def __repr__(self):
        return f"Node({self.id!r}, degree={self.degree})"


class Edge:
    """An undirected edge between two nodes.

    Attributes:
        u: first endpoint.
        v: second endpoint.
    """

    __slots__ = ("u", "v")

    def __init__(self, u, v):
        self.u = u
        self.v = v

    def endpoints(self):
        """Return the pair of endpoints (u, v)."""
        return (self.u, self.v)

    def __repr__(self):
        return f"Edge({self.u!r}, {self.v!r})"


class Graph:
    """Undirected graph with a lazily built arc (directed-edge) representation.

    Attributes:
        arcs: list of arcs (tail, head).
        arc_index: dict (tail, head) -> index into arcs.
        flip: array where flip[i] is the index of the reverse arc of i.
        node_arcs: dict node -> indices of the arcs leaving it.
    """

    def __init__(self):
        self._nodes = {}       # id -> Node
        self._edges = []       # list of Edge
        self._dirty = True
        self._coords = {}      # id -> coordinate array
        self.arcs = []
        self.arc_index = {}
        self.flip = []
        self.node_arcs = {}

    # ------------------------------------------------------------------
    # Construction

    def add_node(self, node_id):
        """Add a node and return its instance.

        Args:
            node_id: the node identifier.

        Returns:
            The Node instance.
        """
        if node_id not in self._nodes:
            self._nodes[node_id] = Node(node_id)
            self._dirty = True
        return self._nodes[node_id]

    def add_edge(self, u, v):
        """Add an undirected edge {u, v}, creating the nodes if missing.

        Args:
            u: first endpoint.
            v: second endpoint.
        """
        self.add_node(u)
        if u == v:
            if u in self._nodes[u].neighbors:
                return
            self._nodes[u].neighbors.append(u)
            self._edges.append(Edge(u, u))
            self._dirty = True
            return
        self.add_node(v)
        if v in self._nodes[u].neighbors:
            return
        self._nodes[u].neighbors.append(v)
        self._nodes[v].neighbors.append(u)
        self._edges.append(Edge(u, v))
        self._dirty = True

    def add_self_loops(self, nodes=None):
        """Add one self-loop to every node, or to the given nodes.

        Args:
            nodes: node ids to loop (defaults to all nodes).

        Returns:
            The graph itself.
        """
        for nid in (self.nodes if nodes is None else nodes):
            self.add_edge(nid, nid)
        return self

    # ------------------------------------------------------------------
    # Properties

    @property
    def nodes(self):
        """List of node ids in insertion order."""
        return list(self._nodes.keys())

    @property
    def edges(self):
        """List of the graph's edges."""
        return list(self._edges)

    @property
    def n_nodes(self):
        """Number of nodes."""
        return len(self._nodes)

    @property
    def n_arcs(self):
        """Number of arcs (twice the number of edges)."""
        self._ensure_arcs()
        return len(self.arcs)

    def node(self, node_id):
        """Return the Node object for a given id.

        Args:
            node_id: the node identifier.

        Returns:
            The Node instance.
        """
        return self._nodes[node_id]

    def neighbors(self, node_id):
        """Neighbor ids of a node.

        Args:
            node_id: the node identifier.

        Returns:
            A list of neighbor ids.
        """
        return list(self._nodes[node_id].neighbors)

    def degree(self, node_id):
        """Degree (number of neighbors) of a node.

        Args:
            node_id: the node identifier.

        Returns:
            The degree (int).
        """
        return self._nodes[node_id].degree

    def node_position(self, node_id):
        """Index of the node in canonical order.

        Args:
            node_id: the node identifier.

        Returns:
            The position index (int).
        """
        self._ensure_arcs()
        return self._node_pos[node_id]

    # ------------------------------------------------------------------
    # Spatial coordinates

    def set_coords(self, coords):
        """Attach spatial coordinates to nodes from a dict {node: coord}.

        Args:
            coords: dict mapping node ids to coordinates.

        Returns:
            The graph itself.
        """
        for nid, xy in coords.items():
            self._coords[nid] = np.atleast_1d(np.asarray(xy, dtype=float))
        return self

    def node_coord(self, node_id):
        """Spatial coordinate of a node, as a 1-D float array.

        Args:
            node_id: the node identifier.

        Returns:
            A 1-D float coordinate array.
        """
        if node_id in self._coords:
            return self._coords[node_id]
        if isinstance(node_id, (int, float)) and not isinstance(node_id, bool):
            return np.array([float(node_id)])
        if isinstance(node_id, tuple) and all(
            isinstance(c, (int, float)) for c in node_id
        ):
            return np.asarray(node_id, dtype=float)
        return np.array([float(self.node_position(node_id))])

    @property
    def coordinates(self):
        """(n_nodes, dim) array of node coordinates, in order."""
        self._ensure_arcs()
        return np.array([self.node_coord(nid) for nid in self._nodes])

    # ------------------------------------------------------------------
    # Arcs

    def _ensure_arcs(self):
        """Rebuild the arc cache if the structure changed since last time."""
        if self._dirty:  # lazy
            self._build_arcs()

    def _build_arcs(self):
        """Build the arc cache from the current nodes and edges."""
        arcs = []
        arc_index = {}
        node_arcs = {nid: [] for nid in self._nodes}

        # arcs leaving the same node stay contiguous
        for nid, node in self._nodes.items():
            for v in node.neighbors:
                idx = len(arcs)
                arcs.append((nid, v))
                arc_index[(nid, v)] = idx
                node_arcs[nid].append(idx)

        flip = [arc_index[(v, u)] for (u, v) in arcs]

        self.arcs = arcs
        self.arc_index = arc_index
        self.flip = flip
        self.node_arcs = node_arcs
        self._node_pos = {nid: i for i, nid in enumerate(self._nodes)}
        self._dirty = False

    # ------------------------------------------------------------------
    # Examples of graphs

    @classmethod
    def line(cls, n):
        """Path of n nodes (a 1-D line).

        Args:
            n: number of nodes (int).

        Returns:
            The Graph.
        """
        g = cls()
        for i in range(n):
            g.add_node(i)
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        return g

    @classmethod
    def cycle(cls, n):
        """Cycle of n nodes (a line with periodic boundary conditions).

        Args:
            n: number of nodes (int).

        Returns:
            The Graph.
        """
        g = cls()
        for i in range(n):
            g.add_node(i)
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        return g

    @classmethod
    def complete(cls, n):
        """Complete graph K_n, every node connected to every other.

        Args:
            n: number of nodes (int).

        Returns:
            The Graph.
        """
        g = cls()
        for i in range(n):
            g.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j)
        return g

    @classmethod
    def grid(cls, rows, cols):
        """2-D grid of rows x cols nodes, with nodes as tuples (row, col).

        Args:
            rows: number of rows (int).
            cols: number of columns (int).

        Returns:
            The Graph.
        """
        g = cls()
        for r, c in product(range(rows), range(cols)):
            g.add_node((r, c))
        for r, c in product(range(rows), range(cols)):
            if c + 1 < cols:
                g.add_edge((r, c), (r, c + 1))
            if r + 1 < rows:
                g.add_edge((r, c), (r + 1, c))
        return g

    @classmethod
    def hypercube(cls, dim):
        """Hypercube graph Q_dim, edges joining ids that differ in one bit.

        Args:
            dim: number of dimensions, giving 2**dim nodes (int).

        Returns:
            The Graph.
        """
        g = cls()
        n = 1 << dim
        for i in range(n):
            g.add_node(i)
        for i in range(n):
            for b in range(dim):
                j = i ^ (1 << b)
                if i < j:
                    g.add_edge(i, j)
        return g

    def __repr__(self):
        return f"Graph(n_nodes={self.n_nodes}, n_edges={len(self._edges)})"
