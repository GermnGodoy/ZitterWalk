"""Graph topology for DTQW.

Graph describes the structure (nodes and edges) and
derives from it the arc (directed-edge) representation that the walk
dynamics needs. It knows nothing about amplitudes or coins, those are the
responsibility of other modules.

The quantum state of a DTQW lives on the arcs of the graph. Each undirected
edge {u, v} gives rise to two (directed) arcs, (u, v) and (v, u). An arc
(u, v) is read as "the walker is at u about to move towards v".
"""

from __future__ import annotations

from itertools import product


class Node:
    """A node of the graph with its own id."""

    __slots__ = ("id", "neighbors")

    def __init__(self, node_id):
        self.id = node_id
        self.neighbors = []  # neighbor ids, in insertion order

    @property
    def degree(self):
        return len(self.neighbors)

    def __repr__(self):
        return f"Node({self.id!r}, degree={self.degree})"


class Edge:
    """An undirected edge between two nodes."""

    __slots__ = ("u", "v")

    def __init__(self, u, v):
        self.u = u
        self.v = v

    def endpoints(self):
        return (self.u, self.v)

    def __repr__(self):
        return f"Edge({self.u!r}, {self.v!r})"


class Graph:
    """Undirected graph.
    
    Attributes:
        - arcs: list of arcs (tail, head) with a fixed index.
        - arc_index: dict (tail, head) -> index.
        - flip: array where flip[i] is the index of the reverse arc of i.
        - node_arcs: dict node -> [indices of arcs leaving it].
    """

    def __init__(self):
        self._nodes = {}       # id -> Node (preserves insertion order)
        self._edges = []       # list of Edge
        self._dirty = True     
        self.arcs = []
        self.arc_index = {}
        self.flip = []
        self.node_arcs = {}

    # ------------------------------------------------------------------ 
    # Construction

    def add_node(self, node_id):
        """Add a node and return its instance"""
        if node_id not in self._nodes:
            self._nodes[node_id] = Node(node_id)
            self._dirty = True
        return self._nodes[node_id]

    def add_edge(self, u, v):
        """Add an undirected edge {u, v} (creates the nodes if missing)."""
        if u == v:
            raise ValueError(f"Self-loops are not allowed: {u!r}")
        self.add_node(u)
        self.add_node(v)
        if v in self._nodes[u].neighbors:
            return  # already exists, do not duplicate
        self._nodes[u].neighbors.append(v)
        self._nodes[v].neighbors.append(u)
        self._edges.append(Edge(u, v))
        self._dirty = True

    # ------------------------------------------------------------------ 
    # Properties 

    @property
    def nodes(self):
        """List of node ids in insertion order."""
        return list(self._nodes.keys())

    @property
    def edges(self):
        return list(self._edges)

    @property
    def n_nodes(self):
        return len(self._nodes)

    @property
    def n_arcs(self):
        self._ensure_arcs()
        return len(self.arcs)

    def node(self, node_id):
        return self._nodes[node_id]

    def neighbors(self, node_id):
        return list(self._nodes[node_id].neighbors)

    def degree(self, node_id):
        return self._nodes[node_id].degree

    def node_position(self, node_id):
        """Index of the node in canonical order (useful for probability arrays)."""
        self._ensure_arcs()
        return self._node_pos[node_id]

    # ------------------------------------------------------------------ 
    # Arcs

    def _ensure_arcs(self):
        if self._dirty:
            self._build_arcs()

    def _build_arcs(self):
        """Build the arc cache from the current structure."""
        arcs = []
        arc_index = {}
        node_arcs = {nid: [] for nid in self._nodes}

        # Arcs leaving the same node stay contiguous
        for nid, node in self._nodes.items():
            for v in node.neighbors:
                idx = len(arcs)
                arcs.append((nid, v))
                arc_index[(nid, v)] = idx
                node_arcs[nid].append(idx)

        # Index of the reverse arc, for the flip-flop shift.
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
        """Path of n nodes (1D line). With reflecting boundaries."""
        g = cls()
        for i in range(n):
            g.add_node(i)
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        return g

    @classmethod
    def cycle(cls, n):
        """Cycle of n nodes (line with periodic boundary conditions)."""
        g = cls()
        for i in range(n):
            g.add_node(i)
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        return g

    @classmethod
    def complete(cls, n):
        """Complete graph K_n (every node connected to every other)."""
        g = cls()
        for i in range(n):
            g.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j)
        return g

    @classmethod
    def grid(cls, rows, cols):
        """2D grid rows x cols. Nodes are tuples (row, col)."""
        g = cls()
        for r, c in product(range(rows), range(cols)):
            g.add_node((r, c))
        for r, c in product(range(rows), range(cols)):
            if c + 1 < cols:
                g.add_edge((r, c), (r, c + 1))
            if r + 1 < rows:
                g.add_edge((r, c), (r + 1, c))
        return g

    def __repr__(self):
        return f"Graph(n_nodes={self.n_nodes}, n_edges={len(self._edges)})"
