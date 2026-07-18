"""The discrete-time quantum walk engine and its observables."""

from __future__ import annotations

import numpy as np

from .coin import build_coin_operator, marked_coins, _as_node_list
from .walker import Walker


class DiscreteTimeWalk:
    """Discrete-time quantum walk on a graph.

    Attributes:
        graph: the graph (Graph).
        coin: the coin specification.
        shift: the shift specification.
        field: the electric field (phase potential), if any.
        marked: nodes carrying a special coin.
        coin_default: coin for nodes missing from a coin dict.
    """

    # ------------------------------------------------------------------
    # Construction

    def __init__(self, graph, coin="grover", shift="flip_flop", field=None, coin_default="grover", coin_schedule=None, absorbing=None):
        graph._ensure_arcs()
        self.graph = graph
        self.coin = coin
        self.shift = shift
        self.field = field
        self.coin_default = coin_default
        self.marked = None
        self._schedule = coin_schedule
        self.S = self._build_shift(graph, shift)
        self.phases = self._build_field_phases(graph, field)
        self._absorb_idx = self._resolve_absorbing(graph, absorbing)
        if coin_schedule is None:
            self.C = build_coin_operator(graph, coin, default=coin_default)
            self.U = self._assemble(self.C)
        else:
            self.C = None
            self.U = None

    @classmethod
    def search(cls, graph, marked, marker=None, coin="grover", **kwargs):
        """Build a quantum-search walk with a marker coin on the marked vertices.

        Args:
            graph: the graph to search (Graph).
            marked: marked vertex id, or a list of them.
            marker: coin for the marked vertices (defaults to -I).
            coin: coin for the unmarked vertices.
            **kwargs: forwarded to the constructor.

        Returns:
            The search DiscreteTimeWalk.
        """
        coins = marked_coins(graph, marked, marker=marker, default=coin)
        walk = cls(graph, coin=coins, coin_default=coin, **kwargs)
        walk.marked = _as_node_list(marked)
        return walk

    def _assemble(self, C):
        """Fold coin, shift and field into a single dense step operator.

        Args:
            C: the coin operator.

        Returns:
            The (A, A) step operator.
        """
        U = self.S @ C
        if self.phases is not None:
            U = self.phases[:, None] * U
        return U

    @staticmethod
    def _build_shift(graph, shift="flip_flop"):
        """Permutation matrix S of the shift operator.

        Args:
            graph: the graph (Graph).
            shift: the shift specification.

        Returns:
            The (A, A) permutation matrix.
        """
        perm = DiscreteTimeWalk._shift_permutation(graph, shift)
        A = graph.n_arcs
        S = np.zeros((A, A), dtype=complex)
        S[perm, np.arange(A)] = 1.0
        return S

    @staticmethod
    def _shift_permutation(graph, shift):
        """Resolve the shift specification to a validated arc permutation.

        Args:
            graph: the graph (Graph).
            shift: 'flip_flop', 'moving', a permutation array, or a callable.

        Returns:
            The arc permutation (int array).
        """
        graph._ensure_arcs()
        A = graph.n_arcs
        if callable(shift):
            shift = shift(graph)
        if isinstance(shift, str):
            key = shift.lower().replace("-", "_")
            if key in ("flip_flop", "flipflop"):
                perm = np.asarray(graph.flip, dtype=int)
            elif key in ("moving", "walk", "directed"):
                perm = DiscreteTimeWalk._moving_shift_permutation(graph)
            else:
                raise ValueError(
                    f"Unknown shift {shift!r}. Options: 'flip_flop', 'moving', "
                    "or an arc-permutation array / callable graph -> array."
                )
        else:
            perm = np.asarray(shift, dtype=int)
        if perm.shape != (A,):
            raise ValueError(
                f"shift permutation must have length {A} (n_arcs), got "
                f"{perm.shape}."
            )
        if not np.array_equal(np.sort(perm), np.arange(A)):
            raise ValueError(
                "shift must be a permutation of the arc indices 0..n_arcs-1."
            )
        return perm

    @staticmethod
    def _moving_shift_permutation(graph):
        """Arc permutation of the moving (keep going straight) shift.

        Args:
            graph: the graph (Graph).

        Returns:
            The arc permutation (int array).
        """
        graph._ensure_arcs()
        perm = np.empty(graph.n_arcs, dtype=int)
        coord = graph.node_coord

        def keydir(vec):
            return tuple(np.round(vec, 6))

        for v in graph.nodes:
            nbrs = graph.neighbors(v)
            d = len(nbrs)
            if d == 0:
                continue
            if d == 1:
                assign = {nbrs[0]: nbrs[0]}  # reflect
            elif d == 2:
                a, b = nbrs
                assign = {a: b, b: a}
            else:
                out_by_dir = {}
                for w in nbrs:
                    out_by_dir.setdefault(keydir(coord(w) - coord(v)), w)
                used, assign, unmatched = set(), {}, []
                for u in nbrs:
                    w = out_by_dir.get(keydir(coord(v) - coord(u)))
                    if w is not None and w not in used:
                        assign[u] = w
                        used.add(w)
                    else:
                        unmatched.append(u)
                leftover = [w for w in nbrs if w not in used]
                for u, w in zip(unmatched, leftover):
                    assign[u] = w

            for u, w in assign.items():
                perm[graph.arc_index[(u, v)]] = graph.arc_index[(v, w)]
        return perm

    @staticmethod
    def _build_field_phases(graph, field):
        """Per-arc phases e^{-i phi(x)} of a static electric field.

        Args:
            graph: the graph (Graph).
            field: None, a number (constant field), or a callable node -> potential.

        Returns:
            A length-n_arcs complex array, or None if there is no field.
        """
        if field is None:
            return None
        potential = field if callable(field) else (lambda node: field * node)
        return np.array([np.exp(-1j * potential(u)) for (u, _) in graph.arcs])

    @staticmethod
    def _resolve_absorbing(graph, absorbing):
        """Arc indices whose amplitude is projected out each step.

        Args:
            graph: the graph (Graph).
            absorbing: absorbing node id, or a list of them.

        Returns:
            An int array of arc indices, or None.
        """
        if absorbing is None:
            return None
        idxs = []
        for nid in _as_node_list(absorbing):
            idxs.extend(graph.node_arcs[nid])
        return np.array(sorted(idxs), dtype=int)

    def _operator_at(self, t):
        """Step operator at time t, rebuilt only for a time-dependent coin.

        Args:
            t: time index (int).

        Returns:
            The (A, A) step operator.
        """
        if self._schedule is None:
            return self.U
        C = build_coin_operator(self.graph, self._schedule(t),
                                default=self.coin_default)
        return self._assemble(C)

    # ------------------------------------------------------------------
    # Evolution

    def step(self, walker, times=1, t0=0):
        """Apply the given number of steps and return the new state.

        Args:
            walker: the current state (Walker).
            times: number of steps (int).
            t0: time index of the first step, only relevant for a time-dependent coin.

        Returns:
            The evolved Walker (not normalized when there are absorbing nodes).
        """
        psi = walker.psi
        for k in range(times):
            psi = self._operator_at(t0 + k) @ psi
            if self._absorb_idx is not None:
                psi = psi.copy()
                psi[self._absorb_idx] = 0.0
        return Walker(self.graph, psi)

    def run(self, walker, steps):
        """Run the walk and return every state from t=0 to t=steps.

        Args:
            walker: the initial state (Walker).
            steps: number of steps (int).

        Returns:
            A list of steps + 1 Walker states, index 0 being the initial state.
        """
        states = [walker.copy()]
        current = walker
        for t in range(steps):
            current = self.step(current, times=1, t0=t)
            states.append(current)
        return states

    # ------------------------------------------------------------------
    # Observables distributions

    def probabilities(self, walker):
        """Probability distribution over the nodes.

        Args:
            walker: the current state (Walker).

        Returns:
            A length-n_nodes probability array.
        """
        graph = self.graph
        p = np.zeros(graph.n_nodes)
        prob_arcs = np.abs(walker.psi) ** 2
        for nid in graph.nodes:
            idxs = graph.node_arcs[nid]
            p[graph.node_position(nid)] = prob_arcs[idxs].sum()
        return p

    def probability_dict(self, walker):
        """Node probabilities as a dict {node: probability}.

        Args:
            walker: the current state (Walker).

        Returns:
            A dict mapping node id to probability.
        """
        p = self.probabilities(walker)
        return {nid: p[self.graph.node_position(nid)] for nid in self.graph.nodes}

    # ------------------------------------------------------------------
    # Observables

    def mean_position(self, walker):
        """Expected position, a float in 1-D or a coordinate array otherwise.

        Args:
            walker: the current state (Walker).

        Returns:
            The mean position (float or array).
        """
        p = self.probabilities(walker)
        total = p.sum()
        coords = self.graph.coordinates
        if total <= 0:
            mean = np.zeros(coords.shape[1])
        else:
            mean = (p[:, None] * coords).sum(axis=0) / total
        return float(mean[0]) if mean.size == 1 else mean

    def variance(self, walker):
        """Mean squared displacement of the position distribution.

        Args:
            walker: the current state (Walker).

        Returns:
            The spread scalar (float).
        """
        p = self.probabilities(walker)
        total = p.sum()
        if total <= 0:
            return 0.0
        coords = self.graph.coordinates
        mean = (p[:, None] * coords).sum(axis=0) / total
        sq = ((coords - mean) ** 2).sum(axis=1)
        return float((p * sq).sum() / total)

    def std(self, walker):
        """Standard deviation of the position, the square root of the variance.

        Args:
            walker: the current state (Walker).

        Returns:
            The standard deviation (float).
        """
        return float(np.sqrt(self.variance(walker)))

    def return_probability(self, walker, origin):
        """Probability of finding the walker back at a node.

        Args:
            walker: the current state (Walker).
            origin: node id to check.

        Returns:
            The return probability (float).
        """
        return float(self.probabilities(walker)[self.graph.node_position(origin)])

    # ------------------------------------------------------------------
    # Observables localization and spreading diagnostics

    def participation_ratio(self, walker):
        """Participation ratio, roughly how many nodes are occupied.

        Args:
            walker: the current state (Walker).

        Returns:
            The participation ratio (float).
        """
        p = self.probabilities(walker)
        total = p.sum()
        if total <= 0:
            return 0.0
        p = p / total
        return float(1.0 / (p ** 2).sum())

    def coin_entropy(self, walker):
        """Coin-position entanglement entropy in bits, for a 2-regular walk.

        Args:
            walker: the current state (Walker).

        Returns:
            The von Neumann entropy in bits, in [0, 1].

        Raises:
            ValueError: if the walk is not 2-regular.
        """
        g = self.graph
        max_deg = max((len(g.node_arcs[n]) for n in g.nodes), default=0)
        if max_deg != 2:
            raise ValueError(
                "coin_entropy needs a 2-regular walk (e.g. Graph.line / "
                f"Graph.cycle); this graph has nodes of degree {max_deg}."
            )
        psi = walker.psi
        rho = np.zeros((2, 2), dtype=complex)
        for nid in g.nodes:
            idxs = g.node_arcs[nid]
            if len(idxs) != 2:
                continue
            heads = [g.arcs[i][1] for i in idxs]
            order = sorted(range(2), key=lambda k: g.node_coord(heads[k])[0])
            a = np.array([psi[idxs[order[0]]], psi[idxs[order[1]]]])
            rho += np.outer(a, a.conj())
        tr = rho.trace().real
        if tr <= 0:
            return 0.0
        rho /= tr
        evals = np.linalg.eigvalsh(rho).real
        evals = evals[evals > 1e-15]
        return float(-(evals * np.log2(evals)).sum())

    # ------------------------------------------------------------------
    # Observables absorbing walks and search

    def survival_probability(self, walker):
        """Total probability still on the graph, 1 without absorption.

        Args:
            walker: the current state (Walker).

        Returns:
            The surviving probability (float).
        """
        return float((np.abs(walker.psi) ** 2).sum())

    def success_probability(self, walker, marked=None):
        """Probability on the marked vertices, the search success amplitude.

        Args:
            walker: the current state (Walker).
            marked: marked vertices (defaults to those the walk searches for).

        Returns:
            The success probability (float).

        Raises:
            ValueError: if no marked vertices are available.
        """
        if marked is None:
            marked = self.marked
        if marked is None:
            raise ValueError("no marked vertices; pass marked=... explicitly.")
        p = self.probabilities(walker)
        return float(sum(p[self.graph.node_position(m)] for m in _as_node_list(marked)))

    # ------------------------------------------------------------------
    # Observables over a trajectory

    def variance_evolution(self, states):
        """Variance over a trajectory of states.

        Args:
            states: trajectory from run.

        Returns:
            An array of variances.
        """
        return np.array([self.variance(s) for s in states])

    def std_evolution(self, states):
        """Standard deviation sigma(t) over a trajectory of states.

        Args:
            states: trajectory from run.

        Returns:
            An array of standard deviations.
        """
        return np.sqrt(self.variance_evolution(states))

    def mean_position_evolution(self, states):
        """Mean position <x>(t) over a trajectory of states.

        Args:
            states: trajectory from run.

        Returns:
            An array of mean positions.
        """
        return np.array([self.mean_position(s) for s in states])

    def return_probability_evolution(self, states, origin):
        """Return probability to a node over a trajectory of states.

        Args:
            states: trajectory from run.
            origin: node id to check.

        Returns:
            An array of return probabilities.
        """
        return np.array([self.return_probability(s, origin) for s in states])

    def limiting_distribution(self, states):
        """Time-averaged (Cesaro) distribution over a trajectory of states.

        Args:
            states: trajectory from run.

        Returns:
            A length-n_nodes probability array.
        """
        P = np.array([self.probabilities(s) for s in states])
        return P.mean(axis=0)

    def mixing_time(self, states, eps=0.05):
        """Empirical mixing time of the time-averaged distribution.

        Args:
            states: trajectory from run.
            eps: total-variation tolerance.

        Returns:
            The first step within eps of the limiting distribution, or None.
        """
        P = np.array([self.probabilities(s) for s in states])
        running = np.cumsum(P, axis=0) / np.arange(1, len(P) + 1)[:, None]
        limit = P.mean(axis=0)
        tv = 0.5 * np.abs(running - limit).sum(axis=1)
        below = np.nonzero(tv < eps)[0]
        return int(below[0]) if below.size else None

    # ------------------------------------------------------------------
    # Spectral analysis

    def spectrum(self):
        """Quasi-energies of the walk, the phases of the eigenvalues of U.

        Returns:
            The sorted quasi-energies in (-pi, pi].

        Raises:
            ValueError: if the coin is time-dependent.
        """
        if self.U is None:
            raise ValueError("spectrum is undefined for a time-dependent coin.")
        eigs = np.linalg.eigvals(self.U)
        return np.sort(np.angle(eigs))

    # ------------------------------------------------------------------
    # Measurement

    def measure(self, walker, seed=None, rng=None, collapse=False):
        """Sample a node from the walker's distribution, a position measurement.

        Args:
            walker: the current state (Walker).
            seed: seed for a fresh generator (int).
            rng: explicit numpy Generator.
            collapse: whether to also return the post-measurement state.

        Returns:
            The sampled node id, or (node, walker) when collapse is True.

        Raises:
            ValueError: if the state is fully absorbed or zero.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        p = self.probabilities(walker)
        total = p.sum()
        if total <= 0:
            raise ValueError("cannot measure a fully absorbed / zero state.")
        nodes = self.graph.nodes
        node = nodes[rng.choice(len(nodes), p=p / total)]
        if not collapse:
            return node
        psi = np.zeros_like(walker.psi)
        idxs = self.graph.node_arcs[node]
        psi[idxs] = walker.psi[idxs]
        norm = np.linalg.norm(psi)
        return node, Walker(self.graph, psi / norm)

    def __repr__(self):
        kind = "time-dependent" if self.U is None else f"n_arcs={self.graph.n_arcs}"
        shift = "" if isinstance(self.shift, str) and self.shift == "flip_flop" \
            else f"shift={self.shift!r}, "
        return (
            f"DiscreteTimeWalk(coin={self.coin!r}, {shift}"
            f"field={self.field!r}, {kind})"
        )
