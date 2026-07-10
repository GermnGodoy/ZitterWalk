# quantum_walks

A simple package to simulate **Discrete-Time Quantum Walks (DTQW)**, meant for
studying and experimenting. It only depends on `numpy` and `matplotlib`.

## Core idea

The state of a DTQW lives on the **arcs** (directed edges) of the graph. Each
edge `{u, v}` gives two arcs, `(u,v)` and `(v,u)`; an arc `(u,v)` means "the
walker is at `u` heading towards `v`". On that basis:

- **Coin `C`**: acts block-wise, one block per node. At a node of degree `d` it
  is a `d x d` unitary that shuffles the directions. This is *where* the coin
  enters.
- **Shift `S`** (flip-flop): `S|u→v> = |v→u>`, a simple permutation of arcs.
- **One step**: `U = S · C` (shuffle and move). It is unitary ⇒ conserves the
  norm.

This arc formulation generalizes to **any graph** (not just the line).

## Structure

| Module | Responsibility |
|--------|----------------|
| `graph.py`  | Topology: `Graph`, `Node`, `Edge` and the arc representation. |
| `coin.py`   | Coins: `hadamard`, `grover`, `fourier` + block assembly. |
| `walker.py` | `Walker`: the initial quantum state. |
| `walk.py`   | `DiscreteTimeWalk`: evolution engine and observables. |
| `viz.py`    | Visualization (kept separate): distribution, graph, evolution. |

The three responsibilities —**topology**, **state** and **dynamics**— are
separated on purpose.

## Quick use

```python
from quantum_walks import Graph, Walker, DiscreteTimeWalk

g = Graph.line(201)                                 # line of 201 nodes
w = Walker.at_node(g, 100, coin_state=[1, 1j])      # symmetric start
walk = DiscreteTimeWalk(g, coin="hadamard")

final = walk.step(w, times=80)                      # 80 steps
p = walk.probabilities(final)                       # per-node distribution
```

Factory graphs: `Graph.line(n)`, `Graph.cycle(n)`, `Graph.complete(n)`,
`Graph.grid(rows, cols)`. You can also build your own with `add_edge`.

Coins: `"grover"` (any degree, default), `"fourier"` (any degree), `"hadamard"`
(power-of-two degree). Or pass your own callable / matrix.

## Examples

```bash
python examples/line_hadamard.py        # static "two-horn" distribution
python examples/line_animation.py       # animated curve + bars + shaded area (GIF)
python examples/cycle_animation.py      # same, on a small cycle (self-interference)
python examples/cycle_nodes_animation.py  # nodes on a ring, colored by population
python examples/bloch_oscillations.py   # electric field: free vs Bloch vs localized (heatmap)
python examples/bloch_animation.py      # electric field: the walker "breathes" (GIF)
python examples/bloch_compare.py        # rational vs irrational field, side by side (GIF)
```

`line_hadamard.py` reproduces the "two-horn" distribution of the Hadamard walk
and compares it with classical diffusion (the quantum one spreads as `O(t)`,
not `O(√t)`). The others save animated GIFs via `viz.animate`, which offers
three styles: `kind="line"` (curve + bars + fill, adaptive y-axis), `"bar"`,
and `"graph"` (nodes colored by probability).

## Tests

```bash
python tests/test_walk.py        # or: python -m pytest
```

They check coin unitarity, norm conservation, that probabilities sum to 1, and
the characteristic ballistic spreading.
