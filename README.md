# ZitterWalk

<p align="center">
  <img src="assets/cycle_nodes.gif" alt="Quantum walk on a cycle, nodes colored by probability" width="380">
</p>

<p align="center">
  <em>A simple, dependency-light package to simulate <strong>Discrete-Time Quantum Walks (DTQW)</strong>,
  built for studying and experimenting.</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="pyproject.toml"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="#tests"><img alt="Dependencies" src="https://img.shields.io/badge/deps-numpy%20%2B%20matplotlib-informational.svg"></a>
</p>

The package is called **`zitterwalk`** — named after *Zitterbewegung* ("trembling
motion"), the jittery back-and-forth of a relativistic particle that DTQWs
reproduce at the discrete level. It only depends on `numpy` and `matplotlib`.

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

## What it looks like

The canonical example — a Hadamard walk on a line — spreads **ballistically**
(`O(t)`) instead of diffusing like a classical random walk (`O(√t)`), giving
the characteristic "two-horn" distribution:

<p align="center">
  <img src="assets/line_hadamard.png" alt="Hadamard walk: two-horn distribution, time evolution and quantum vs classical comparison" width="820">
</p>

An animated version of the same walk, with an adaptive y-axis so the spread
stays visible over time:

<p align="center">
  <img src="assets/line_walk.gif" alt="Animated Hadamard walk on a line" width="560">
</p>

Adding an electric field along the line reveals **Bloch oscillations** and
**dynamical localization**, depending on whether the field is a rational or
irrational multiple of `2π`:

<p align="center">
  <img src="assets/bloch_oscillations.png" alt="Electric quantum walk: free spreading vs Bloch oscillations vs localization" width="820">
</p>

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

## Install

```bash
pip install -e .
```

## Quick use

```python
from zitterwalk import Graph, Walker, DiscreteTimeWalk

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
