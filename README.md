# ZitterWalk

<p align="center">
  <img src="assets/cycle_nodes.gif" alt="Quantum walk on a cycle, nodes colored by probability" width="380">
</p>

<p align="center">
  <em>A small, dependency-light package to simulate <strong>Discrete-Time Quantum Walks (DTQW)</strong> — built for studying and experimenting.</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="pyproject.toml"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <img alt="Dependencies" src="https://img.shields.io/badge/deps-numpy%20%2B%20matplotlib-informational.svg">
  <a href="https://germngodoy.github.io/ZitterWalk/"><img alt="Documentation" src="https://img.shields.io/badge/docs-online-5b5bd6.svg"></a>
  <a href="https://github.com/GermnGodoy/ZitterWalk/actions/workflows/docs.yml"><img alt="Docs build" src="https://github.com/GermnGodoy/ZitterWalk/actions/workflows/docs.yml/badge.svg"></a>
</p>

## What is a quantum walk?

A classical **random walk** is a movement where at each step
the walker flips a coin and moves left or right. After $t$ steps its position
spreads as

$$\sigma(t) \sim \sqrt{t}.$$

A **quantum walk** replaces the coin flip and the move by *unitary* operators
acting on a superposition of states. The walker is no longer at a single site:
it is a wave of complex amplitudes that can **interfere** with itself. The state
lives on the **arcs** (directed edges) of a graph, and one step is

$$U = S\,C,$$

where the **coin** $C$ mixes the directions at each node and the **shift** $S$
moves amplitude along the arcs. Since $U$ is unitary, probability is conserved.

Interference changes the physics completely. Instead of diffusing, the quantum
walker spreads **ballistically**,

$$\sigma(t) \sim t,$$

quadratically faster than its classical brother. This makes them really suitable
for some algorithms, specially search ones. They are theorically interesting as well,
as they reproduce Dirac and Schrödinger equations in certain limits.

<p align="center">
  <img src="assets/line_hadamard.png" alt="Hadamard walk: two-horn distribution, time evolution and quantum vs classical comparison" width="820">
</p>

The signature distribution of a Hadamard walk on a line, probability
piles up at the ballistic fronts near $\pm t/\sqrt{2}$.

## Install

Requires Python 3.10+ and depends only on `numpy` and `matplotlib`.

```bash
pip install -e .
```

## Quick use

```python
from zitterwalk import Graph, Walker, DiscreteTimeWalk

g = Graph.line(201)                              # line of 201 nodes
w = Walker.at_node(g, 100, coin_state=[1, 1j])   # symmetric start at the center
walk = DiscreteTimeWalk(g, coin="hadamard")

final = walk.step(w, times=80)                   # evolve 80 steps
p = walk.probabilities(final)                    # per-node probability distribution
walk.std(final)                                  # spread ~ t (ballistic)
```

A few basics you can reach for:

- **Graphs**: `Graph.line(n)`, `Graph.cycle(n)`, `Graph.complete(n)`,
  `Graph.grid(rows, cols)`, `Graph.hypercube(dim)`, or build your own with `add_edge`.
- **Coins**: `"hadamard"`, `"grover"`, `"fourier"`, and the tunable `rotation(θ)`
  and `su2(θ)`.
- **Initial states**: `Walker.at_node`, `Walker.uniform`, `Walker.superposition`,
  `Walker.gaussian`.
- **Observables**: `probabilities`, `mean_position`, `std`, `participation_ratio`,
  `coin_entropy`, and their `..._evolution` trajectory versions.
- **Visualization** (`from zitterwalk import viz`): distributions, node-link
  diagrams, heatmaps and animated GIFs.

## Examples

```bash
python examples/line_hadamard.py         # static "two-horn" distribution
python examples/line_animation.py        # animated curve + bars + shaded area (GIF)
python examples/cycle_animation.py       # same, on a small cycle (self-interference)
python examples/cycle_nodes_animation.py # nodes on a ring, colored by population
python examples/coin_dispersion.py       # rotation coin: tuning the spread velocity
```

There is also an interactive notebook,
[`examples/01_fundamentos.ipynb`](examples/01_fundamentos.ipynb), that walks
through these step by step.

## Tests

```bash
python -m pytest
```

---

<sub>This package was originally developed as part of my final degree project
(<em>Trabajo de Fin de Grado</em>) at the <strong>Universitat de València</strong>.</sub>
</content>
