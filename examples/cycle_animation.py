"""Hadamard walk on a small cycle, as a curve + bars + shaded area.

Same style as the line animation but on a cycle with a much smaller N, so the
walker wraps around and interferes with itself. Run::

    python examples/cycle_animation.py

Saves the animation to 'cycle_walk.gif'.
"""

from quantum_walks import Graph, Walker, DiscreteTimeWalk
from quantum_walks import viz


def main():
    n = 24           # small cycle
    start = 0
    t_max = 60

    g = Graph.cycle(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")

    # Symmetric start: (|left> + i|right>) / sqrt(2).
    w = Walker.at_node(g, start, coin_state=[1, 1j])

    states = walk.run(w, t_max)
    viz.animate(walk, states, save_path="cycle_walk.gif", kind="line", fps=10)
    print("Animation saved to cycle_walk.gif")


if __name__ == "__main__":
    main()
