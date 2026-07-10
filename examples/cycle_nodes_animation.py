"""Node-colored quantum walk on a small cycle.

The nodes are laid out on a ring and their color is animated according to how
"populated" they are (their probability). Run::

    python examples/cycle_nodes_animation.py

Saves the animation to 'cycle_nodes.gif'.
"""

from quantum_walks import Graph, Walker, DiscreteTimeWalk
from quantum_walks import viz


def main():
    n = 24           # small cycle so the nodes are clearly visible
    start = 0
    t_max = 60

    g = Graph.cycle(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")

    w = Walker.at_node(g, start, coin_state=[1, 1j])
    states = walk.run(w, t_max)

    # kind="graph": nodes colored by probability, on the circular layout.
    viz.animate(walk, states, save_path="cycle_nodes.gif", kind="graph",
                fps=10, node_size=650)
    print("Animation saved to cycle_nodes.gif")


if __name__ == "__main__":
    main()
