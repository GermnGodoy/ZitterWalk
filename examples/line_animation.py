"""Animated Hadamard walk on the line, saved as a GIF.

Builds a line graph, places a walker at the center, runs it up to t_max and
animates the per-node probability. Run::

    python examples/line_animation.py

Saves the animation to 'line_walk.gif'.
"""

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk import viz


def main():
    """Run the line animation and save the GIF."""
    n = 151          # number of nodes on the line
    center = n // 2
    t_max = 70       # steps to animate, below the distance to the boundary

    g = Graph.line(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")

    # symmetric start (|left> + i|right>) / sqrt(2)
    w = Walker.at_node(g, center, coin_state=[1, 1j])

    states = walk.run(w, t_max)   # states[t] for t = 0..t_max

    # line kind uses an adaptive y-axis so the curve stays visible
    viz.animate(walk, states, save_path="line_walk.gif", kind="line", fps=10)
    print("Animation saved to line_walk.gif")


if __name__ == "__main__":
    main()
