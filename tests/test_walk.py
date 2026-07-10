"""Basic tests checking that the DTQW physics is correct.

Run with ``python -m pytest`` or directly with ``python tests/test_walk.py``.
Do not depend on matplotlib.
"""

import numpy as np

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk.coin import grover, fourier, hadamard


# ---------------------------------------------------------------------- #
# Coins: unitarity                                                        #
# ---------------------------------------------------------------------- #
def _is_unitary(M):
    M = np.asarray(M)
    return np.allclose(M.conj().T @ M, np.eye(M.shape[0]), atol=1e-12)


def test_coins_are_unitary():
    for d in range(1, 6):
        assert _is_unitary(grover(d)), f"Grover d={d} not unitary"
        assert _is_unitary(fourier(d)), f"Fourier d={d} not unitary"
    for d in (1, 2, 4, 8):
        assert _is_unitary(hadamard(d)), f"Hadamard d={d} not unitary"


def test_hadamard_requires_power_of_two():
    try:
        hadamard(3)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Hadamard(3) should fail")


# ---------------------------------------------------------------------- #
# The step operator is unitary -> the norm is conserved                  #
# ---------------------------------------------------------------------- #
def test_step_operator_unitary():
    g = Graph.cycle(7)
    walk = DiscreteTimeWalk(g, coin="grover")
    assert _is_unitary(walk.U)


def test_norm_conserved():
    g = Graph.line(41)
    w = Walker.at_node(g, 20, coin_state=[1, 1j])
    walk = DiscreteTimeWalk(g, coin="hadamard")
    states = walk.run(w, 15)
    for s in states:
        assert abs(s.norm - 1.0) < 1e-10


def test_probabilities_sum_to_one():
    g = Graph.grid(5, 5)
    w = Walker.at_node(g, (2, 2))
    walk = DiscreteTimeWalk(g, coin="grover")
    final = walk.step(w, times=6)
    p = walk.probabilities(final)
    assert abs(p.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------- #
# Physics: the quantum walk spreads ballistically (~t), not like          #
# diffusion (~sqrt(t)). We check that the standard deviation grows         #
# linearly with the number of steps.                                      #
# ---------------------------------------------------------------------- #
def test_ballistic_spreading():
    n = 201
    center = n // 2
    g = Graph.line(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")
    w = Walker.at_node(g, center, coin_state=[1, 1j])

    positions = np.arange(n) - center
    stds = {}
    for steps in (20, 40, 80):
        final = walk.step(w, times=steps)
        p = walk.probabilities(final)
        mean = (positions * p).sum()
        var = ((positions - mean) ** 2 * p).sum()
        stds[steps] = np.sqrt(var)

    # Ballistic: std(80)/std(20) ~ 4 (linear), well above sqrt(4)=2.
    ratio = stds[80] / stds[20]
    assert ratio > 3.0, f"Expected ~4 (ballistic), got {ratio:.2f}"


def test_symmetric_start_is_symmetric():
    n = 121
    center = n // 2
    g = Graph.line(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")
    # Initial state (|left> + i|right>)/sqrt(2): symmetric distribution.
    w = Walker.at_node(g, center, coin_state=[1, 1j])
    final = walk.step(w, times=40)
    p = walk.probabilities(final)
    assert np.allclose(p, p[::-1], atol=1e-9), "The distribution should be symmetric"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nAll tests passed.")
