"""Basic tests checking that the DTQW physics is correct.

Run with ``python -m pytest`` or directly with ``python tests/test_walk.py``.
Do not depend on matplotlib.
"""

import numpy as np

from zitterwalk import Graph, Walker, DiscreteTimeWalk
from zitterwalk.coin import grover, fourier, hadamard, rotation, su2, random_coins


# ---------------------------------------------------------------------- #
# Coins unitarity                                                        #
# ---------------------------------------------------------------------- #
def _is_unitary(M):
    """Whether a matrix is unitary to numerical tolerance.

    Args:
        M: the matrix to test.

    Returns:
        True if M is unitary.
    """
    M = np.asarray(M)
    return np.allclose(M.conj().T @ M, np.eye(M.shape[0]), atol=1e-12)


def test_coins_are_unitary():
    """Every standard coin is unitary across a range of dimensions."""
    for d in range(1, 6):
        assert _is_unitary(grover(d)), f"Grover d={d} not unitary"
        assert _is_unitary(fourier(d)), f"Fourier d={d} not unitary"
    for d in (1, 2, 4, 8):
        assert _is_unitary(hadamard(d)), f"Hadamard d={d} not unitary"


def test_hadamard_requires_power_of_two():
    """The Hadamard coin rejects a non-power-of-2 dimension."""
    try:
        hadamard(3)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Hadamard(3) should fail")


# ---------------------------------------------------------------------- #
# The step operator is unitary so the norm is conserved                  #
# ---------------------------------------------------------------------- #
def test_step_operator_unitary():
    """The assembled step operator U is unitary."""
    g = Graph.cycle(7)
    walk = DiscreteTimeWalk(g, coin="grover")
    assert _is_unitary(walk.U)


def test_norm_conserved():
    """The state norm stays 1 along the whole trajectory."""
    g = Graph.line(41)
    w = Walker.at_node(g, 20, coin_state=[1, 1j])
    walk = DiscreteTimeWalk(g, coin="hadamard")
    states = walk.run(w, 15)
    for s in states:
        assert abs(s.norm - 1.0) < 1e-10


def test_probabilities_sum_to_one():
    """The node distribution sums to 1 for a unitary walk."""
    g = Graph.grid(5, 5)
    w = Walker.at_node(g, (2, 2))
    walk = DiscreteTimeWalk(g, coin="grover")
    final = walk.step(w, times=6)
    p = walk.probabilities(final)
    assert abs(p.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------- #
# Physics ballistic spreading (std grows linearly with the steps)        #
# ---------------------------------------------------------------------- #
def test_ballistic_spreading():
    """The standard deviation grows linearly (ballistic), not like sqrt(t)."""
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

    # ballistic ratio std(80)/std(20) ~ 4, well above the diffusive 2
    ratio = stds[80] / stds[20]
    assert ratio > 3.0, f"Expected ~4 (ballistic), got {ratio:.2f}"


def test_symmetric_start_is_symmetric():
    """A symmetric initial coin state gives a symmetric distribution."""
    n = 121
    center = n // 2
    g = Graph.line(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")
    w = Walker.at_node(g, center, coin_state=[1, 1j])
    final = walk.step(w, times=40)
    p = walk.probabilities(final)
    assert np.allclose(p, p[::-1], atol=1e-9), "The distribution should be symmetric"


# ---------------------------------------------------------------------- #
# Todo 2, inhomogeneous coins                                            #
# ---------------------------------------------------------------------- #
def test_rotation_recovers_hadamard():
    """rotation(pi/4) is the Hadamard coin, so the two walks coincide."""
    g = Graph.line(81)
    w = Walker.at_node(g, 40, coin_state=[1, 1j])
    a = DiscreteTimeWalk(g, coin="hadamard").step(w, 30)
    b = DiscreteTimeWalk(g, coin=rotation(np.pi / 4)).step(w, 30)
    assert np.allclose(a.psi, b.psi, atol=1e-12)


def test_per_node_coin_dict_is_unitary():
    """A per-node coin dict still assembles a unitary step operator."""
    g = Graph.line(31)
    walk = DiscreteTimeWalk(g, coin={15: rotation(0.3)}, coin_default="hadamard")
    assert _is_unitary(walk.U)


def test_disorder_localizes():
    """Static coin disorder localizes, with far smaller spread than the clean walk."""
    n = 201
    g = Graph.line(n)
    w = Walker.at_node(g, n // 2, coin_state=[1, 1j])
    clean = DiscreteTimeWalk(g, coin="hadamard")
    dirty = DiscreteTimeWalk(g, coin=random_coins(g, seed=0))
    assert dirty.std(dirty.step(w, 120)) < 0.4 * clean.std(clean.step(w, 120))


def test_time_dependent_coin_preserves_norm():
    """A time-dependent coin schedule preserves the state norm."""
    g = Graph.line(41)
    w = Walker.at_node(g, 20, coin_state=[1, 1j])
    walk = DiscreteTimeWalk(g, coin_schedule=lambda t: rotation(0.1 * t))
    assert walk.U is None
    for s in walk.run(w, 12):
        assert abs(s.norm - 1.0) < 1e-10


# ---------------------------------------------------------------------- #
# Todo 3, observables                                                    #
# ---------------------------------------------------------------------- #
def test_mean_and_variance_match_manual():
    """mean_position and variance match a manual computation."""
    n = 151
    center = n // 2
    g = Graph.line(n)
    walk = DiscreteTimeWalk(g, coin="hadamard")
    w = Walker.at_node(g, center, coin_state=[1, 1j])
    final = walk.step(w, 40)
    p = walk.probabilities(final)
    pos = np.arange(n)
    mean = (pos * p).sum()
    var = ((pos - mean) ** 2 * p).sum()
    assert abs(walk.mean_position(final) - mean) < 1e-9
    assert abs(walk.variance(final) - var) < 1e-9


def test_coin_entropy_bounds():
    """Coin entropy starts at zero and stays within [0, 1] bit."""
    g = Graph.cycle(64)
    walk = DiscreteTimeWalk(g, coin="hadamard")
    w = Walker.at_node(g, 0, coin_state=[1, 0])
    assert abs(walk.coin_entropy(w)) < 1e-12
    e = walk.coin_entropy(walk.step(w, 20))
    assert 0.0 <= e <= 1.0 + 1e-9


def test_spectrum_on_unit_circle():
    """The quasi-energies lie in (-pi, pi] with one per arc."""
    g = Graph.cycle(10)
    walk = DiscreteTimeWalk(g, coin="grover")
    omega = walk.spectrum()
    assert omega.shape == (g.n_arcs,)
    assert np.all(omega >= -np.pi - 1e-9) and np.all(omega <= np.pi + 1e-9)


# ---------------------------------------------------------------------- #
# Configurable shift, flip-flop (default), moving, and custom            #
# ---------------------------------------------------------------------- #
def test_moving_shift_is_unitary():
    """The moving shift is a permutation, so U stays unitary."""
    for g in (Graph.line(20), Graph.cycle(12), Graph.grid(5, 5)):
        walk = DiscreteTimeWalk(g, coin="grover", shift="moving")
        assert _is_unitary(walk.S), "moving shift is not a permutation"
        assert _is_unitary(walk.U), "moving-shift step operator is not unitary"


def test_moving_shift_rotates_the_cycle():
    """On a cycle the moving shift translates arcs, (i, i+1) to (i+1, i+2)."""
    g = Graph.cycle(8)
    S = DiscreteTimeWalk(g, shift="moving").S
    for i in range(8):
        src = g.arc_index[(i, (i + 1) % 8)]
        dst = g.arc_index[((i + 1) % 8, (i + 2) % 8)]
        out = np.zeros(g.n_arcs, dtype=complex)
        out[src] = 1.0
        assert np.argmax(np.abs(S @ out)) == dst


def test_moving_shift_goes_straight_on_the_line():
    """Interior line sites pass straight through, (2, 3) to (3, 4)."""
    g = Graph.line(10)
    S = DiscreteTimeWalk(g, shift="moving").S
    src = g.arc_index[(2, 3)]
    out = np.zeros(g.n_arcs, dtype=complex)
    out[src] = 1.0
    assert np.argmax(np.abs(S @ out)) == g.arc_index[(3, 4)]


def test_flip_flop_is_the_default():
    """The default shift is flip-flop."""
    g = Graph.cycle(9)
    default = DiscreteTimeWalk(g, coin="hadamard")
    explicit = DiscreteTimeWalk(g, coin="hadamard", shift="flip_flop")
    assert np.allclose(default.S, explicit.S)


def test_custom_permutation_shift():
    """A custom permutation (and a callable) equal to flip reproduces flip-flop."""
    g = Graph.cycle(6)
    ref = DiscreteTimeWalk(g, coin="grover")
    custom = DiscreteTimeWalk(g, coin="grover", shift=np.asarray(g.flip))
    assert np.allclose(ref.U, custom.U)
    cb = DiscreteTimeWalk(g, coin="grover", shift=lambda gr: gr.flip)
    assert np.allclose(ref.U, cb.U)


def test_invalid_shift_rejected():
    """An unknown name or a malformed permutation is rejected."""
    g = Graph.cycle(6)
    for bad in ("nonsense", np.arange(g.n_arcs - 1), np.zeros(g.n_arcs, int)):
        try:
            DiscreteTimeWalk(g, shift=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shift={bad!r} should have been rejected")


def test_moving_shift_passes_through_search_kwargs():
    """search forwards the shift keyword to the constructor."""
    g = Graph.complete(8)
    walk = DiscreteTimeWalk.search(g, marked=3, shift="flip_flop")
    assert _is_unitary(walk.U)


# ---------------------------------------------------------------------- #
# Reproducing the papers, SU(2) coin, Gaussian packets, Dirac physics    #
# ---------------------------------------------------------------------- #
def test_su2_coin_unitary_with_gap_theta():
    """The su2 coin is SU(2) with eigenphase gap equal to theta."""
    for theta in (0.3, np.pi / 4, np.pi / 2, 1.7):
        for axis in ("x", "y", "z"):
            C = su2(theta, axis)(2)
            assert _is_unitary(C), f"su2({theta},{axis}) not unitary"
            assert abs(np.linalg.det(C) - 1.0) < 1e-12, "su2 must be SU(2)"
            ph = np.angle(np.linalg.eigvals(C))  # eigenphase gap equals theta
            gap = abs(ph[0] - ph[1])
            gap = min(gap, 2 * np.pi - gap)
            assert abs(gap - theta) < 1e-9, f"gap {gap} != theta {theta}"
    assert su2(0.5)(1).shape == (1, 1)  # degree-1 boundary falls back to [[1]]


def test_gaussian_packet_normalized_and_localized():
    """A Gaussian packet is normalized and its mass sits near the centre."""
    g = Graph.line(201)
    w = Walker.gaussian(g, center=100, width=8.0, coin_state=[1, -1])
    assert abs(w.norm - 1.0) < 1e-12
    walk = DiscreteTimeWalk(g, coin="hadamard")
    p = walk.probabilities(w)
    assert p[100] > p[130]
    assert p[:60].sum() < 1e-6 and p[140:].sum() < 1e-6


def test_gaussian_momentum_drives_a_drift():
    """A nonzero momentum gives the packet a net group-velocity displacement."""
    g = Graph.line(301)
    walk = DiscreteTimeWalk(g, coin=su2(np.pi / 2), shift="moving")
    still = Walker.gaussian(g, center=150, width=18.0, coin_state=[0, 1])
    moving = Walker.gaussian(g, center=150, width=18.0, coin_state=[0, 1],
                             momentum=0.8)
    d_still = abs(walk.mean_position(walk.step(still, 40)) - 150)
    d_moving = abs(walk.mean_position(walk.step(moving, 40)) - 150)
    assert d_moving > d_still + 2.0


def test_trajectory_observables_shapes():
    """The trajectory observables return arrays of length steps + 1."""
    g = Graph.line(81)
    walk = DiscreteTimeWalk(g, coin="hadamard", shift="moving")
    w = Walker.at_node(g, 40, coin_state=[1, 1j])
    states = walk.run(w, 10)
    assert walk.mean_position_evolution(states).shape == (11,)
    assert walk.std_evolution(states).shape == (11,)
    assert walk.return_probability_evolution(states, 40).shape == (11,)


def test_dirac_walk_zitterbewegung_frequency():
    """The Dirac walk trembles at omega_ZB equal to theta."""
    n, center, theta = 601, 300, np.pi / 2
    g = Graph.line(n)
    walk = DiscreteTimeWalk(g, coin=su2(theta, "x"), shift="moving")
    w = Walker.gaussian(g, center=center, width=22.0, coin_state=[1, 0])
    states = walk.run(w, 80)
    xt = walk.mean_position_evolution(states)
    t = np.arange(len(xt))
    detrended = xt - np.poly1d(np.polyfit(t, xt, 1))(t)
    amp = np.abs(np.fft.rfft(detrended))
    freq = np.fft.rfftfreq(len(detrended))
    omega = 2 * np.pi * freq[1:][np.argmax(amp[1:])]
    assert abs(omega - theta) < 0.15, f"omega_ZB {omega} != theta {theta}"


def test_electric_field_localizes_under_moving_shift():
    """A rational electric field bounds <x> (Bloch), unlike the free walk."""
    n, center = 401, 200
    g = Graph.line(n)
    phi = 2 * np.pi / 30
    field = DiscreteTimeWalk(g, coin=rotation(np.pi / 4), shift="moving",
                             field=lambda x: phi * (x - center))
    w = Walker.gaussian(g, center=center, width=12.0, coin_state=[1, -1])
    states = field.run(w, 120)
    xt = field.mean_position_evolution(states) - center
    assert np.abs(xt).max() < 2.5 / phi  # bounded region ~ 1/phi
    assert np.abs(xt[-20:]).min() < np.abs(xt).max()  # it comes back


# ---------------------------------------------------------------------- #
# Todo 4, self-loops, absorption, search, measurement                    #
# ---------------------------------------------------------------------- #
def test_self_loop_adds_one_arc():
    """A self-loop adds one arc and the walk stays unitary."""
    g = Graph.line(5)
    d_before = g.degree(2)
    g.add_edge(2, 2)
    assert g.degree(2) == d_before + 1
    walk = DiscreteTimeWalk(g, coin="grover")
    assert _is_unitary(walk.U)  # self-loop arc is its own flip


def test_absorbing_node_loses_probability():
    """An absorbing node drains probability monotonically over time."""
    g = Graph.line(41)
    w = Walker.at_node(g, 20, coin_state=[1, 1j])
    walk = DiscreteTimeWalk(g, coin="hadamard", absorbing=0)
    states = walk.run(w, 80)
    surv = [walk.survival_probability(s) for s in states]
    assert surv[0] > 0.999
    assert surv[-1] < surv[0]  # some amplitude was absorbed
    assert all(a >= b - 1e-12 for a, b in zip(surv, surv[1:]))  # monotone down


def test_search_beats_uniform():
    """Grover search lifts the marked vertex well above the uniform 1/N."""
    N = 32
    g = Graph.complete(N)
    walk = DiscreteTimeWalk.search(g, marked=7)
    u = Walker.uniform(g)
    peak = max(walk.success_probability(s) for s in walk.run(u, 25))
    assert peak > 0.4  # vs 1/N ~ 0.03


def test_measure_collapses_to_a_node():
    """A collapsing measurement puts all probability on the measured node."""
    g = Graph.line(21)
    walk = DiscreteTimeWalk(g, coin="hadamard")
    w = Walker.at_node(g, 10, coin_state=[1, 1j])
    final = walk.step(w, 8)
    node, collapsed = walk.measure(final, seed=0, collapse=True)
    assert abs(collapsed.norm - 1.0) < 1e-12
    assert abs(walk.probabilities(collapsed)[g.node_position(node)] - 1.0) < 1e-12


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nAll tests passed.")
