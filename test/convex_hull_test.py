import pytest
import itertools
import numpy as np
import zmsh


def permute_eq(A, B):
    if A.shape != B.shape:
        return False

    N = A.shape[1]
    for p in itertools.permutations(list(range(N)), N):
        diff = np.linalg.norm(A - B[:, p], ord=1)
        if diff == 0:
            return True

    return False


def test_square():
    r"""Test computing the convex hull of a square with a single point in the
    center"""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])

    geometry = zmsh.convex_hull(points)
    delta = geometry.topology.boundary(1).todense()

    delta_true = np.array(
        [[-1, +1, 0, 0, 0], [0, -1, +1, 0, 0], [0, 0, -1, +1, 0], [+1, 0, 0, -1, 0]],
        dtype=np.int8,
    ).T

    assert permute_eq(delta, delta_true)


def test_degenerate_points():
    r"""Test computing the convex hull of a point set where there are
    collinear points on the hull"""
    points = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.75, 0.25]]
    )

    geometry = zmsh.convex_hull(points)
    delta = geometry.topology.boundary(1).todense()

    delta_true = np.array(
        [
            [-1, 0, +1],
            [0, 0, 0],
            [+1, -1, 0],
            [0, +1, -1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )

    assert permute_eq(delta, delta_true)


def test_hull_invariants():
    r"""Check that the number of edges is increasing and the number of
    candidate points is decreasing as the algorithm progresses"""
    rng = np.random.default_rng(42)
    num_points = 40
    points = rng.uniform(size=(num_points, 2))

    hull_machine = zmsh.ConvexHullMachine(points)
    num_candidates = len(hull_machine.candidates)
    while not hull_machine.is_done():
        hull_machine.step()
        assert len(hull_machine.candidates) <= num_candidates
        num_candidates = len(hull_machine.candidates)


def convex_hull_fuzz_test(rng, dimension, num_points):
    r"""Generate a random point set, compute the hull, and check it's convex"""
    points = rng.normal(size=(num_points, dimension))
    machine = zmsh.ConvexHullMachine(points, vertex_elimination_heuristic=True)
    num_candidates = [len(machine.candidates)]
    while not machine.is_done():
        machine.step()
        num_candidates.append(len(machine.candidates))

    geometry = machine.finalize()

    # Check that the set of candidate points shrinks at every iteration
    num_candidates = np.array(num_candidates)
    assert np.max(np.diff(num_candidates)) <= 0

    # Check that all the faces are non-empty
    for k in range(1, dimension):
        for face_ids, signs in geometry.topology.cells(k):
            assert len(face_ids) > 0

    # Check that all the vertices are inside the hull
    cells = geometry.topology.cells(dimension - 1)
    for cell_id in range(len(cells)):
        cells_ids, Ds = cells.closure(cell_id)
        orientation = zmsh.simplicial.orientation(Ds)
        X = geometry.points[cells_ids[0]]

        for z in points:
            volume = orientation * zmsh.predicates.volume(*X, z)
            assert volume >= 0


@pytest.mark.parametrize(
    "dimension, num_points, num_trials", [(2, 120, 20), (3, 40, 10), (4, 20, 3)]
)
def test_random_point_set(dimension, num_points, num_trials):
    rng = np.random.default_rng(seed=42)
    for trial in range(num_trials):
        convex_hull_fuzz_test(rng, dimension, num_points)
