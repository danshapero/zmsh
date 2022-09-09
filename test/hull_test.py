import itertools
import numpy as np
from numpy import linalg, random
import zmsh


def permute_eq(A, B):
    if A.shape != B.shape:
        return False

    N = A.shape[1]
    for p in itertools.permutations(list(range(N)), N):
        diff = linalg.norm(A - B[:, p], ord=1)
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
            [-1, +1, 0, 0, 0, 0],
            [0, -1, +1, 0, 0, 0],
            [0, 0, -1, +1, 0, 0],
            [0, 0, 0, -1, +1, 0],
            [+1, 0, 0, 0, -1, 0],
        ],
        dtype=np.int8,
    ).T

    assert permute_eq(delta, delta_true)


def test_hull_invariants():
    r"""Check that the number of edges is increasing and the number of
    candidate points is decreasing as the algorithm progresses"""
    rng = random.default_rng(42)
    num_points = 40
    points = rng.uniform(size=(num_points, 2))

    hull_machine = zmsh.ConvexHullMachine(points)
    num_candidates = len(hull_machine.candidates)
    num_edges = hull_machine.num_edges
    while not hull_machine.is_done():
        hull_machine.step()
        assert len(hull_machine.candidates) <= num_candidates
        assert hull_machine.num_edges >= num_edges

        num_candidates = len(hull_machine.candidates)
        num_edges = hull_machine.num_edges


def test_random_point_set():
    r"""Generate a random point set, compute the hull, and check it's convex"""
    rng = random.default_rng(42)
    num_points = 40
    points = rng.uniform(size=(num_points, 2))

    geometry = zmsh.convex_hull(points)
    for vertices, signs in geometry.topology.cells(1):
        if signs[0] == +1:
            vertices = (vertices[1], vertices[0])
        x = points[vertices[0], :]
        y = points[vertices[1], :]

        for z in points:
            area = zmsh.predicates.volume(x, y, z)
            assert area >= 0
