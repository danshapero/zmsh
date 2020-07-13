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
    points = np.array(
        [[0., 0.],
         [1., 0.],
         [1., 1.],
         [0., 1.],
         [.5, .5]]
    )

    hull_machine = zmsh.ConvexHull(points)
    topology = hull_machine.run()
    delta = topology.boundary(dimension=1).todense()

    delta_true = np.array(
        [[-1, +1, 0, 0, 0],
         [0, -1, +1, 0, 0],
         [0, 0, -1, +1, 0],
         [+1, 0, 0, -1, 0]],
        dtype=np.int8
    ).T

    assert permute_eq(delta, delta_true)


def test_degenerate_points():
    r"""Test computing the convex hull of a point set where there are
    collinear points on the hull"""
    points = np.array(
        [[0., 0.],
         [.5, 0.],
         [1., 0.],
         [1., 1.],
         [.5, .5],
         [.75, .25]]
    )

    hull_machine = zmsh.ConvexHull(points)
    topology = hull_machine.run()
    delta = topology.boundary(dimension=1).todense()

    delta_true = np.array(
        [[-1, +1, 0, 0, 0, 0],
         [0, -1, +1, 0, 0, 0],
         [0, 0, -1, +1, 0, 0],
         [0, 0, 0, -1, +1, 0],
         [+1, 0, 0, 0, -1, 0]],
        dtype=np.int8
    ).T

    assert permute_eq(delta, delta_true)


def test_random_point_set():
    r"""Generate a random point set, compute the hull, and check it's convex"""
    rng = random.default_rng(42)
    num_points = 40
    points = rng.uniform(size=(num_points, 2))

    hull_machine = zmsh.ConvexHull(points)
    topology = hull_machine.run()
    num_edges = topology.num_cells(dimension=1)
    for vertices, incidence in topology.cells(dimension=1):
        if incidence[0] == +1:
            vertices = (vertices[1], vertices[0])
        x = points[vertices[0], :]
        y = points[vertices[1], :]

        for z in points:
            area = zmsh.predicates.area(x, y, z)
            assert area >= 0
