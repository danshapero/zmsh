import pytest
import numpy as np
from numpy import pi as π
import predicates
from zmsh import convex_hull


def is_convex(simplices: np.ndarray, points: np.ndarray):
    for simplex in simplices:
        xs = points[simplex]
        for z in points:
            volume = predicates.volume(np.column_stack((z, *xs)))
            if volume < 0:
                return False

    return True


def test_extreme_points():
    r"""Test computing a starting set of extreme points for initializing a
    convex hull"""
    xs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    indices = convex_hull.extreme_points(xs)
    indices_expected = np.array([0, 1, 2])
    assert np.setdiff1d(indices, indices_expected).size == 0

    xs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.25]])
    indices = convex_hull.extreme_points(xs)
    assert np.setdiff1d(indices, indices_expected).size == 0


def test_convex_hull_2d():
    r"""Test computing the hull of an octagon"""
    num_points = 8
    θs = np.linspace(0.0, 2 * π, num_points + 1)[:-1]
    xs = np.cos(θs)
    ys = np.sin(θs)
    points = np.column_stack((xs, ys))

    machine = convex_hull.ConvexHull(points)
    simplices = machine.run()
    assert np.array_equal(np.unique(simplices), list(range(num_points)))
    assert is_convex(simplices, points)


def test_degenerate_points_2d():
    r"""Test computing the convex hull of a 2D point set where there are both
    collinear points on and almost on the hull"""
    points = np.array(
        [
            [0.5, 0.5 - 1e-16],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.75, 0.25],
        ]
    )

    machine = convex_hull.ConvexHull(points)
    simplices = machine.run()
    for simplex in simplices:
        xs = points[simplex]
        for z in points:
            assert predicates.volume(np.column_stack((z, *xs))) >= 0.0

    assert np.setdiff1d(np.unique(simplices), [1, 2, 3, 4, 5]).size == 0
    assert is_convex(simplices, points)


def test_degenerate_points_3d():
    r"""Test computing the convex hull of a 3D point set where there are both
    collinear points on and almost on the hull"""
    points = np.array(
        [
            [0.25, 0.25, +1e-16],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.5, 0.25],
        ]
    )

    machine = convex_hull.ConvexHull(points)
    simplices = machine.run()
    hull_vertex_ids = np.unique(simplices)

    interior_vertex_id = 0
    expected_hull_vertex_ids = np.delete(np.arange(len(points)), interior_vertex_id)
    assert np.setdiff1d(expected_hull_vertex_ids, hull_vertex_ids).size == 0
    assert is_convex(simplices, points)


def test_alternative_volume_fn():
    r"""Test passing in a different method to compute signed volumes"""
    rng = np.random.default_rng(seed=1729)
    num_points = 20
    points = rng.standard_normal((num_points, 2))

    num_calls = 0

    def volume_fn(*args):
        nonlocal num_calls
        num_calls += 1
        return predicates.volume(*args)

    machine = convex_hull.ConvexHull(points, volume_fn=volume_fn)
    simplices = machine.run()

    assert num_calls > 0


def test_input_start_topology():
    r"""Test picking the starting topology of the hull"""
    num_points = 8
    θs = np.linspace(0.0, 2 * π, num_points + 1)[:-1]
    xs = np.cos(θs)
    ys = np.sin(θs)
    points = np.column_stack((xs, ys))

    start_topology = np.array([[0, 2], [2, 4], [4, 6], [6, 0]], dtype=np.uintp)
    machine = convex_hull.ConvexHull(points, topology=start_topology)
    simplices = machine.run()
    assert np.isin(np.arange(num_points), simplices).all()


@pytest.mark.parametrize(
    "dimension, num_points, num_trials", [(2, 120, 20), (3, 40, 10), (4, 20, 3)]
)
def test_convex_hull_fuzz(dimension, num_points, num_trials):
    rng = np.random.default_rng(seed=1729)
    for trial in range(num_trials):
        points = rng.standard_normal(size=(num_points, dimension))
        machine = convex_hull.ConvexHull(points)
        simplices = machine.run()
        assert is_convex(simplices, points)
