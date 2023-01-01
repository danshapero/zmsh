import numpy as np
import zmsh
from zmsh.predicates import circumcircle


def test_cocircular_points_2d():
    points = np.array(
        [
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0],
            [1.1, 0.5],
            [1.0, 1.0],
            [0.5, 1.1],
            [0.0, 1.0],
            [-0.1, 0.5],
            [0.25, 0.5],
            [0.5, 0.5],
            [0.75, 0.5],
            [0.5, 0.25],
            [0.5, 0.75],
        ]
    )

    geometry = zmsh.delaunay(points)
    topology = geometry.topology
    cells = topology.cells(2)
    for cell_id in range(len(cells)):
        faces_ids, matrices = cells.closure(cell_id)
        orientation = zmsh.simplicial.orientation(matrices)
        X = points[faces_ids[0]]
        assert all([orientation * circumcircle(z, *X) >= 0 for z in points])


def delaunay_fuzz_test(rng, num_points):
    points = rng.normal(size=(num_points, 2))
    geometry = zmsh.delaunay(points)
    topology = geometry.topology
    cells = topology.cells(2)
    for cell_id in range(len(cells)):
        faces_ids, matrices = cells.closure(cell_id)
        orientation = zmsh.simplicial.orientation(matrices)
        X = points[faces_ids[0]]
        assert all([orientation * circumcircle(z, *X) >= 0 for z in points])


def test_random_point_set():
    num_trials, num_points = 10, 40
    rng = np.random.default_rng(seed=42)
    for trial in range(num_trials):
        delaunay_fuzz_test(rng, num_points)
