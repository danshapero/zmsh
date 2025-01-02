import pytest
import numpy as np
from zmsh.simplification import compute_plane
from zmsh.polytopal import (
    standard_simplex,
    join_vertices,
    edge_collapse,
    from_simplicial,
    to_simplicial,
)


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_computing_planes(dimension):
    num_trials = 20
    for trial in range(num_trials):
        rng = np.random.default_rng(seed=1729)
        xs = rng.standard_normal((dimension, dimension))
        p = compute_plane(xs)
        for trial in range(num_trials):
            ts = rng.uniform(size=dimension)
            ts /= np.sum(ts)
            w = ts @ xs
            assert np.abs(p[0] + np.dot(p[1:], w)) < 1e-6


def test_edge_collapse():
    D_0 = np.ones((1, 5), dtype=np.int8)

    D_1 = np.array(
        [
            [0, 0, 0, 0, -1, -1, -1, -1],
            [-1, 0, 0, +1, +1, 0, 0, 0],
            [+1, -1, 0, 0, 0, +1, 0, 0],
            [0, +1, -1, 0, 0, 0, +1, 0],
            [0, 0, +1, -1, 0, 0, 0, +1],
        ],
        dtype=np.int8,
    )

    D_2 = np.array(
        [
            [+1, 0, 0, 0],
            [0, +1, 0, 0],
            [0, 0, +1, 0],
            [0, 0, 0, +1],
            [+1, 0, 0, -1],
            [-1, +1, 0, 0],
            [0, -1, +1, 0],
            [0, 0, -1, +1],
        ],
        dtype=np.int8,
    )

    Ds = [D_0, D_1, D_2]
    vertex_ids = [1, 2]
    Es = edge_collapse(Ds, vertex_ids)
    simplices = to_simplicial(Es)
    assert (vertex_ids[0] in simplices) and not (vertex_ids[1] in simplices)


def test_bunny_collapse():
    patch = np.array(
        [
            [3, 5, 1],
            [7, 5, 8],
            [3, 8, 5],
            [5, 7, 4],
            [5, 4, 9],
            [7, 3, 2],
            [8, 3, 7],
            [5, 9, 0],
            [3, 1, 10],
            [6, 3, 10],
            [0, 1, 5],
            [2, 3, 6],
        ],
    )

    Ds = from_simplicial(patch)
    vertex_ids = [3, 5]
    Es = edge_collapse(Ds, vertex_ids)
    new_patch = to_simplicial(Es)
    assert (vertex_ids[0] in new_patch) and not (vertex_ids[1] in new_patch)
