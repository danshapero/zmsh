import pytest
import numpy as np
import scipy
import predicates
from zmsh import polytopal, delaunay


def is_delaunay(simplices, points):
    for simplex in simplices:
        xs = points[simplex]
        for z in points:
            if predicates.insphere(np.column_stack((z, *xs))) > 0:
                return False

    return True


def test_basic():
    points = np.array([[0.1, 0.0], [1.0, 0], [1.0, 1.0], [0.0, 1.0]])
    machine = delaunay.Delaunay(points)
    simplices = machine.run()
    simplices_expected = np.array([[0, 2, 3], [0, 1, 2]], dtype=np.uintp)
    for simplex_ex in simplices_expected:
        found = False
        for simplex in simplices:
            found |= np.setdiff1d(simplex, simplex_ex).size == 0
        assert found

    assert is_delaunay(simplices, points)


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

    machine = delaunay.Delaunay(points)
    simplices = machine.run()
    assert is_delaunay(simplices, points)


@pytest.mark.parametrize(
    "dimension, num_points, num_trials", [(2, 40, 20), (3, 20, 10)]
)
def test_delaunay_fuzz(dimension, num_points, num_trials):
    rng = np.random.default_rng(seed=1729)
    for trial in range(num_trials):
        points = rng.standard_normal(size=(num_points, dimension))
        machine = delaunay.Delaunay(points)
        simplices = machine.run()
        assert is_delaunay(simplices, points)


def test_edge_intersection():
    # Two crossing lines
    xs = np.array([[0.5, 0.5], [1.5, 1.5]])
    ys = np.array([[0.5, 1.5], [1.5, 0.5]])
    assert delaunay.line_segments_intersect(xs, ys) < 0

    # Segments intersect at a single point
    ys = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert delaunay.line_segments_intersect(xs, ys) == 0
    assert delaunay.line_segments_intersect(ys, xs) == 0

    # Segments do not intersect
    ys += 1e-15
    assert delaunay.line_segments_intersect(xs, ys) > 0
    assert delaunay.line_segments_intersect(ys, xs) > 0


def test_triangle_edge_intersections():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]
    )
    simplices = np.array(
        [
            [0, 1, 4],
            [1, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 6],
            [3, 7, 6],
        ],
        dtype=np.uintp,
    )

    xs = points[[0, 6], :]
    cell_ids = delaunay.find_crossings(simplices, points, xs)
    assert np.array_equal(cell_ids, [0, 1, 2, 3])


def test_finding_splitting_vertex():
    points = np.array([[0.0, 0.0], [4.0, 0.0], [3.0, 2.0], [2.0, 1.0], [1.0, 2.0]])
    d_0 = np.ones((1, 5), dtype=np.int8)
    d_1 = np.array(
        [
            [-1, 0, 0, 0, +1],
            [+1, -1, 0, 0, 0],
            [0, +1, -1, 0, 0],
            [0, 0, +1, -1, 0],
            [0, 0, 0, +1, -1],
        ],
        dtype=np.int8,
    )
    d_2 = np.ones((len(points), 1), dtype=np.int8)
    edge_id = 0
    vertex_id = delaunay.find_splitting_vertex([d_0, d_1, d_2], edge_id, points)
    assert vertex_id == 3

    d_1[:, edge_id] *= -1
    d_2[edge_id, :] *= -1
    vertex_id = delaunay.find_splitting_vertex([d_0, d_1, d_2], edge_id, points)
    assert vertex_id == 3


def test_finding_splitting_vertex_hanging_edge():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [1.0, 1.0]])
    d_0 = np.ones((1, 4), dtype=np.int8)
    d_1 = np.array(
        [[-1, 0, +1, 0], [+1, -1, 0, 0], [0, +1, -1, -1], [0, 0, 0, +1]], dtype=np.int8
    )
    d_2 = np.array([[+1], [+1], [+1], [0]], dtype=np.int8)
    edge_id = 0
    vertex_id = delaunay.find_splitting_vertex([d_0, d_1, d_2], edge_id, points)
    assert vertex_id == 3

    d_1[:, edge_id] *= -1
    d_2[edge_id, :] *= -1
    vertex_id = delaunay.find_splitting_vertex([d_0, d_1, d_2], edge_id, points)
    assert vertex_id == 3


def test_retriangulating_cavity():
    points = np.array(
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-2.0, 0.0],
            [-1.0, -1.0],
            [1.0, -0.5],
        ]
    )
    simplices = delaunay.Delaunay(points).run()

    edge = (0, 4)
    cell_ids = delaunay.find_crossings(simplices, points, points[edge, :])
    lsimplices = simplices[cell_ids]
    vertex_ids = np.unique(lsimplices)
    lpoints = points[vertex_ids]
    id_map = np.vectorize({idx: val for val, idx in enumerate(vertex_ids)}.get)

    machine = delaunay.Retriangulation.from_simplices(lsimplices, lpoints, id_map(edge))
    d_0, d_1, d_2 = machine.run()
    edge_ids = np.flatnonzero(np.count_nonzero(d_1[id_map(edge), :], axis=0) == 2)
    assert edge_ids.size == 1


def test_retriangulating_bigger_cavity():
    num_points = 8
    points0 = np.column_stack((np.arange(num_points), np.ones(num_points)))
    points1 = np.column_stack((np.arange(num_points), -np.ones(num_points)))
    points2 = np.array([[-1.0, 0.0], [num_points, 0.0]])
    points = np.vstack((points0, points1, points2))
    simplices = delaunay.Delaunay(points).run()

    edge = (2 * num_points, 2 * num_points + 1)
    cell_ids = delaunay.find_crossings(simplices, points, points[edge, :])
    lsimplices = simplices[cell_ids]
    vertex_ids = np.unique(lsimplices)
    lpoints = points[vertex_ids]
    id_map = np.vectorize({idx: val for val, idx in enumerate(vertex_ids)}.get)

    machine = delaunay.Retriangulation.from_simplices(lsimplices, lpoints, id_map(edge))
    d_0, d_1, d_2 = machine.run()
    edge_ids = np.flatnonzero(np.count_nonzero(d_1[id_map(edge), :], axis=0) == 2)
    assert edge_ids.size == 1


def test_retriangulating_hanging_edge():
    # This choice of length forces an edge to hang (found by plotting it and
    # lots of trial and error).
    a = 7.0
    points = np.array(
        [[+a, 0.0], [0.0, 1.0], [-a, 0.0], [-0.5, -1.0], [0.5, -1.0], [0.0, 0.5]]
    )
    simplices = delaunay.Delaunay(points).run()

    edge = (0, 2)
    cell_ids = delaunay.find_crossings(simplices, points, points[edge, :])
    lsimplices = simplices[cell_ids]
    vertex_ids = np.unique(lsimplices)
    lpoints = points[vertex_ids]
    id_map = np.vectorize({idx: val for val, idx in enumerate(vertex_ids)}.get)

    machine = delaunay.Retriangulation.from_simplices(lsimplices, lpoints, id_map(edge))

    d_0, d_1, d_2 = machine.run()
    edge_ids = np.flatnonzero(np.count_nonzero(d_1[id_map(edge), :], axis=0) == 2)
    assert edge_ids.size == 1


def test_retriangulating_cheng_dey_shewchuk():
    r"""The example from Figure 13 on page 75 of `Delaunay Mesh Generation` by
    Cheng, Dey, and Shewchuk"""
    points = np.array(
        [
            [16.5, -3.5],
            [8.0, 7.75],
            [9.5, -2.0],
            [6.5, -2.0],
            [8.0, 1.5],
            [-3.5, -0.0],
            [-4.25, -2.0],
            [-12.0, -0.0],
            [-10.0, 2.25],
            [-19.0, -3.5],
        ],
    )

    d_0 = np.ones((1, len(points)), dtype=np.int8)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (5, 7),
        (7, 8),
        (8, 9),
        (9, 0),
    ]
    d_1 = np.zeros((len(points), len(edges)), dtype=np.int8)
    for edge_id, (v0, v1) in enumerate(edges):
        d_1[(v0, v1), edge_id] = (-1, +1)

    d_2 = np.ones((len(edges), 1), dtype=np.int8)
    d_2[5] = 0

    queue = [list(range(len(edges)))[::-1]]
    machine = delaunay.Retriangulation([d_0, d_1, d_2], points, queue)
    topology = machine.run()
    simplices = polytopal.to_simplicial(topology)
    assert len(simplices) == 9

    edge_ids = np.flatnonzero(np.count_nonzero(d_1[edges[-1], :], axis=0) == 2)
    assert edge_ids.size == 1


def test_retriangulating_end_indices():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    simplices = np.array([[0, 1, 3], [1, 2, 4], [1, 4, 3], [2, 5, 4]])

    edge = (1, 5)
    cell_ids = delaunay.find_crossings(simplices, points, points[edge, :])
    assert np.setdiff1d(cell_ids, np.array([1, 3])).size == 0

    lsimplices = simplices[cell_ids]
    vertex_ids = np.unique(lsimplices)
    lpoints = points[vertex_ids]
    id_map = np.vectorize({idx: val for val, idx in enumerate(vertex_ids)}.get)

    machine = delaunay.Retriangulation.from_simplices(lsimplices, lpoints, id_map(edge))
    d_0, d_1, d_2 = machine.run()
    edge_ids = np.flatnonzero(np.count_nonzero(d_1[id_map(edge), :], axis=0) == 2)
    new_simplices = vertex_ids[polytopal.to_simplicial([d_0, d_1, d_2])]
    new_simplices_expected = np.array([[1, 2, 5], [4, 1, 5]])
    for simplex1 in new_simplices:
        found = False
        for simplex2 in new_simplices_expected:
            found |= np.setdiff1d(simplex1, simplex2).size == 0
        assert found


def test_constrained_delaunay_basic():
    points = np.array([[-2.0, 0.0], [0.0, -1.0], [2.0, 0.0], [0.0, 1.0]])
    machine = delaunay.ConstrainedDelaunay(points, np.array([[0, 2]]))
    num_iterations = 0
    while not machine.is_done():
        machine.step()
        num_iterations += 1
    simplices = machine.finalize()
    assert num_iterations > 0


def constrained_delaunay_trial(rng: np.random.Generator):
    num_points = 40
    poisson_disk = scipy.stats.qmc.PoissonDisk(2, radius=0.05, seed=rng)
    points = poisson_disk.random(num_points)

    num_edges = 5
    edges = []
    while len(edges) < num_edges:
        proposed_edge = rng.choice(num_points, size=2, replace=False)
        xs = points[proposed_edge, :]
        intersections = [
            delaunay.line_segments_intersect(xs, points[edge, :]) < 0.0
            for edge in edges
        ]

        if not any(intersections):
            edges.append(proposed_edge)

    edges = np.array(edges)
    constrained_simplices = delaunay.ConstrainedDelaunay(points, edges).run()
    d_0, d_1, d_2 = polytopal.from_simplicial(constrained_simplices)
    for edge in edges:
        assert (np.count_nonzero(d_1[edge, :], axis=0) == 2).any()


def test_constrained_delaunay_fuzz():
    rng = np.random.default_rng(seed=112358)
    num_trials = 20
    for trial in range(num_trials):
        constrained_delaunay_trial(rng)
