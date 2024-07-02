import pytest
import itertools
import numpy as np
import zmsh
import predicates


def permute_eq(A, B):
    if A.shape != B.shape:
        return False

    N = A.shape[1]
    for p in itertools.permutations(list(range(N)), N):
        diff = np.linalg.norm(A - B[:, p], ord=1)
        if diff == 0:
            return True

    return False


def get_num_candidates(machine):
    candidates = list(machine.visible.vertex_to_cell.keys())
    return len(candidates)


def test_square():
    r"""Test computing the convex hull of a square with a single point in the
    center"""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])

    geometry = zmsh.convex_hull(points)
    delta = geometry.topology.boundary(1).toarray()

    delta_true = np.array(
        [[-1, +1, 0, 0, 0], [0, -1, +1, 0, 0], [0, 0, -1, +1, 0], [+1, 0, 0, -1, 0]],
        dtype=np.int8,
    ).T

    assert permute_eq(delta, delta_true)


def test_alternate_signed_volume():
    r"""Test supplying a different signed volume predicate"""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])

    def signed_volume(points):
        A = np.column_stack((np.ones(points.shape[1]), *points))
        return np.linalg.det(A)

    machine = zmsh.ConvexHullMachine(points, signed_volume=signed_volume)
    geometry = machine.run()
    delta = geometry.topology.boundary(1).toarray()

    delta_true = np.array(
        [[-1, +1, 0, 0, 0], [0, -1, +1, 0, 0], [0, 0, -1, +1, 0], [+1, 0, 0, -1, 0]],
        dtype=np.int8,
    ).T

    assert permute_eq(delta, delta_true)


def test_degenerate_points_2d():
    r"""Test computing the convex hull of a 2D point set where there are
    collinear points on the hull"""
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

    geometry = zmsh.convex_hull(points)
    delta = geometry.topology.boundary(1).toarray()

    delta_true = np.array(
        [
            [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, +1],
            [+1, -1, 0, 0, 0],
            [0, +1, -1, 0, 0],
            [0, 0, +1, -1, 0],
            [0, 0, 0, +1, -1],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )

    assert permute_eq(delta, delta_true)


def test_degenerate_points_3d():
    r"""Test computing the convex hull of a 3D points set where there are
    coplanar points on the hull"""
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

    geometry = zmsh.convex_hull(points)
    covertices = geometry.topology.cocells(0)
    interior_vertex_id = 0
    for vertex_id, (edge_ids, signs) in enumerate(covertices):
        if vertex_id == interior_vertex_id:
            assert len(edge_ids) == 0
        else:
            assert len(edge_ids) > 0


def test_visibility_3d():
    r"""Point 4 is not visible from initial triangle 0, but becomes visible to
    the faces created by splitting that triangle on point 3."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.1, -0.1, -0.1],
        ]
    )

    topology = zmsh.Topology(dimension=2, num_cells=[5, 3, 2])
    edges = topology.cells(1)
    edges[:3, :3] = np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]], dtype=np.int8)
    triangles = topology.cells(2)
    triangles[:, :] = np.array([[+1, -1], [+1, -1], [+1, -1]], dtype=np.int8)

    geometry = zmsh.Geometry(topology, points)
    machine = zmsh.ConvexHullMachine(geometry)
    visibility = machine.visible
    assert all(len(visibility.vertex_to_cell[idx]) > 0 for idx in [3, 4])

    machine.step()
    candidate_id = 4
    for vertex_id in range(len(points)):
        num_visible_cells = len(visibility.vertex_to_cell.get(vertex_id, {}))
        assert num_visible_cells == (3 if vertex_id == candidate_id else 0)


def test_coplanar_face_3d():
    r"""The convex hull of this point set is degenerate -- there are four co-
    planar points lying on the hyperplane {z = 0}. We do *not* merge these
    triangles into one polygon; instead, the hull is non-unique."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )

    geometry = zmsh.convex_hull(points)
    assert len(geometry.topology.cells(2)) == 6


def test_cocircular_points():
    plane_points = np.array(
        [
            [0.0, 1.0],  # NOTE: this is the bad point
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0],
            [1.1, 0.5],
            [1.0, 1.0],
            [0.5, 1.1],
            [-0.1, 0.5],
            [0.25, 0.5],
            [0.5, 0.5],
            [0.75, 0.5],
            [0.5, 0.25],
            [0.5, 0.75],
        ]
    )

    magnitudes = np.sum(plane_points**2, axis=1)
    points = np.column_stack((plane_points, magnitudes))
    geometry = zmsh.convex_hull(points)

    num_points = len(plane_points)
    covertices = geometry.topology.cocells(0)
    assert all(len(edge_ids) > 0 for edge_ids, signs in covertices)


def test_hull_invariants():
    r"""Check that the number of edges is increasing and the number of
    candidate points is decreasing as the algorithm progresses"""
    rng = np.random.default_rng(42)
    num_points = 40
    points = rng.uniform(size=(num_points, 2))

    machine = zmsh.ConvexHullMachine(points)
    num_candidates = [get_num_candidates(machine)]
    while not machine.is_done():
        machine.step()
        num_candidates.append(get_num_candidates(machine))

    num_candidates = np.array(num_candidates)
    assert np.max(np.diff(num_candidates)) <= 0


def convex_hull_fuzz_test(rng, dimension, num_points):
    r"""Generate a random point set, compute the hull, and check it's convex"""
    points = rng.normal(size=(num_points, dimension))
    machine = zmsh.ConvexHullMachine(points)
    num_candidates = [get_num_candidates(machine)]
    while not machine.is_done():
        machine.step()
        num_candidates.append(get_num_candidates(machine))

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
        face_ids, matrices = cells.closure(cell_id)
        orientation = zmsh.simplicial.orientation(matrices)
        X = geometry.points[face_ids[0]]

        for z in points:
            volume = orientation * predicates.volume(np.column_stack((z, *X)))
            assert volume >= 0


@pytest.mark.parametrize(
    "dimension, num_points, num_trials", [(2, 120, 20), (3, 40, 10), (4, 20, 3)]
)
def test_random_point_set(dimension, num_points, num_trials):
    rng = np.random.default_rng(seed=42)
    for trial in range(num_trials):
        convex_hull_fuzz_test(rng, dimension, num_points)
