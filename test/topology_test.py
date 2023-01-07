from math import comb as binomial
import pytest
import numpy as np
import numpy.linalg
import scipy.sparse.linalg
import zmsh


def matrix_norm(*args, **kwargs):
    try:
        return scipy.sparse.linalg.norm(*args, **kwargs)
    except TypeError:
        return numpy.linalg.norm(*args, **kwargs)


def check_boundaries(topology):
    for k in range(topology.dimension):
        A = topology.boundary(k)
        B = topology.boundary(k + 1)
        C = A @ B
        if matrix_norm(C) != 0:
            return False

    return True


def test_allocating():
    for dimension in range(1, 5):
        num_cells = [binomial(dimension + 1, k + 1) for k in range(dimension + 1)]
        topology = zmsh.Topology(dimension, num_cells)
        for d in range(dimension + 1):
            assert len(topology.cells(d)) == num_cells[d]


def test_zero_initialize_and_resize():
    for dimension in range(1, 5):
        topology = zmsh.Topology(dimension)
        assert topology.dimension == dimension
        for d in range(dimension + 1):
            assert len(topology.cells(d)) == 0

        for d in range(dimension + 1):
            topology.cells(d).resize(dimension - d + 1)

        for d in range(dimension + 1):
            assert len(topology.cells(d)) == dimension - d + 1


def test_edge():
    topology = zmsh.Topology(dimension=1, num_cells=[2, 1])

    # Check that there are no non-zero ∂∂-products
    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)

    vertex_ids, signs = edges[0]
    assert np.array_equal(vertex_ids, (0, 1))
    assert np.array_equal(signs, (-1, +1))
    assert check_boundaries(topology)

    covertices = topology.cocells(0)
    edge_ids, signs = covertices[0]
    assert np.array_equal(edge_ids, (0,))

    # Now make an edge with two endpoints and check that the vertex * edge
    # matrix is non-zero
    edges[(0, 1), 0] = (+1, +1)
    assert not check_boundaries(topology)


def test_setting_multiple_cells():
    topology = zmsh.Topology(dimension=1, num_cells=[3, 2])
    edges = topology.cells(1)
    vertex_ids = (0, 1, 2)
    edge_ids = (0, 1)
    edges[vertex_ids, edge_ids] = np.array([[-1, 0], [+1, -1], [0, +1]])
    assert check_boundaries(topology)
    assert matrix_norm(topology.boundary(1)) != 0

    edges[vertex_ids, :2] = np.array([[0, -1], [-1, +1], [+1, 0]])
    assert check_boundaries(topology)
    assert matrix_norm(topology.boundary(1)) != 0

    edges[vertex_ids, :] = np.array([[-1, 0], [+1, -1], [0, +1]])
    assert check_boundaries(topology)
    assert matrix_norm(topology.boundary(1)) != 0

    edges[:, :] = np.array([[-1, 0], [+1, -1], [0, +1]])
    assert check_boundaries(topology)
    assert matrix_norm(topology.boundary(1)) != 0


def test_setting_cells_integral_index():
    # This can fail if you do type checking naively because
    # `isinstance(np.int64(0), int)` returns `False`.
    topology = zmsh.Topology(dimension=1, num_cells=[2, 1])
    edges = topology.cells(1)
    index = np.int64(0)
    edges[(0, 1), index] = (-1, +1)
    assert check_boundaries(topology)
    vertex_ids, signs = edges[index]
    assert signs.shape == (2,)


def test_setting_cells_bad_index():
    topology = zmsh.Topology(dimension=1, num_cells=[2, 1])
    edges = topology.cells(1)
    with pytest.raises(Exception):
        index = "a"
        edges[(0, 1), index] = (-1, +1)


def test_no_excess_zero_entries():
    topology = zmsh.Topology(dimension=1, num_cells=[3, 2])
    A = topology.boundary(1)
    assert A.count_nonzero() == 0
    edges = topology.cells(1)
    edges[(0, 1, 2), (0, 1)] = np.array([[-1, 0], [+1, -1], [0, +1]])
    assert A.count_nonzero() == 4


def test_getting_empty_cell():
    topology = zmsh.Topology(dimension=1, num_cells=[2, 2])
    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    face_ids, signs = edges[1]
    assert np.array_equal(face_ids, ())
    assert np.array_equal(signs, ())


def test_reset_cell():
    topology = zmsh.Topology(dimension=1, num_cells=[4, 1])

    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    edges[(2, 3), 0] = (-1, +1)
    vertex_ids, signs = edges[0]
    assert np.array_equal(vertex_ids, (2, 3))


def test_triangle():
    topology = zmsh.Topology(dimension=2, num_cells=[3, 3, 1])

    edges = topology.cells(1)
    edges[(0, 1, 2), :] = np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]])

    triangles = topology.cells(2)
    triangles[(0, 1, 2), :] = np.array([[+1, +1, +1]])

    # Check that the faces and cofaces make sense
    face_ids, signs = triangles[0]
    assert np.array_equal(face_ids, (0, 1, 2))

    face_ids, signs = edges[(0, 1)]
    assert np.array_equal(face_ids, (0, 1, 2))
    assert np.array_equal(signs, np.array([[-1, 0], [+1, -1], [0, +1]]))

    face_ids, signs = edges[:2]
    assert np.array_equal(face_ids, (0, 1, 2))
    signs_expected = np.array([[-1, 0], [+1, -1], [0, +1]])
    assert np.array_equal(signs, signs_expected)

    coedges = topology.cocells(1)
    triangle_ids, signs = coedges[0]
    assert np.array_equal(triangle_ids, (0,))

    covertices = topology.cocells(0)
    edge_ids, signs = covertices[(0, 1)]
    assert np.array_equal(edge_ids, (0, 1, 2))
    signs_expected = np.array([[-1, +1], [0, -1], [+1, 0]])
    assert np.array_equal(signs, signs_expected)

    # Check that getting cell closures makes sense
    cells_ids, matrices = triangles.closure(0)
    seq = zip(cells_ids[:-1], cells_ids[1:], matrices)
    for d, (face_ids, cell_ids, matrix) in enumerate(seq, start=1):
        assert np.array_equal(topology.cells(d)[cell_ids][0], face_ids)

    for D_1, D_2 in zip(matrices[:-1], matrices[1:]):
        assert matrix_norm(D_1 @ D_2) == 0

    # Check that there are no non-zero ∂∂-products
    assert check_boundaries(topology)

    # Now change the triangle so that one of the edges is reversed and check
    # that the triangle * edge matrix is non-zero
    triangles[(0, 1, 2), 0] = (+1, -1, +1)
    assert not check_boundaries(topology)


def test_triangle_pair():
    topology = zmsh.Topology(dimension=2, num_cells=(4, 5, 2))

    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    edges[(1, 2), 1] = (-1, +1)
    edges[(2, 0), 2] = (-1, +1)
    edges[(0, 3), 3] = (-1, +1)
    edges[(3, 1), 4] = (-1, +1)

    triangles = topology.cells(2)
    triangles[(0, 1, 2), 0] = (+1, +1, +1)
    triangles[(0, 3, 4), 1] = (-1, +1, +1)

    # Check that the faces and cofaces make sense
    coedges = topology.cocells(1)
    triangle_ids, signs = coedges[0]
    assert np.array_equal(triangle_ids, (0, 1))
    assert np.array_equal(signs, (+1, -1))

    covertices = topology.cocells(0)
    edge_ids, signs = covertices[(0, 1)]
    assert np.array_equal(edge_ids, (0, 1, 2, 3, 4))

    # Check that getting cell closures makes sense
    cells_ids, matrices = triangles.closure([0, 1])
    seq = zip(cells_ids[:-1], cells_ids[1:], matrices)
    for d, (face_ids, cell_ids, matrix) in enumerate(seq, start=1):
        expected_face_ids, expected_matrix = topology.cells(d)[cell_ids]
        assert np.array_equal(expected_face_ids, face_ids)

    for D_1, D_2 in zip(matrices[:-1], matrices[1:]):
        assert matrix_norm(D_1 @ D_2) == 0

    cells_ids, matrices = covertices.closure([0, 1])
    for D_1, D_2 in zip(matrices[:-1], matrices[1:]):
        assert matrix_norm(D_1.T @ D_2.T) == 0

    # Check that there are no non-zero ∂∂-products
    assert check_boundaries(topology)
    assert topology.coboundary(1).shape == (2, 5)

    # Make a topologically valid transformation -- reverse all the incidences
    # of a single cell
    triangles[(0, 3, 4), 1] = (+1, -1, -1)
    assert check_boundaries(topology)


def test_iterating_over_cells():
    topology = zmsh.Topology(dimension=1, num_cells=(3, 2))

    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    for face_ids, signs in topology.cells(1):
        assert np.array_equal(face_ids, (0, 1)) ^ (len(face_ids) == 0)


def test_permutation():
    topology = zmsh.Topology(dimension=2, num_cells=(4, 5, 2))

    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    edges[(1, 2), 1] = (-1, +1)
    edges[(2, 0), 2] = (-1, +1)
    edges[(0, 3), 3] = (-1, +1)
    edges[(3, 1), 4] = (-1, +1)

    triangles = topology.cells(2)
    triangles[(0, 1, 2), 0] = (+1, +1, +1)
    triangles[(0, 3, 4), 1] = (-1, +1, +1)

    p = np.array([4, 1, 2, 3, 0], dtype=int)
    edges.permute(p)

    assert check_boundaries(topology)
    assert np.array_equal(triangles[0][0], np.array([1, 2, 4]))
    assert np.array_equal(triangles[1][0], np.array([0, 3, 4]))

    q = np.array([1, 0], dtype=int)
    triangles.permute(q)
    assert check_boundaries(topology)
    assert np.array_equal(triangles[1][0], np.array([1, 2, 4]))
    assert np.array_equal(triangles[0][0], np.array([0, 3, 4]))


def test_removing_empty_cells():
    topology = zmsh.Topology(dimension=2, num_cells=(3, 4, 2))

    edges = topology.cells(1)
    D = np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]], dtype=np.int8)
    edges[(0, 1, 2), (0, 2, 3)] = D

    triangles = topology.cells(2)
    triangles[(0, 2, 3), 0] = (+1, +1, +1)

    triangles.remove_empty_cells()
    edges.remove_empty_cells()

    assert len(edges) == 3
    assert len(triangles) == 1
    assert check_boundaries(topology)
    assert np.array_equal(topology.boundary(1).toarray(), D)

    triangles.remove_empty_cells()
    edges.remove_empty_cells()

    assert len(edges) == 3
    assert len(triangles) == 1
    assert check_boundaries(topology)
