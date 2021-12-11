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
    edges[0] = (0, 1), (-1, +1)

    faces, signs = edges[0]
    assert np.array_equal(faces, (0, 1))
    assert np.array_equal(signs, (-1, +1))
    assert check_boundaries(topology)

    covertices = topology.cocells(0)
    cofaces, signs = covertices[0]
    assert np.array_equal(cofaces, (0,))

    # Now make an edge with two endpoints and check that the vertex * edge
    # matrix is non-zero
    edges[0] = (0, 1), (+1, +1)
    assert not check_boundaries(topology)


def test_setting_multiple_cells():
    topology = zmsh.Topology(dimension=1, num_cells=[3, 2])
    edges = topology.cells(1)
    edges[(0, 1)] = (0, 1, 2), np.array([[-1, 0], [+1, -1], [0, +1]])
    assert check_boundaries(topology)
    assert matrix_norm(topology.boundary(1)) != 0
    edges[:2] = (0, 1, 2), np.array([[0, -1], [-1, +1], [+1, 0]])
    assert check_boundaries(topology)
    assert matrix_norm(topology.boundary(1)) != 0
    edges[:] = (0, 1, 2), np.array([[-1, 0], [+1, -1], [0, +1]])
    assert check_boundaries(topology)
    assert matrix_norm(topology.boundary(1)) != 0


def test_setting_cells_integral_index():
    # This can fail if you do type checking naively because
    # `isinstance(np.int64(0), int)` returns `False`.
    topology = zmsh.Topology(dimension=1, num_cells=[2, 1])
    edges = topology.cells(1)
    index = np.int64(0)
    edges[index] = (0, 1), (-1, +1)
    assert check_boundaries(topology)
    vertices, signs = edges[index]
    assert signs.shape == (2,)


def test_setting_cells_bad_index():
    topology = zmsh.Topology(dimension=1, num_cells=[2, 1])
    edges = topology.cells(1)
    with pytest.raises(Exception):
        index = "a"
        edges[index] = (0, 1), (-1, +1)


def test_no_excess_zero_entries():
    topology = zmsh.Topology(dimension=1, num_cells=[3, 2])
    A = topology.boundary(1)
    assert A.count_nonzero() == 0
    edges = topology.cells(1)
    edges[(0, 1)] = (0, 1, 2), np.array([[-1, 0], [+1, -1], [0, +1]])
    assert A.count_nonzero() == 4


def test_getting_empty_cell():
    topology = zmsh.Topology(dimension=1, num_cells=[2, 2])
    edges = topology.cells(1)
    edges[0] = (0, 1), (-1, +1)
    faces, signs = edges[1]
    assert np.array_equal(faces, ())
    assert np.array_equal(signs, ())


def test_reset_cell():
    topology = zmsh.Topology(dimension=1, num_cells=[4, 1])

    edges = topology.cells(1)
    edges[0] = (0, 1), (-1, +1)
    edges[0] = (2, 3), (-1, +1)
    vertices, signs = edges[0]
    assert np.array_equal(vertices, (2, 3))


def test_triangle():
    topology = zmsh.Topology(dimension=2, num_cells=[3, 3, 1])

    edges = topology.cells(1)
    edges[:] = (0, 1, 2), np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]])

    triangles = topology.cells(2)
    triangles[:] = (0, 1, 2), np.array([[+1, +1, +1]])

    # Check that the faces and cofaces make sense
    faces, signs = triangles[0]
    assert np.array_equal(faces, (0, 1, 2))

    faces, signs = edges[(0, 1)]
    assert np.array_equal(faces, (0, 1, 2))
    assert np.array_equal(signs, np.array([[-1, 0], [+1, -1], [0, +1]]))

    faces, signs = edges[:2]
    assert np.array_equal(faces, (0, 1, 2))
    signs_expected = np.array([[-1, 0], [+1, -1], [0, +1]])
    assert np.array_equal(signs, signs_expected)

    coedges = topology.cocells(1)
    cofaces, signs = coedges[0]
    assert np.array_equal(cofaces, (0,))

    covertices = topology.cocells(0)
    cofaces, signs = covertices[(0, 1)]
    assert np.array_equal(cofaces, (0, 1, 2))
    signs_expected = np.array([[-1, +1], [0, -1], [+1, 0]])
    assert np.array_equal(signs, signs_expected)

    # Check that there are no non-zero ∂∂-products
    assert check_boundaries(topology)

    # Now change the triangle so that one of the edges is reversed and check
    # that the triangle * edge matrix is non-zero
    triangles[0] = (0, 1, 2), (+1, -1, +1)
    assert not check_boundaries(topology)


def test_triangle_pair():
    topology = zmsh.Topology(dimension=2, num_cells=(4, 5, 2))

    edges = topology.cells(1)
    edges[0] = (0, 1), (-1, +1)
    edges[1] = (1, 2), (-1, +1)
    edges[2] = (2, 0), (-1, +1)
    edges[3] = (0, 3), (-1, +1)
    edges[4] = (3, 1), (-1, +1)

    triangles = topology.cells(2)
    triangles[0] = (0, 1, 2), (+1, +1, +1)
    triangles[1] = (0, 3, 4), (-1, +1, +1)

    # Check that the faces and cofaces make sense
    coedges = topology.cocells(1)
    cofaces, signs = coedges[0]
    assert np.array_equal(cofaces, (0, 1))
    assert np.array_equal(signs, (+1, -1))

    covertices = topology.cocells(0)
    cofaces, signs = covertices[(0, 1)]
    assert np.array_equal(cofaces, (0, 1, 2, 3, 4))

    # Check that there are no non-zero ∂∂-products
    assert check_boundaries(topology)
    assert topology.coboundary(1).shape == (2, 5)

    # Make a topologically valid transformation -- reverse all the incidences
    # of a single cell
    triangles[1] = (0, 3, 4), (+1, -1, -1)
    assert check_boundaries(topology)


def test_iterating_over_cells():
    topology = zmsh.Topology(dimension=1, num_cells=(3, 2))

    topology.cells(1)[0] = (0, 1), (-1, +1)
    for faces, signs in topology.cells(1):
        assert np.array_equal(faces, (0, 1)) ^ (len(faces) == 0)


def test_example_topologies():
    for dimension in [1, 2, 3]:
        topology = zmsh.examples.simplex(dimension)
        assert check_boundaries(topology)
        assert topology.is_simplicial()

    for dimension in [1, 2, 3]:
        topology = zmsh.examples.cube(dimension)
        assert check_boundaries(topology)
        assert topology.is_cubical()

    for dimension in [1, 2]:
        topology = zmsh.examples.torus(dimension)
        assert check_boundaries(topology)
        assert topology.is_cubical()
