import numpy as np
from numpy import random
import zmsh


def test_triangulate_triangle():
    edges = np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]], dtype=np.int8)

    triangles = zmsh.triangulate_skeleton(edges, with_exterior=True).compute()
    assert triangles is not None
    assert triangles.shape == (3, 2)
    triangles_expected = np.array([[+1, -1], [+1, -1], [+1, -1]], dtype=np.int8)
    assert np.array_equal(triangles, triangles_expected)
    assert np.max(np.abs(edges @ triangles)) == 0

    # Flip the orientation of one edge and see if we can still triangulate it
    edges[:, 2] *= -1
    triangles = zmsh.triangulate_skeleton(edges, with_exterior=False).compute()
    assert triangles is not None
    assert np.max(np.abs(edges @ triangles)) == 0

    # But if we ask for a consistent sign w.r.t. the exterior we can't
    triangles = zmsh.triangulate_skeleton(edges, with_exterior=True).compute()
    assert triangles is None


def test_triangulate_quadrilateral():
    edges = np.array(
        [[-1, 0, +1, 0, 0], [+1, -1, 0, -1, 0], [0, +1, -1, 0, +1], [0, 0, 0, +1, -1]],
        dtype=np.int8,
    )

    triangles = zmsh.triangulate_skeleton(edges, with_exterior=True).compute()
    assert triangles is not None
    assert triangles.shape == (5, 3)
    assert np.max(np.abs(edges @ triangles)) == 0


def test_edge_flip():
    topology = zmsh.Topology(dimension=2, num_cells=(4, 5, 2))

    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    edges[(1, 2), 1] = (-1, +1)
    edges[(2, 0), 2] = (-1, +1)
    edges[(1, 3), 3] = (-1, +1)
    edges[(3, 2), 4] = (-1, +1)

    triangles = topology.cells(2)
    triangles[(0, 1, 2), 0] = (+1, +1, +1)
    triangles[(3, 4, 1), 1] = (+1, +1, -1)

    d_2 = topology.boundary(2).copy()
    d_1 = topology.boundary(1).copy()

    edge_id = 1
    cell_ids, (D_1, D_2) = zmsh.flip_edge(topology, edge_id)
    edges[cell_ids[0], cell_ids[1]] = D_1
    triangles[cell_ids[1], cell_ids[2]] = D_2

    D_2 = topology.boundary(2)
    D_1 = topology.boundary(1)

    assert np.max(np.abs(D_2 - d_2)) > 0
    assert np.max(np.abs(D_1 - d_1)) > 0
    assert np.max(np.abs(D_1 @ D_2)) == 0

    assert set(edges[edge_id][0]) == {0, 3}

    cell_ids, (D_1, D_2) = zmsh.flip_edge(topology, edge_id)
    edges[cell_ids[0], cell_ids[1]] = D_1
    triangles[cell_ids[1], cell_ids[2]] = D_2
    assert set(edges[edge_id][0]) == {1, 2}


def test_complex_edge_flip():
    topology = zmsh.Topology(dimension=2, num_cells=[5, 7, 3])

    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    edges[(1, 2), 1] = (-1, +1)
    edges[(2, 0), 2] = (-1, +1)
    edges[(1, 3), 3] = (-1, +1)
    edges[(3, 2), 4] = (-1, +1)
    edges[(3, 4), 5] = (-1, +1)
    edges[(4, 2), 6] = (-1, +1)

    triangles = topology.cells(2)
    triangles[(0, 1, 2), 0] = (+1, +1, +1)
    triangles[(3, 4, 1), 1] = (+1, +1, -1)
    triangles[(4, 5, 6), 2] = (-1, +1, +1)

    D_1 = topology.boundary(1)
    D_2 = topology.boundary(2)
    assert np.max(np.abs(D_1 @ D_2)) == 0
    cell_ids, (D_1, D_2) = zmsh.flip_edge(topology, 4)
    topology.cells(1)[cell_ids[0], cell_ids[1]] = D_1
    topology.cells(2)[cell_ids[1], cell_ids[2]] = D_2

    D_1 = topology.boundary(1)
    D_2 = topology.boundary(2)
    assert np.max(np.abs(D_1 @ D_2)) == 0


def test_splitting_polygon():
    topology = zmsh.Topology(dimension=2, num_cells=(5, 8, 4))
    edges = topology.cells(1)
    vertex_ids = (0, 1, 2, 3)
    D = np.array([[-1, 0, 0, +1], [+1, -1, 0, 0], [0, +1, -1, 0], [0, 0, +1, -1]])
    edges[vertex_ids, :4] = D

    polys = topology.cells(2)
    polys[(0, 1, 2, 3), 0] = (+1, +1, +1, +1)

    D_1 = topology.boundary(1)
    D_2 = topology.boundary(2)
    assert np.max(np.abs(D_1 @ D_2)) == 0

    cell_ids, boundary_matrices = zmsh.split_polygon(topology, 0, 4)

    # Check that we get good boundary matrices
    d_1, d_2 = boundary_matrices
    assert np.max(np.abs(d_1 @ d_2)) == 0

    # Check that the result covers all vertices, that the caller has to decide
    # how to assign a certain number of edge and polygon IDs
    vertex_ids, edge_ids, poly_ids = cell_ids
    assert set(vertex_ids) == set(range(5))
    assert sum(edge_ids.mask) == 4
    assert sum(poly_ids.mask) == 3

    # Find the IDs of empty edges and polygons and assign these (arbitrarily)
    # to where we will put the new edges and polygons
    empty_edge_ids = [i for i in range(len(edges)) if len(edges[i][0]) == 0]
    empty_poly_ids = [k for k in range(len(polys)) if len(polys[k][0]) == 0]
    edge_ids[edge_ids.mask] = empty_edge_ids
    poly_ids[poly_ids.mask] = empty_poly_ids

    # Update the topology and check that it's still good
    polys[edge_ids, poly_ids] = d_2
    edges[vertex_ids, edge_ids] = d_1

    D_1 = topology.boundary(1)
    D_2 = topology.boundary(2)
    assert np.max(np.abs(D_1 @ D_2)) == 0


def test_point_location():
    topology = zmsh.Topology(dimension=2, num_cells=(4, 5, 2))

    edges = topology.cells(1)
    edges[(0, 1), 0] = (-1, +1)
    edges[(1, 2), 1] = (-1, +1)
    edges[(2, 0), 2] = (-1, +1)
    edges[(1, 3), 3] = (-1, +1)
    edges[(3, 2), 4] = (-1, +1)

    triangles = topology.cells(2)
    triangles[(0, 1, 2), 0] = (+1, +1, +1)
    triangles[(3, 4, 1), 1] = (+1, +1, -1)

    points = np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, +1.0], [+1.0, 0.0]])
    geometry = zmsh.Geometry(topology, points)

    z = np.array([-0.5, 0.0])
    assert zmsh.delaunay.locate_point(geometry, z) == 0

    z = np.array([+0.5, 0.0])
    assert zmsh.delaunay.locate_point(geometry, z) == 1

    z = np.array([-1.0, -1.0])
    assert zmsh.delaunay.locate_point(geometry, z) is None
