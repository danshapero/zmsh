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
    edges[0] = (0, 1), (-1, +1)
    edges[1] = (1, 2), (-1, +1)
    edges[2] = (2, 0), (-1, +1)
    edges[3] = (1, 3), (-1, +1)
    edges[4] = (3, 2), (-1, +1)

    triangles = topology.cells(2)
    triangles[0] = (0, 1, 2), (+1, +1, +1)
    triangles[1] = (3, 4, 1), (+1, +1, -1)

    d_2 = topology.boundary(2).copy()
    d_1 = topology.boundary(1).copy()

    edge = 1
    (vertices, edges, triangles), (D_1, D_2) = zmsh.flip_edge(topology, edge)
    topology.cells(1)[edges] = vertices, D_1
    topology.cells(2)[triangles] = edges, D_2

    D_2 = topology.boundary(2)
    D_1 = topology.boundary(1)

    assert np.max(np.abs(D_2 - d_2)) > 0
    assert np.max(np.abs(D_1 - d_1)) > 0
    assert np.max(np.abs(D_1 @ D_2)) == 0

    assert set(topology.cells(1)[edge][0]) == {0, 3}

    (vertices, edges, triangles), (D_1, D_2) = zmsh.flip_edge(topology, edge)
    topology.cells(1)[edges] = vertices, D_1
    topology.cells(2)[triangles] = edges, D_2
    assert set(topology.cells(1)[edge][0]) == {1, 2}


def test_complex_edge_flip():
    topology = zmsh.Topology(dimension=2, num_cells=[5, 7, 3])

    edges = topology.cells(1)
    edges[0] = (0, 1), (-1, +1)
    edges[1] = (1, 2), (-1, +1)
    edges[2] = (2, 0), (-1, +1)
    edges[3] = (1, 3), (-1, +1)
    edges[4] = (3, 2), (-1, +1)
    edges[5] = (3, 4), (-1, +1)
    edges[6] = (4, 2), (-1, +1)

    triangles = topology.cells(2)
    triangles[0] = (0, 1, 2), (+1, +1, +1)
    triangles[1] = (3, 4, 1), (+1, +1, -1)
    triangles[2] = (4, 5, 6), (-1, +1, +1)

    D_1 = topology.boundary(1)
    D_2 = topology.boundary(2)
    assert np.max(np.abs(D_1 @ D_2)) == 0
    (vertices, edges, triangles), (D_1, D_2) = zmsh.flip_edge(topology, 4)
    topology.cells(1)[edges] = vertices, D_1
    topology.cells(2)[triangles] = edges, D_2

    D_1 = topology.boundary(1)
    D_2 = topology.boundary(2)
    assert np.max(np.abs(D_1 @ D_2)) == 0


def test_point_location():
    topology = zmsh.Topology(dimension=2, num_cells=(4, 5, 2))

    edges = topology.cells(1)
    edges[0] = (0, 1), (-1, +1)
    edges[1] = (1, 2), (-1, +1)
    edges[2] = (2, 0), (-1, +1)
    edges[3] = (1, 3), (-1, +1)
    edges[4] = (3, 2), (-1, +1)

    triangles = topology.cells(2)
    triangles[0] = (0, 1, 2), (+1, +1, +1)
    triangles[1] = (3, 4, 1), (+1, +1, -1)

    points = np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, +1.0], [+1.0, 0.0]])
    geometry = zmsh.Geometry(topology, points)

    z = np.array([-0.5, 0.0])
    assert zmsh.delaunay.locate_point(geometry, z) == 0

    z = np.array([+0.5, 0.0])
    assert zmsh.delaunay.locate_point(geometry, z) == 1

    z = np.array([-1.0, -1.0])
    assert zmsh.delaunay.locate_point(geometry, z) is None
