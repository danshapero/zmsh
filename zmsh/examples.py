from math import comb as binomial
import numpy as np
from .topology import Topology
from .geometry import Geometry


def simplex(dimension):
    r"""Return the topology of the standard d-simplex

    The `k`-cells of the `d`-simplex can be identified with the set of all bit
    vectors `v` of length `d + 1` such that `popcount(v) == k + 1`. For example
    the vertices of the 3-simplex are `0001`, `0010`, etc., the edges are
    `0011`, `0101`, and so forth. In particular, this means that the number of
    `k`-simplices of the `d`-simplex is `binomial(n + 1, k + 1)`.
    """
    if dimension <= 0:
        raise ValueError("Dimension must be > 0!")

    num_cells = [binomial(dimension + 1, k + 1) for k in range(dimension + 1)]
    topology = Topology(dimension=dimension, num_cells=num_cells)

    # TODO: Write some code that works for simplices of arbitrary dimension.
    # Probably it's too annoying so just use Z3 again so we don't have to think.
    if dimension == 1:
        edges = topology.cells(1)
        edges[0] = (0, 1), (-1, +1)
        points = np.array([-1.0, 1.0])
    elif dimension == 2:
        edges = topology.cells(1)
        edge_matrix = np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]], dtype=np.int8)
        edges[:] = (0, 1, 2), edge_matrix

        triangles = topology.cells(2)
        triangles[0] = (0, 1, 2), (+1, +1, +1)
        thetas = 2 * np.pi * np.array([0.0, 2 / 3, 4 / 3])
        points = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
    elif dimension == 3:
        edges = topology.cells(1)
        edge_matrix = np.array(
            [
                [+1, -1, 0, +1, 0, 0],
                [-1, 0, -1, 0, +1, 0],
                [0, +1, +1, 0, 0, -1],
                [0, 0, 0, -1, -1, +1],
            ],
            dtype=np.int8,
        )
        edges[:] = (0, 1, 2, 3), edge_matrix

        triangles = topology.cells(2)
        triangle_matrix = np.array(
            [
                [-1, +1, 0, 0],
                [-1, 0, +1, 0],
                [+1, 0, 0, -1],
                [0, -1, +1, 0],
                [0, +1, 0, -1],
                [0, 0, +1, -1],
            ],
            dtype=np.int8,
        )
        triangles[:] = tuple(range(6)), triangle_matrix

        tetrahedra = topology.cells(3)
        tetrahedra[0] = (0, 1, 2, 3), (+1, +1, +1, +1)
        points = np.array(
            [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
        )
    else:
        raise NotImplementedError("Haven't got to higher dimensions yet!")

    return Geometry(topology, points)


def cube(dimension):
    r"""Return the geometry of the standard d-cube

    The vertices of the `d`-cube can be identified with the set of all bit
    vectors `v` of length `d + 1`.
    """
    if dimension <= 0:
        raise ValueError("Dimension must be > 0!")

    num_cells = [
        2 ** (dimension - k) * binomial(dimension, k) for k in range(dimension + 1)
    ]
    topology = Topology(dimension=dimension, num_cells=num_cells)

    if dimension == 1:
        edges = topology.cells(1)
        edges[0] = (0, 1), (-1, +1)
        points = np.array([-1, 1], dtype=np.float64)
    elif dimension == 2:
        edges = topology.cells(1)
        edge_matrix = np.array(
            [[-1, 0, 0, +1], [+1, -1, 0, 0], [0, +1, -1, 0], [0, 0, +1, -1]],
            dtype=np.int8,
        )
        edges[:] = (0, 1, 2, 3), edge_matrix

        quads = topology.cells(2)
        quads[0] = (0, 1, 2, 3), (+1, +1, +1, +1)
        points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    elif dimension == 3:
        edges = topology.cells(1)
        edge_matrix = np.array(
            [
                [-1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, +1, 0, 0, 0, -1, 0, -1, 0],
                [+1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, +1, 0, +1, 0, 0, -1],
                [0, -1, +1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, +1, 0, 0, 0, -1, +1, 0],
                [0, +1, 0, +1, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, +1, 0, +1, 0, +1],
            ],
            dtype=np.int8,
        )
        edges[:] = tuple(range(num_cells[0])), edge_matrix

        quads = topology.cells(2)
        quad_matrix = np.array(
            [
                [+1, 0, 0, 0, 0, -1],
                [-1, 0, 0, +1, 0, 0],
                [-1, 0, +1, 0, 0, 0],
                [+1, 0, 0, 0, -1, 0],
                [0, 0, -1, 0, 0, +1],
                [0, 0, +1, -1, 0, 0],
                [0, 0, 0, 0, +1, -1],
                [0, 0, 0, +1, -1, 0],
                [0, -1, 0, 0, 0, +1],
                [0, +1, 0, -1, 0, 0],
                [0, +1, -1, 0, 0, 0],
                [0, -1, 0, 0, +1, 0],
            ],
            dtype=np.int8,
        )
        quads[:] = tuple(range(num_cells[1])), quad_matrix

        cubes = topology.cells(3)
        cubes[0] = tuple(range(num_cells[2])), (+1, +1, +1, +1, +1, +1)
        points = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
    else:
        raise NotImplementedError("Haven't got to higher dimensions yet!")

    return Geometry(topology, points)
