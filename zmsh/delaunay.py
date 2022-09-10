import numpy as np
import numpy.ma as ma
import z3
from . import predicates, Geometry, Topology, Transformation


def triangulate_skeleton(edges, with_exterior=False):
    r"""Given a skeleton set of edges, return the boundary matrix of triangles
    that fills the skeleton

    The exterior polygon is always the last column.
    """
    # From the Euler-Poincare formula V - E + T = 2 - 2G
    num_vertices, num_edges = edges.shape
    num_polygons = 2 + num_edges - num_vertices

    transformation = Transformation((num_edges, num_polygons), name="t")
    transformation.constrain_range(-1, +1)
    transformation.constrain_num_faces(3, columns=range(num_polygons - 1))
    transformation.constrain_num_cofaces(2)
    if with_exterior:
        for edge in range(num_edges):
            constraint = transformation.matrix[edge, -1] <= 0
            transformation.solver.add(constraint)

    transformation.constrain_boundary(edges)
    Z = np.ones((num_polygons, 1), dtype=int)
    transformation.constrain_coboundary(Z)

    return transformation


def flip_edge(topology: Topology, edge: int):
    triangles = topology.cocells(1)[edge][0]
    if len(triangles) != 2:
        raise ValueError("Edge must have two triangles in its coboundary!")

    edges, signs = topology.cells(2)[triangles]
    exterior = -np.sum(signs, axis=1)
    vertices, E = topology.cells(1)[edges]

    column = np.where(edges == edge)[0][0]
    rows = (E[:, column] == 0).nonzero()[0]
    if len(rows) != 2:
        raise ValueError("This is very bad")
    E[:, column] = 0
    E[rows, column] = (-1, +1)

    # Set up the transformation to the new quadrilateral
    transformation = triangulate_skeleton(E)

    # Make sure the new quadrilateral is oriented the same way w.r.t. the
    # exterior as the old quad
    for row in range(transformation.matrix.shape[0]):
        constraint = transformation.matrix[row, -1] == int(exterior[row])
        transformation.solver.add(constraint)

    # Find a satisfactory transformation and drop the exterior
    T = transformation.compute()[:, :-1]

    return (vertices, edges, triangles), (E, T)


def split_polygon(topology: Topology, polygon: int, vertex: int):
    r"""Return the boundary matrices for splitting a cell of the topology along
    a vertex

    Returns
    -------
    cell_indices, boundary_matrices
        Arrays of the vertex, edge, and polygon indices, masked if the caller
        must assign the index, and the local boundary matrices
    """
    edges, esigns = topology.cells(2)[polygon]
    vertices, vsigns = topology.cells(1)[edges]

    # Take the edge -> vertex boundary matrix, stick a copy of the identity on
    # the right, and a row of -1s below that
    n = len(vertices)
    I = np.eye(n, dtype=np.int8)
    row = np.hstack((np.zeros(n, dtype=np.int8), -np.ones(n, dtype=np.int8)))
    E = np.vstack((np.hstack((vsigns, I)), row))

    # Compute the poly -> edge boundary matrix. Note how we flip all the signs
    # around the exterior from what they were in the original polygon, rather
    # than setting them all to be -1.
    transformation = triangulate_skeleton(E)
    for row in range(n):
        constraint = transformation.matrix[row, -1] == -int(esigns[row])
        transformation.solver.add(constraint)

    T = transformation.compute()[:, :-1]

    # Return the vertices, edges, and polygons of the split polygon; the IDs
    # where the new edges and polygons will go need to be assigned by the
    # caller, for which we use masked arrays.
    vertices = np.concatenate((vertices, (vertex,)))
    edges = ma.masked_equal(np.concatenate((edges, n * (-1,))), -1)
    polys = ma.masked_equal(np.concatenate(((polygon,), (n - 1) * (-1,))), -1)
    return (vertices, edges, polys), (E, T)


def locate_point(geometry: Geometry, z: np.ndarray):
    r"""Return the index of the cell of the topology containing a given point
    if it exists, or None if it doesn't

    Notes
    -----
    This runs in O(number of triangles) time. This is inefficient and should be
    replaced with a better spatial data structure that runs in logarithmic time.
    """
    if (geometry.dimension != 2) or (geometry.topology.dimension != 2):
        raise NotImplementedError("Point location only implemented in 2D!")

    edges = geometry.topology.cells(1)
    polygons = geometry.topology.cells(2)

    for index, (faces, signs) in enumerate(polygons):
        inside = True
        for face, sign in zip(faces, signs):
            vertices, vsigns = edges[face]
            x, y = geometry.points[vertices, :]
            area = sign * vsigns[1] * predicates.volume(x, y, z)
            inside &= area >= 0

        if inside:
            return index

    return None
