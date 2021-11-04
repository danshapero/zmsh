import numpy as np
import z3
from . import predicates, Topology, Transformation


def triangulate_skeleton(edges, with_exterior=False):
    r"""Given a skeleton set of edges, return the boundary matrix of triangles
    that fills the skeleton"""
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


def flip_edge(topology, edge):
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

    topology.cells(1)[edges] = vertices, E
    topology.cells(2)[triangles] = edges, T
