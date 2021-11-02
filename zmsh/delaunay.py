import numpy as np
import z3
from . import predicates, Topology, Transformation


def triangulate_skeleton(edges):
    r"""Given a skeleton set of edges, return the boundary matrix of triangles
    that fills the skeleton"""
    # From the Euler-Poincare formula V - E + T = 2 - 2G
    num_vertices, num_edges = edges.shape
    num_polygons = 2 + num_edges - num_vertices

    transformation = Transformation((num_edges, num_polygons), name="t")
    transformation.constrain_range(-1, +1)
    transformation.constrain_num_faces(3, columns=range(num_polygons - 1))
    transformation.constrain_num_cofaces(2)
    # TODO: Something about the cell representing the exterior being all -1s

    transformation.constrain_boundary(edges)
    Z = np.ones((num_polygons, 1), dtype=int)
    transformation.constrain_coboundary(Z)

    return transformation.compute()


def flip_edge(topology, edge):
    triangles = topology.cocells(1)[edge][0]
    if len(triangles) != 2:
        raise ValueError("Edge must have two triangles in its coboundary!")

    edges = topology.cells(2)[triangles][0]
    vertices, E = topology.cells(1)[edges]

    column = np.where(edges == edge)[0][0]
    rows = (E[:, column] == 0).nonzero()[0]
    if len(rows) != 2:
        raise ValueError("This is very bad")
    E[:, column] = 0
    E[rows, column] = (-1, +1)

    T = triangulate_skeleton(E)
    # Get rid of the exterior polygon
    T = T[:, :-1]

    topology.cells(1)[edges] = vertices, E
    topology.cells(2)[triangles] = edges, T
