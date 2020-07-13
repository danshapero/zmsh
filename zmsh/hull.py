import numpy as np
from . import predicates
from .topology import Topology

class ConvexHullMachine(object):
    def __init__(self, points):
        if points.shape[1] != 2:
            raise NotImplementedError("Haven't got to 3D hulls yet!")

        self._points = points

        index_xmin = np.argmin(points[:, 0])
        index_xmax = np.argmax(points[:, 0])

        if index_xmin == index_xmax:
            raise ValueError("All points are collinear!")

        self._candidates = {index for index in range(len(points))}
        self._candidates -= {index_xmin, index_xmax}

        self._topology = Topology(1)
        self._topology.set_num_cells(0, len(points))
        self._topology.set_num_cells(1, len(points))

        self._topology.set_cell(1, 0, (index_xmin, index_xmax), (-1, +1))
        self._topology.set_cell(1, 1, (index_xmax, index_xmin), (-1, +1))

        self._edge_queue = [0, 1]
        self._num_edges = 2

    @property
    def candidates(self):
        r"""The set of indices of all points that might be in the hull"""
        return self._candidates

    @property
    def edge_queue(self):
        r"""The list of edges that still need inspecting"""
        return self._edge_queue

    @property
    def topology(self):
        r"""The current topology for the hull"""
        return self._topology

    @property
    def num_edges(self):
        r"""The number of edges that have been added to the hull so far"""
        return self._num_edges

    def best_candidate(self, edge_index):
        r"""Return the index of the candidate point that forms the triangle
        of largest area with the given edge"""
        vertices, incidence = self._topology.cell(1, edge_index)
        if incidence[0] == +1:
            vertices = (vertices[1], vertices[0])

        x = self._points[vertices[0], :]
        y = self._points[vertices[1], :]

        best_index = None
        best_area = np.inf
        for index in self._candidates:
            z = self._points[index, :]
            area = predicates.area(x, y, z)
            if area < best_area:
                best_index = index
                best_area = area

        return best_index, best_area

    def is_done(self):
        return (not self._edge_queue) or (not self._candidates)

    def step(self):
        r"""Process the next edge -- either do nothing, or split it if there's
        an extreme point across from it"""
        if self.is_done():
            return

        # Pop the next edge and find any extreme points across from it; if
        # there aren't any, we're done.
        edge_index = self._edge_queue.pop(0)
        extreme_vertex_index, area = self.best_candidate(edge_index)
        if area > 0:
            return

        # Split the edge at the extreme point
        vertices, incidence = self.topology.cell(1, edge_index)
        if incidence[0] == +1:
            vertices = (vertices[1], vertices[0])

        faces1 = (vertices[0], extreme_vertex_index)
        faces2 = (extreme_vertex_index, vertices[1])

        self.topology.set_cell(1, edge_index, faces1, (-1, +1))
        self.topology.set_cell(1, self._num_edges, faces2, (-1, +1))

        # Filter out all candidate points inside the triangle formed by the old
        # edge and the two new edges
        dropouts = {extreme_vertex_index}
        x = self._points[vertices[0], :]
        y = self._points[extreme_vertex_index, :]
        z = self._points[vertices[1], :]
        for index in self._candidates:
            w = self._points[index, :]
            inside = (
                (predicates.area(x, y, w) > 0) and
                (predicates.area(y, z, w) > 0) and
                (predicates.area(z, x, w) > 0)
            )
            if inside:
                dropouts.add(index)

        self._edge_queue.append(edge_index)
        self._edge_queue.append(self._num_edges)

        self._candidates -= dropouts
        self._num_edges += 1

    def run(self):
        while not self.is_done():
            self.step()

        self.topology.set_num_cells(1, self._num_edges)
        return self.topology


def convex_hull(points):
    r"""Calculate the convex hull of a 2D point set"""
    machine = ConvexHullMachine(points)
    return machine.run()
