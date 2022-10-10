import numpy as np
from . import predicates
from .topology import Topology
from .geometry import Geometry


class ConvexHullMachine:
    def __init__(self, points):
        if points.shape[1] != 2:
            raise NotImplementedError("Haven't got to 3D hulls yet!")

        centroid = np.mean(points, axis=0)
        index1 = np.argmax(np.sum((points - centroid) ** 2, axis=1))
        extremal_point = points[index1]
        index2 = np.argmax(np.sum((points - extremal_point) ** 2, axis=1))

        # TODO: check for collinearity

        self._candidates = {index for index in range(len(points))}
        self._candidates -= {index1, index2}

        n = len(points)
        topology = Topology(dimension=1, num_cells=(n, n))
        self._geometry = Geometry(topology, points.copy())

        edges = self._geometry.topology.cells(1)
        edges[(index1, index2), (0, 1)] = np.array([[-1, +1], [+1, -1]])

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
    def geometry(self):
        r"""The current topology for the hull"""
        return self._geometry

    @property
    def num_edges(self):
        r"""The number of edges that have been added to the hull so far"""
        return self._num_edges

    def best_candidate(self, edge_index):
        r"""Return the index of the candidate point that forms the triangle
        of largest area with the given edge"""
        vertex_ids, signs = self.geometry.topology.cells(1)[edge_index]
        if signs[0] == +1:
            vertex_ids = (vertex_ids[1], vertex_ids[0])

        x = self.geometry.points[vertex_ids[0], :]
        y = self.geometry.points[vertex_ids[1], :]

        best_index = None
        best_area = np.inf
        for index in self._candidates:
            z = self.geometry.points[index, :]
            area = predicates.volume(x, y, z)
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
        vertex_ids, signs = self.geometry.topology.cells(1)[edge_index]
        if signs[0] == +1:
            vertex_ids = (vertex_ids[1], vertex_ids[0])

        edges = self.geometry.topology.cells(1)
        edge_ids = (edge_index, self._num_edges)
        vertex_ids = (vertex_ids[0], extreme_vertex_index, vertex_ids[1])
        signs = np.array([[-1, 0], [+1, -1], [0, +1]])
        edges[vertex_ids, edge_ids] = signs

        # Filter out all candidate points inside the triangle formed by the old
        # edge and the two new edges
        dropouts = {extreme_vertex_index}
        x = self.geometry.points[vertex_ids[0], :]
        y = self.geometry.points[extreme_vertex_index, :]
        z = self.geometry.points[vertex_ids[1], :]
        for index in self._candidates:
            w = self.geometry.points[index, :]
            inside = (
                (predicates.volume(x, y, w) > 0)
                and (predicates.volume(y, z, w) > 0)
                and (predicates.volume(z, x, w) > 0)
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

        self.geometry.topology.cells(1).resize(self._num_edges)
        return self.geometry


def convex_hull(points):
    r"""Calculate the convex hull of a 2D point set"""
    machine = ConvexHullMachine(points)
    return machine.run()
