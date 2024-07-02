from math import comb as binomial
import numpy as np
import predicates
from .topology import Topology
from .geometry import Geometry
from . import simplicial
from .convex_hull import extreme_points as _extreme_points, ConvexHullMachine


def extreme_points(points: np.ndarray):
    r"""Given a `d`-dimensional point set, return the indices of `d + 1` points
    that are on the convex hull of the parabolic lift"""
    square_magnitudes = np.sum(points**2, axis=1)
    lifted_points = np.column_stack((points, square_magnitudes))
    return _extreme_points(lifted_points)


class DelaunayMachine:
    def __init__(self, points: np.ndarray):
        n = len(points)
        indices = extreme_points(points)

        d = points.shape[1]
        num_cells = [n] + [binomial(d + 1, k + 1) for k in range(1, d)] + [2]
        topology = Topology(dimension=d, num_cells=num_cells)
        geometry = Geometry(topology, points.copy())

        matrices = simplicial.simplex_to_chain_complex(d)[1:]
        cell_ids = [indices] + [tuple(range(D.shape[1])) for D in matrices]
        for k, D in enumerate(matrices, start=1):
            cells = geometry.topology.cells(k)
            cells[cell_ids[k - 1], cell_ids[k]] = D

        cells = geometry.topology.cells(d)
        cells[cell_ids[-2], 1] = -matrices[-1]

        self._convex_hull_machine = ConvexHullMachine(
            geometry, signed_volume=predicates.insphere
        )

    @property
    def geometry(self):
        return self._convex_hull_machine.geometry

    def is_done(self):
        return self._convex_hull_machine.is_done()

    def step(self):
        self._convex_hull_machine.step()

    def finalize(self):
        geometry = self._convex_hull_machine.finalize()
        self._convex_hull_machine = None

        topology = geometry.topology
        cells = topology.cells(topology.dimension)
        cell_ids_to_remove = []
        for cell_id in range(len(cells)):
            faces_ids, matrices = cells.closure(cell_id)
            orientation = simplicial.orientation(matrices)
            X = geometry.points[faces_ids[0], :]
            volume = orientation * predicates.volume(X.T)
            if volume > 0:
                cell_ids_to_remove.append(cell_id)

        D = topology.boundary(topology.dimension)
        for cell_id in cell_ids_to_remove:
            D[:, cell_id] = 0

        for k in range(topology.dimension - 1, 0, -1):
            cocells = topology.cocells(k)
            face_ids_to_remove = []
            for face_id, (cell_ids, signs) in enumerate(cocells):
                if len(cell_ids) == 0:
                    face_ids_to_remove.append(face_id)

            D = topology.boundary(k)
            for face_id in face_ids_to_remove:
                D[:, face_id] = 0

        for k in range(1, topology.dimension + 1):
            topology.cells(k).remove_empty_cells()

        return geometry

    def run(self):
        while not self.is_done():
            self.step()

        return self.finalize()


def delaunay(points):
    r"""Compute the Delaunay mesh of a point set"""
    machine = DelaunayMachine(points)
    return machine.run()
