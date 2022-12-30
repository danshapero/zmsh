from math import comb as binomial
from typing import Callable
import numpy as np
from . import predicates, simplicial, transformations
from .topology import Topology
from .geometry import Geometry


def extreme_points(points: np.ndarray):
    r"""Given a `d`-dimensional point set, return the indices of `d` points
    that are on the convex hull"""
    dimension = points.shape[1]
    centroid = np.mean(points, axis=0)
    indices = [np.argmax(np.sum((points - centroid) ** 2, axis=1))]
    x_0 = points[indices[0]]
    indices.append(np.argmax(np.sum((points - x_0) ** 2, axis=1)))

    Id = np.eye(dimension)
    while len(indices) < dimension:
        A = np.array([points[k] - x_0 for k in indices[1:]]).T
        Q, R = np.linalg.qr(A)
        delta = (Id - Q @ Q.T) @ (points - x_0).T
        norms = np.linalg.norm(delta, axis=0)
        indices.append(np.argmax(norms))

    return indices


class VisibilityGraph:
    def __init__(self, geometry, signed_volume):
        covertices = geometry.topology.cocells(0)
        d = geometry.topology.dimension
        cells = geometry.topology.cells(d)
        cell_to_vertex = {}
        for cell_id in range(len(cells)):
            face_ids, matrices = cells.closure(cell_id)
            orientation = simplicial.orientation(matrices)
            X = geometry.points[face_ids[0], :]
            entry = {}
            for vertex_id in range(len(covertices)):
                edge_ids, signs = covertices[vertex_id]
                if len(edge_ids) == 0:
                    z = geometry.points[vertex_id, :]
                    volume = orientation * signed_volume(z, *X)
                    if volume <= 0:
                        entry[vertex_id] = volume

            if entry:
                cell_to_vertex[cell_id] = entry

        vertex_to_cell = {}
        for cell_id, entry in cell_to_vertex.items():
            for vertex_id, volume in entry.items():
                if vertex_id not in vertex_to_cell:
                    vertex_to_cell[vertex_id] = {}
                vertex_to_cell[vertex_id][cell_id] = volume

        self.geometry = geometry
        self.signed_volume = signed_volume
        self.cell_to_vertex = cell_to_vertex
        self.vertex_to_cell = vertex_to_cell

    def get_next_vertex_and_cells(self):
        if not self.cell_to_vertex:
            return None, None

        # TODO: This is an O(m) operation, store #nz somewhere to make it fast
        num_visible = {
            cell_id: len(entry) for cell_id, entry in self.cell_to_vertex.items()
        }
        cell_id = max(num_visible, key=lambda cell_id: num_visible[cell_id])
        entry = self.cell_to_vertex[cell_id]
        extreme_vertex_id = min(entry, key=lambda vertex_id: entry[vertex_id])
        visible_cell_ids = list(self.vertex_to_cell[extreme_vertex_id].keys())

        return extreme_vertex_id, visible_cell_ids

    def remove(self, cell_ids):
        for cell_id in cell_ids:
            entry = self.cell_to_vertex.pop(cell_id, {})
            for vertex_id, _ in entry.items():
                if vertex_id in self.vertex_to_cell:
                    self.vertex_to_cell[vertex_id].pop(cell_id, None)
                    if len(self.vertex_to_cell[vertex_id]) == 0:
                        self.vertex_to_cell.pop(vertex_id)

    def update(self, new_cell_ids, old_cell_ids):
        topology = self.geometry.topology
        dimension = topology.dimension
        cells = topology.cells(dimension)
        cofaces = topology.cocells(dimension - 1)

        face_ids = cells[old_cell_ids][0]
        cell_ids = cofaces[face_ids][0]

        for new_cell_id in new_cell_ids:
            faces_ids, matrices = cells.closure(new_cell_id)
            orientation = simplicial.orientation(matrices)
            X = self.geometry.points[faces_ids[0]]

            entry = {}
            for cell_id in cell_ids:
                for vertex_id, _ in self.cell_to_vertex.get(cell_id, {}).items():
                    if vertex_id not in faces_ids[0]:
                        z = self.geometry.points[vertex_id]
                        volume = orientation * self.signed_volume(z, *X)
                        if volume <= 0:
                            entry[vertex_id] = volume

            if entry:
                self.cell_to_vertex[new_cell_id] = entry
                for vertex_id, volume in entry.items():
                    self.vertex_to_cell[vertex_id][new_cell_id] = volume

        self.remove(old_cell_ids)


class ConvexHullMachine:
    def _init_with_geometry(
        self, geometry: Geometry, signed_volume: Callable = predicates.volume
    ):
        self._geometry = geometry
        d = self._geometry.topology.dimension

        # Store which numeric IDs can still be assigned to cells of each
        # dimension
        self._free_cell_ids = [set() for k in range(d + 1)]
        for k in range(1, d + 1):
            cells = self._geometry.topology.cells(k)
            self._free_cell_ids[k] = set(
                cell_id
                for cell_id, (face_ids, signs) in enumerate(cells)
                if len(face_ids) == 0
            )

        self._visible = VisibilityGraph(self._geometry, signed_volume)

    def _init_with_points(
        self, points: np.ndarray, signed_volume: Callable = predicates.volume
    ):
        # TODO: check for collinearity
        n = len(points)
        indices = extreme_points(points)

        d = points.shape[1] - 1
        num_cells = [n] + [binomial(d + 1, k + 1) for k in range(1, d)] + [2]
        topology = Topology(dimension=d, num_cells=num_cells)
        geometry = Geometry(topology, points.copy())

        matrices = simplicial.simplex_to_chain_complex(d)[1:]
        cell_ids = [indices] + [tuple(range(D.shape[1])) for D in matrices]
        for k, D in enumerate(matrices, start=1):
            cells = geometry.topology.cells(k)
            cells[cell_ids[k - 1], cell_ids[k]] = D

        # Take the initial simplex and add its mirror image
        cells = geometry.topology.cells(d)
        cells[cell_ids[-2], 1] = -matrices[-1]

        # Continue initialization now that we have a geometry
        self._init_with_geometry(geometry, signed_volume=signed_volume)

    def __init__(self, geometry, signed_volume: Callable = predicates.volume):
        if isinstance(geometry, np.ndarray):
            self._init_with_points(geometry, signed_volume=signed_volume)
        elif isinstance(geometry, Geometry):
            self._init_with_geometry(geometry, signed_volume=signed_volume)

    @property
    def visible(self):
        r"""Stores which vertices are visible from which faces"""
        return self._visible

    def free_cell_ids(self, dimension: int):
        r"""The sets of numeric IDs of a given dimension that have not yet
        been assigned"""
        return self._free_cell_ids[dimension]

    def _get_new_cell_ids(self, dimension: int, num_new_cells: int):
        free_cell_ids = self.free_cell_ids(dimension)
        if num_new_cells > len(free_cell_ids):
            num_cells = len(self.geometry.topology.cells(dimension))
            exp = int(np.ceil(np.log2(1 + num_new_cells / num_cells)))
            self.geometry.topology.cells(dimension).resize(2**exp * num_cells)
            free_cell_ids.update(set(range(num_cells, 2**exp * num_cells)))

        return [free_cell_ids.pop() for i in range(num_new_cells)]

    @property
    def geometry(self):
        r"""The current topology for the hull"""
        return self._geometry

    def is_done(self):
        r"""Return `True` if there are no more vertices left to add"""
        return not self.visible.cell_to_vertex

    def adjoin_extreme_vertex(self, vertex_id, visible_cell_ids):
        r"""Adding a cone from the visible cells to the extreme vertex"""
        topology = self.geometry.topology
        dimension = topology.dimension
        cells_ids, Ds = topology.cells(dimension).closure(visible_cell_ids)
        Es = transformations.split(Ds)

        new_cells_ids = [np.append(cells_ids[0], vertex_id)]
        for k in range(1, dimension):
            num_new_cells = Es[k].shape[1] - Ds[k].shape[1]
            new_cell_ids = self._get_new_cell_ids(k, num_new_cells)
            new_cells_ids.append(np.append(cells_ids[k], new_cell_ids))
            cells = topology.cells(k)
            cells[new_cells_ids[k - 1], new_cells_ids[k]] = Es[k]

        new_cell_ids = self._get_new_cell_ids(dimension, Es[-1].shape[1])
        new_cells_ids.append(new_cell_ids)
        cells = topology.cells(dimension)
        cells[new_cells_ids[-2], new_cells_ids[-1]] = Es[-1]

        return new_cells_ids[-1]

    def remove_old_cells(self, cell_ids):
        topology = self.geometry.topology
        for k in range(topology.dimension, 0, -1):
            cells = topology.cells(k)
            face_ids, signs = cells[cell_ids]
            cells[face_ids, cell_ids] = np.zeros_like(signs)
            self.free_cell_ids(k).update(cell_ids)

            cofaces = topology.cocells(k - 1)
            cell_ids = [idx for idx in face_ids if len(cofaces[idx][0]) == 0]

    def step(self):
        r"""Find an extreme vertex and split all the cells that can see it"""
        vertex_id, visible_cell_ids = self.visible.get_next_vertex_and_cells()
        if (vertex_id is None) or (visible_cell_ids is None):
            return

        new_cell_ids = self.adjoin_extreme_vertex(vertex_id, visible_cell_ids)
        self.visible.update(new_cell_ids, visible_cell_ids)
        self.remove_old_cells(visible_cell_ids)

    def finalize(self):
        for k in range(1, self.geometry.topology.dimension + 1):
            self.geometry.topology.cells(k).remove_empty_cells()

        return self.geometry

    def run(self):
        while not self.is_done():
            self.step()

        return self.finalize()


def convex_hull(points):
    r"""Compute the convex hull of a point set"""
    machine = ConvexHullMachine(points)
    return machine.run()
