from math import comb as binomial
import numpy as np
import scipy.sparse
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


class ConvexHullMachine:
    def _init_with_geometry(self, geometry: Geometry):
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

        # Create the visibility map
        covertices = self._geometry.topology.cocells(0)
        visible = scipy.sparse.dok_matrix((len(covertices), len(cells)))
        for cell_id in range(len(cells)):
            face_ids, matrices = cells.closure(cell_id)
            orientation = simplicial.orientation(matrices)
            X = self.geometry.points[face_ids[0], :]
            for vertex_id in range(len(covertices)):
                edge_ids, signs = covertices[vertex_id]
                if len(edge_ids) == 0:
                    z = self.geometry.points[vertex_id, :]
                    volume = orientation * predicates.volume(z, *X)
                    if volume <= 0:
                        visible[vertex_id, cell_id] = volume

        self._visible = visible

    def _init_with_points(self, points: np.ndarray):
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
        self._init_with_geometry(geometry)

    def __init__(self, geometry, vertex_elimination_heuristic=True):
        if isinstance(geometry, np.ndarray):
            self._init_with_points(geometry)
        elif isinstance(geometry, Geometry):
            self._init_with_geometry(geometry)

        self._vertex_elimination_heuristic = vertex_elimination_heuristic

    @property
    def visible(self):
        r"""A matrix storing which vertices are visible from which faces"""
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
            self.visible.resize((self.visible.shape[0], 2**exp * num_cells))
            free_cell_ids.update(set(range(num_cells, 2**exp * num_cells)))

        return [free_cell_ids.pop() for i in range(num_new_cells)]

    @property
    def geometry(self):
        r"""The current topology for the hull"""
        return self._geometry

    def is_done(self):
        # TODO: Checking this at every step is inefficient, store #nz
        return self._visible.count_nonzero() == 0

    def get_next_vertex_and_cells(self):
        r"""Return the IDs of an extreme vertex and all the cells visible from
        it, or None if there are no extreme vertices left"""
        visible = self.visible
        dimension = self.geometry.topology.dimension
        num_cells = len(self.geometry.topology.cells(dimension))
        # TODO: This is an O(m) operation, store #nz in a member
        num_visible_vertices = np.array(
            [len(visible[:, idx].nonzero()[0]) for idx in range(num_cells)]
        )
        cell_id = np.argmax(num_visible_vertices)
        col = self.visible.getcol(cell_id)
        if col.nnz == 0:
            return None

        extreme_vertex_id = np.argmin(col)
        if self.visible[extreme_vertex_id, cell_id] >= 0:
            return

        visible_cell_ids = self.visible[extreme_vertex_id, :].nonzero()[1]
        return extreme_vertex_id, visible_cell_ids

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

    def update_visibility(self, new_cell_ids, old_cell_ids):
        topology = self.geometry.topology
        dimension = topology.dimension
        cells = topology.cells(dimension)
        cofaces = topology.cocells(dimension - 1)
        for new_cell_id in new_cell_ids:
            face_ids, Ds = cells.closure(new_cell_id)
            orientation = simplicial.orientation(Ds)
            X = self.geometry.points[face_ids[0]]

            # TODO: Replace this with a breadth-first search
            for vertex_id, z in enumerate(self.geometry.points):
                volume = orientation * predicates.volume(z, *X)
                if volume < 0:
                    self.visible[vertex_id, new_cell_id] = volume

        self.visible[:, old_cell_ids] = 0.0

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
        try:
            vertex_id, visible_cell_ids = self.get_next_vertex_and_cells()
        except TypeError:
            return

        new_cell_ids = self.adjoin_extreme_vertex(vertex_id, visible_cell_ids)
        self.update_visibility(new_cell_ids, visible_cell_ids)
        self.remove_old_cells(visible_cell_ids)

    def finalize(self):
        for k in range(1, self.geometry.topology.dimension + 1):
            self.geometry.topology.cells(k).remove_empty_cells()

        return self.geometry

    def run(self):
        while not self.is_done():
            self.step()

        return self.finalize()


def convex_hull(points, **kwargs):
    r"""Calculate the convex hull of a 2D point set"""
    machine = ConvexHullMachine(points, **kwargs)
    return machine.run()
