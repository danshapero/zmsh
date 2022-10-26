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


class ConvexHullMachine:
    def __init__(self, points, vertex_elimination_heuristic=True):
        # TODO: check for collinearity
        n = len(points)
        indices = extreme_points(points)

        dimension = points.shape[1]
        # TODO: This is a grotesque upper bound. To fix, give it only enough
        # space for `2 * d` faces, and double the number of cells of a given
        # dimension as needed.
        num_cells = [n] + [dimension * n ** (dimension // 2)] * (dimension - 1)
        topology = Topology(dimension=dimension - 1, num_cells=num_cells)
        self._geometry = Geometry(topology, points.copy())

        Ds = simplicial.simplex_to_chain_complex(dimension - 1)[1:]
        cell_ids = [indices] + [tuple(range(D.shape[1])) for D in Ds]
        for k, D in enumerate(Ds, start=1):
            cells = self._geometry.topology.cells(k)
            cells[cell_ids[k - 1], cell_ids[k]] = D

        # Take the initial simplex and add its mirror image
        cells = self._geometry.topology.cells(dimension - 1)
        cells[cell_ids[-2], 1] = -Ds[-1]

        # Store which numeric IDs of each dimension can still be used
        # TODO: Does this need to be a set or can it be a stack (LIFO)?
        # TODO: Grossly inefficient, store a `slice(m, n ** (dimension // 2))`
        # for the end of the thing
        self._free_cell_ids = [set()] + [
            set(range(num_cells[k])) - set(cell_ids[k]) for k in range(1, dimension)
        ]
        self._free_cell_ids[-1].remove(1)

        self._candidates = set(range(n)) - set(indices)
        self._cell_queue = [0, 1]
        self._vertex_elimination_heuristic = vertex_elimination_heuristic

    @property
    def candidates(self):
        r"""The set of indices of all points that might be in the hull"""
        return self._candidates

    @property
    def cell_queue(self):
        r"""The list of edges that still need inspecting"""
        return self._cell_queue

    def free_cell_ids(self, dimension):
        r"""The sets of numeric IDs of a given dimension that have not yet
        been assigned"""
        return self._free_cell_ids[dimension]

    @property
    def geometry(self):
        r"""The current topology for the hull"""
        return self._geometry

    def best_candidate(self, cell_index):
        r"""Return the index of the candidate point that forms the simplex
        of largest volume with the given cell"""
        topology = self.geometry.topology
        cell_ids, Ds = topology.cells(topology.dimension).closure(cell_index)
        orientation = simplicial.orientation(Ds)
        X = self.geometry.points[cell_ids[0], :]

        best_vertex_id = None
        best_volume = np.inf
        for vertex_id in self._candidates:
            z = self.geometry.points[vertex_id, :]
            volume = orientation * predicates.volume(*X, z)
            if volume < best_volume:
                best_vertex_id = vertex_id
                best_volume = volume

        return best_vertex_id, best_volume

    def find_interior_vertices(self, X, sign):
        r"""Return the indices of all candidate points that are strictly
        contained in a simplex"""
        interior_vertex_ids = []
        for index in self._candidates:
            z = self.geometry.points[index]
            # TODO: Revisit whether this is the right condition in the case of
            # collinear points, either in the interior or on the boundary of
            # the convex hull.
            inside = all(
                sign * (-1) ** k * predicates.volume(*np.delete(X, k, axis=0), z) > 0
                for k in range(X.shape[0])
            )
            if inside:
                interior_vertex_ids.append(index)

        return interior_vertex_ids

    def find_visible_cells(self, z, starting_cell_id=None):
        r"""Return a list of all cell IDs such that the point `z` is visible
        from that cell"""
        topology = self.geometry.topology
        cells = topology.cells(topology.dimension)

        if starting_cell_id is None:
            for cell_id in range(len(cells)):
                try:
                    face_ids, matrices = cells.closure(cell_id)
                    orientation = simplicial.orientation(matrices)
                    X = self.geometry.points[face_ids[0]]
                    # TODO: Again, check `<` vs `<=`
                    if orientation * predicates.volume(*X, z) <= 0:
                        return self.find_visible_cells(z, starting_cell_id=cell_id)
                # TODO: Better handling of empty cells than this nonsense
                except IndexError:
                    pass

        visible_cell_ids = set()
        queue = {starting_cell_id}
        cofaces = topology.cocells(topology.dimension - 1)
        while len(queue) > 0:
            cell_id = queue.pop()
            try:
                face_ids, matrices = cells.closure(cell_id)
                orientation = simplicial.orientation(matrices)
                X = self.geometry.points[face_ids[0]]
                if orientation * predicates.volume(*X, z) <= 0:
                    visible_cell_ids.add(cell_id)
                    neighbor_cell_ids = cofaces[face_ids[-2]][0]
                    queue.update(set(neighbor_cell_ids) - visible_cell_ids)
            except IndexError:
                pass

        return list(visible_cell_ids)

    def is_done(self):
        return (not self._cell_queue) or (not self._candidates)

    def step(self):
        r"""Process the next edge -- either do nothing, or split it if there's
        an extreme point across from it"""
        if self.is_done():
            return

        # Pop the next edge and find any extreme points across from it; if
        # there aren't any, we're done.
        cell_id = self._cell_queue.pop(0)
        extreme_vertex_id, volume = self.best_candidate(cell_id)
        # TODO: Revisit whether this should be `>` or `>=`
        if volume >= 0:
            return

        # Find all cells that are visible from the extreme vertex
        z = self.geometry.points[extreme_vertex_id]
        visible_cell_ids = self.find_visible_cells(z, starting_cell_id=cell_id)
        topology = self.geometry.topology
        dimension = topology.dimension

        # Filter out all candidate points inside the triangle formed by the old
        # edge and the two new edges
        self._candidates.remove(extreme_vertex_id)
        if self._vertex_elimination_heuristic:
            cells = topology.cells(dimension)
            for cell_id in visible_cell_ids:
                face_ids, matrices = cells.closure(cell_id)
                sign = simplicial.orientation(matrices)
                X = np.vstack((z, self.geometry.points[face_ids[0]]))
                interior_vertex_ids = self.find_interior_vertices(X, -sign)
                self._candidates -= set(interior_vertex_ids)

        # Compute the transformation that splits the visible cells
        cells_ids, Ds = topology.cells(dimension).closure(visible_cell_ids)
        Es = transformations.split(Ds)

        # Assign IDs to any newly-created cells and set the new cells in the
        # topology
        new_cells_ids = [np.append(cells_ids[0], extreme_vertex_id)]
        for k in range(1, dimension + 1):
            num_new_cells = Es[k].shape[1] - Ds[k].shape[1]
            free_cell_ids = self.free_cell_ids(k)
            new_cell_ids = [free_cell_ids.pop() for i in range(num_new_cells)]
            new_cells_ids.append(np.append(cells_ids[k], new_cell_ids))

        for k in range(1, dimension + 1):
            cells = topology.cells(k)
            cells[new_cells_ids[k - 1], new_cells_ids[k]] = Es[k]

        # Return the IDs for any deleted cells to the free sets
        for k in range(1, dimension + 1):
            for index in range(Ds[k].shape[1]):
                if np.all(Es[k][:, index] == 0):
                    self.free_cell_ids(k).add(cells_ids[k][index])

        self._cell_queue.extend(new_cells_ids[-1])

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
