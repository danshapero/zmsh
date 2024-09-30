import numpy as np
from numpy import ma, count_nonzero, flatnonzero as nonzero
import predicates
from . import simplicial, polytopal, convex_hull


class Delaunay:
    def __init__(self, points: np.ndarray):
        square_norms = np.sum(points**2, axis=1)
        lifted_points = np.column_stack((points, square_norms))
        starting_point_ids = convex_hull.extreme_points(lifted_points)
        start_points = points[starting_point_ids]
        volume = predicates.insphere(start_points.T)

        if volume > 0.0:
            starting_point_ids[:2] = starting_point_ids[:2][::-1]

        dimension = points.shape[1]
        num_cells = dimension + 3
        topology = ma.masked_all((num_cells, dimension + 1), dtype=np.uintp)

        for k in range(0, dimension + 2, 2):
            cell = np.delete(starting_point_ids, k)
            topology[k] = cell

        for k in range(1, dimension + 2, 2):
            cell = np.delete(starting_point_ids, k)
            cell[:2] = cell[:2][::-1]
            topology[k] = cell

        self._points = points
        volume_fn = lambda *args: -predicates.insphere(*args)
        self._hull = convex_hull.ConvexHull(
            points, topology=topology, volume_fn=volume_fn
        )

    @property
    def topology(self):
        return self._hull.topology

    def step(self):
        return self._hull.step()

    def is_done(self):
        return self._hull.is_done()

    def finalize(self):
        simplices = self._hull.finalize()
        # TODO: Check the signs...
        simplex_ids = [
            index
            for index, cell in enumerate(simplices)
            if predicates.volume(self._points[cell].T) > 0
        ]
        return simplices[simplex_ids]

    def run(self):
        while not self.is_done():
            self.step()
        return self.finalize()


def line_segments_intersect(xs, ys):
    cross1 = np.prod([predicates.volume(np.column_stack((y, *xs))) for y in ys])
    cross2 = np.prod([predicates.volume(np.column_stack((x, *ys))) for x in xs])
    return max(cross1, cross2)


def find_crossings(simplices, points, xs) -> np.ndarray:
    cell_ids = set()
    for cell_id, cell in enumerate(simplices):
        for edge in zip(cell, np.roll(cell, 1)):
            ys = points[edge, :]
            if line_segments_intersect(xs, ys) < 0.0:
                cell_ids.add(cell_id)

    return np.array(list(cell_ids))


def find_splitting_vertex(
    topology: polytopal.Topology, edge_id: int, points: np.ndarray
) -> int:
    sign = topology[2][edge_id, 0]
    all_vertex_ids = nonzero(count_nonzero(topology[1], 1))
    vertex_ids = [nonzero(topology[1][:, edge_id] == s)[0] for s in (-sign, +sign)]
    xs = points[vertex_ids]
    candidate_vertex_ids = np.setdiff1d(all_vertex_ids, vertex_ids)
    zs = points[candidate_vertex_ids]
    queue = candidate_vertex_ids.copy()
    while queue.size > 0:
        vertex_id, queue = queue[0], queue[1:]
        y = points[vertex_id]
        if (
            np.array([predicates.insphere(np.column_stack((*xs, y, z))) for z in zs])
            >= 0.0
        ).all():
            return vertex_id


class Retriangulation:
    @classmethod
    def from_simplices(cls, simplices, points, new_edge):
        new_edge = np.array(new_edge)
        d_0, d_1, d_2 = polytopal.from_simplicial(simplices)

        # Find the indices of all the edges that cross the constrained edge
        xs = points[new_edge, :]
        crossing_edge_ids = [
            index
            for index, col in enumerate(d_1.T)
            if line_segments_intersect(xs, points[nonzero(col), :]) < 0.0
        ]

        d_1 = np.delete(d_1, crossing_edge_ids, axis=1)
        d_2 = np.delete(np.sum(d_2, axis=1), crossing_edge_ids, axis=0)[..., None]

        # Add the constrained edge
        column = np.zeros(d_1.shape[0], dtype=np.int8)
        column[new_edge] = (-1, +1)
        d_1, d_2 = polytopal.add((d_1, d_2), column)

        # Split the merged polygon along the constrained edge
        edge_id = d_2.shape[0] - 1
        components = polytopal.mark_components([d_1, d_2], [edge_id])
        e_2 = polytopal.face_split([d_1, d_2], components)

        queue = [[edge_id] + list(nonzero(components == index)) for index in [1, 2]]
        return cls([d_0, d_1, e_2], points, queue)

    def __init__(self, topology, points, queue):
        self._topology = topology
        self._points = points
        self._queue = queue

    @property
    def topology(self):
        return self._topology

    def step(self):
        if self.is_done():
            return

        edge_ids = self._queue.pop()
        if len(edge_ids) <= 3:
            return

        d_0, d_1, d_2 = self.topology

        # Find which polygon the edges are contained in
        # TODO: Is this unnecessary? Can we pack it into the queue?
        poly_id = [
            index
            for index, poly in enumerate(d_2.T)
            if np.isin(nonzero(poly), edge_ids).all()
        ][0]

        # Get the local subcomplex
        cells_ids = [..., ..., edge_ids, [poly_id]]
        f_0, f_1, f_2 = polytopal.subcomplex(self.topology, cells_ids)

        # Find the splitting vertex, then check if we need to add 1 edge or 2
        vertex_id = find_splitting_vertex([f_0, f_1, f_2], 0, self._points)
        edge_vertex_ids = [
            index
            for index in nonzero(f_1[:, 0])
            if (count_nonzero(d_1[[vertex_id, index], :], axis=0) < 2).all()
        ]

        # Add the new edges to the local subcomplex
        edges = np.zeros((f_1.shape[0], len(edge_vertex_ids)), dtype=np.int8)
        for index, edge_vertex_id in enumerate(edge_vertex_ids):
            edges[[vertex_id, edge_vertex_id], index] = (-1, +1)

        f_1, f_2 = polytopal.add((f_1, f_2), edges)

        # Compute the new polygons
        separator_ids = [f_1.shape[1] - idx - 1 for idx in range(edges.shape[1])]
        components = polytopal.mark_components([f_1, f_2], separator_ids)
        g_2 = polytopal.face_split([f_1, f_2], components)

        # Add the new edges into the global topology
        d_1 = np.column_stack((d_1, edges))

        # Remove the old polygon from the global topology
        d_2 = np.delete(d_2, poly_id, axis=1)

        # Add the new polygon into the global topology
        edge_ids = np.append(edge_ids, np.arange(edges.shape[1]) + d_2.shape[0])
        d_2 = np.vstack((d_2, np.zeros((edges.shape[1], d_2.shape[1]), dtype=np.int8)))
        polys = np.zeros((d_2.shape[0], g_2.shape[1]), dtype=np.int8)
        polys[edge_ids, :] = g_2
        d_2 = np.column_stack((d_2, polys))

        self._topology[1] = d_1
        self._topology[2] = d_2

        # Add the new edge sets to the queue
        local_edge_ids = [
            np.union1d(nonzero(components == index + 1), nonzero(g_2[:, index]))
            for index in range(components.max())
        ]
        edge_id_sets = [
            list(edge_ids[local_ids][::-1])
            for local_ids in local_edge_ids
            if len(local_ids) > 3
        ]
        self._queue.extend(edge_id_sets)

    def is_done(self):
        return len(self._queue) == 0

    def finalize(self):
        return self.topology

    def run(self):
        while not self.is_done():
            self.step()
        return self.finalize()


class ConstrainedDelaunay:
    def __init__(self, points: np.ndarray, edges: np.ndarray):
        self._points = points
        self._topology = Delaunay(points).run()
        self._edges = edges.copy()

    @property
    def topology(self):
        return self._topology

    def step(self):
        edge, self._edges = self._edges[0], self._edges[1:]
        cell_ids = find_crossings(self._topology, self._points, self._points[edge, :])
        if len(cell_ids) == 0:
            return

        lsimplices = self._topology[cell_ids]
        vertex_ids = np.unique(lsimplices)
        lpoints = self._points[vertex_ids]
        id_map = np.vectorize({idx: val for val, idx in enumerate(vertex_ids)}.get)
        retria = Retriangulation.from_simplices(lsimplices, lpoints, id_map(edge))
        new_simplices = vertex_ids[polytopal.to_simplicial(retria.run())]
        self._topology[cell_ids] = new_simplices

    def is_done(self):
        return self._edges.size == 0

    def finalize(self):
        return self._topology

    def run(self):
        while not self.is_done():
            self.step()
        return self.finalize()
