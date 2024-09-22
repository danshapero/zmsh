import numpy as np
import numpy.ma as ma
import predicates
from . import simplicial, polytopal


def extreme_points(points: np.ndarray) -> list[int]:
    r"""Given a set of input points, return a list of the indices of some
    points on the convex hull"""
    dimension = points.shape[1]
    centroid = np.mean(points, axis=0)
    indices = [np.argmax(np.sum((points - centroid) ** 2, axis=1))]
    x_0 = points[indices[0]]
    indices.append(np.argmax(np.sum((points - x_0) ** 2, axis=1)))

    Id = np.eye(dimension)
    while len(indices) <= dimension:
        # Project all of the points of the input set onto the plane formed by
        # current set of extreme points
        A = np.array([points[k] - x_0 for k in indices[1:]]).T
        Q, R = np.linalg.qr(A)
        delta = (Id - Q @ Q.T) @ (points - x_0).T

        # Compute the distance from each point to its projection and add the
        # point of greatest distance
        norms = np.linalg.norm(delta, axis=0)
        indices.append(np.argmax(norms))

    return indices


class VisibilityGraph:
    def __init__(self, topology: simplicial.Topology, points: np.ndarray, volume_fn):
        self._cells_to_points = {}
        self._points_to_cells = {}

        for cell_id, cell in enumerate(topology):
            if not cell.mask.any():
                xs = points[cell]
                cell_to_points = {}
                for point_id, z in enumerate(points):
                    if not (point_id in cell):
                        volume = volume_fn(np.column_stack((z, *xs)))
                        if volume <= 0.0:
                            cell_to_points[point_id] = volume

                if cell_to_points:
                    self._cells_to_points[cell_id] = cell_to_points

        for cell_id, point_volume_pairs in self._cells_to_points.items():
            for point_id, volume in point_volume_pairs.items():
                if not point_id in self._points_to_cells:
                    self._points_to_cells[point_id] = {}
                self._points_to_cells[point_id][cell_id] = volume

    @property
    def cells_to_points(self):
        return self._cells_to_points

    @property
    def points_to_cells(self):
        return self._points_to_cells

    def get_next_point_and_cells(self):
        r"""Return an extreme point and all the cells that are visible to it"""
        if not self.cells_to_points:
            return
        cell_id = next(iter(self.cells_to_points.keys()))
        entry = self.cells_to_points[cell_id]
        point_id = min(entry, key=lambda point_id: entry[point_id])
        cell_ids = list(self.points_to_cells[point_id].keys())
        return np.uintp(point_id), cell_ids

    def add(self, cell_id: int, point_id: int, volume: float):
        if not cell_id in self._cells_to_points:
            self._cells_to_points[cell_id] = {}
        self._cells_to_points[cell_id][point_id] = volume
        if not point_id in self._points_to_cells:
            self._points_to_cells[point_id] = {}
        self._points_to_cells[point_id][cell_id] = volume


class ConvexHull:
    def _initial_topology(self, points: np.ndarray, volume_fn):
        starting_point_ids = extreme_points(points)
        start_points = points[starting_point_ids]
        volume = volume_fn(start_points.T)
        if volume == 0.0:
            # TODO: Handle degenerate case
            raise NotImplementedError("Can't handle degenerate input sets")
        elif volume < 0.0:
            starting_point_ids[:2] = starting_point_ids[:2][::-1]

        dimension = points.shape[1]
        num_cells = dimension + 2
        topology = ma.masked_all((num_cells, dimension), dtype=np.uintp)
        for k in range(0, dimension + 1, 2):
            cell = np.delete(starting_point_ids, k)
            topology[k] = cell

        for k in range(1, dimension + 1, 2):
            cell = np.delete(starting_point_ids, k)
            cell[:2] = cell[:2][::-1]
            topology[k] = cell

        return topology

    def __init__(
        self,
        points: np.ndarray,
        *,
        topology=None,
        volume_fn=predicates.volume,
    ):
        if topology is None:
            topology = self._initial_topology(points, volume_fn)
        elif not isinstance(topology, ma.MaskedArray):
            topology = ma.array(topology, mask=np.zeros_like(topology, dtype=bool))

        num_cells, dimension = topology.shape
        self._volume = volume_fn
        self._free_cell_ids = {
            index for index, simplex in enumerate(topology) if simplex.mask.all()
        }
        self._visibility = VisibilityGraph(topology, points, self._volume)
        self._topology = topology
        self._points = points

    @property
    def topology(self):
        return self._topology

    @property
    def visibility(self):
        return self._visibility

    def _compute_new_simplices(self, point_id: int, cell_ids: np.ndarray) -> np.ndarray:
        simplices = self.topology[cell_ids]
        point_ids = np.append(np.unique(simplices), point_id)
        id_map = np.vectorize({idx: val for val, idx in enumerate(point_ids)}.get)
        reordered_simplices = id_map(simplices)

        matrices = polytopal.from_simplicial(reordered_simplices)
        new_matrices = polytopal.vertex_split(matrices)
        return point_ids[polytopal.to_simplicial(new_matrices)]

    def step(self):
        if self.is_done():
            return

        # Make the new cells (don't add them yet)
        extreme_point_id, cell_ids = self.visibility.get_next_point_and_cells()
        new_cells = self._compute_new_simplices(extreme_point_id, cell_ids)

        # Remove the old simplices
        self.topology.mask[cell_ids] = True
        self._free_cell_ids.update(cell_ids)
        for cell_id in cell_ids:
            for point_id in self.visibility.cells_to_points[cell_id].keys():
                del self.visibility.points_to_cells[point_id][cell_id]
            del self.visibility.cells_to_points[cell_id]

        # Resize if need be
        num_new_cells = len(new_cells)
        if num_new_cells > len(self._free_cell_ids):
            num_cells, dimension = self.topology.shape
            topology = ma.masked_all((2 * num_cells, dimension), dtype=np.uintp)
            topology[:num_cells] = self.topology
            self._topology = topology
            new_free_cell_ids = list(range(num_cells, 2 * num_cells))
            self._free_cell_ids.update(new_free_cell_ids)

        # Add the new simplices
        new_cell_ids = [self._free_cell_ids.pop() for idx in range(num_new_cells)]
        self.topology[new_cell_ids] = new_cells

        # Update the visibility map
        # TODO: Narrow this down to not looking over all points.
        hull_point_ids = np.unique(self.topology.compressed())
        for cell_id, cell in zip(new_cell_ids, new_cells):
            xs = self._points[cell]
            for point_id, z in enumerate(self._points):
                if not (point_id in hull_point_ids):
                    volume = self._volume(np.column_stack((z, *xs)))
                    if volume <= 0:
                        self.visibility.add(cell_id, point_id, volume)

    def is_done(self) -> bool:
        return len(self.visibility.cells_to_points) == 0

    def finalize(self) -> simplicial.Topology:
        cell_ids = np.array(
            [
                index
                for index, cell in enumerate(self.topology)
                if cell.compressed().size != 0
            ]
        )
        return self.topology[cell_ids].data

    def run(self):
        while not self.is_done():
            self.step()
        return self.finalize()
