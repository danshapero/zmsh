import numpy as np
from scipy.sparse import dok_matrix


# Hack to make getting faces of 0-dimensional cells work nice; the boundary
# matrix needs to have a `todense` method.
class SparseView(np.ndarray):
    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def todense(self):
        return self


class Cells:
    def __init__(self, topology, dimension):
        r"""A view of the cells of a particular dimension of a topology"""
        # Do we want a weakref here?
        self._topology = topology
        self._dimension = dimension

    @property
    def boundary(self):
        r"""A matrix representing the boundary operator"""
        return self._topology._boundaries[self._dimension]

    def __len__(self):
        return self.boundary.shape[1]

    def __getitem__(self, index):
        r"""Get the faces and corresponding signs of a particular cell"""
        faces = self.boundary[:, index].nonzero()[0]
        signs = np.array(self.boundary[faces, index].todense()).flatten()
        return faces, signs

    def __setitem__(self, index, value):
        r"""Set the faces and corresponding signs of a particular cell"""
        faces, signs = value
        self.boundary[:, index] = 0
        self.boundary[faces, index] = signs

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def resize(self, size):
        if self._dimension == 0:
            dtype = self.boundary.dtype
            self._topology._boundaries[0] = np.ones((1, size), dtype=dtype)
        else:
            shape = self.boundary.shape
            self.boundary.resize((shape[0], size))

        if self._dimension < self._topology.dimension:
            coboundary = self._topology._boundaries[self._dimension + 1]
            coboundary.resize((size, coboundary.shape[1]))

class Topology:
    def __init__(self, dimension, num_cells=None, **kwargs):
        if num_cells is None:
            num_cells = [0] * (dimension + 1)

        dtype = kwargs.get("dtype", np.int8)
        mat_type = kwargs.get("mat_type", dok_matrix)
        self._boundaries = [
            SparseView(np.ones((1, num_cells[0]), dtype=dtype))
        ] + [
            mat_type((num_cells[d - 1], num_cells[d]), dtype=dtype)
            for d in range(1, dimension + 1)
        ]

    @property
    def dimension(self):
        r"""The max dimension of all cells of the topology"""
        return len(self._boundaries) - 1

    def cells(self, dimension):
        r"""Get the cells of the given dimension"""
        return Cells(self, dimension)
