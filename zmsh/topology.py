import numpy as np
from scipy.sparse import dok_matrix as sparse_matrix
import scipy.sparse.linalg
import numpy.linalg

def matrix_norm(*args, **kwargs):
    try:
        return scipy.sparse.linalg.norm(*args, **kwargs)
    except TypeError:
        return numpy.linalg.norm(*args, **kwargs)


class Topology(object):
    def __init__(self, dimension):
        # TODO: replace these with something representing a row/column vector
        # of all 1s without actually storing them
        bottom_boundary = np.zeros((0, 0), dtype=np.int8)
        top_boundary = np.zeros((0, 0), dtype=np.int8)

        cell_boundaries = [sparse_matrix((0, 0), dtype=np.int8)
                           for d in range(dimension)]
        self._boundary = [bottom_boundary] + cell_boundaries + [top_boundary]

    @property
    def dimension(self):
        r"""The max dimension of all cells of the mesh"""
        return len(self._boundary) - 2

    def num_cells(self, dimension):
        r"""The number of cells of a given dimension"""
        return self._boundary[dimension].shape[1]

    def set_num_cells(self, dimension, num_cells):
        r"""Set the number of a cells of a given dimension"""
        if dimension == 0:
            self._boundary[0] = np.ones((1, num_cells), dtype=np.int8)
        else:
            matrix = self.boundary(dimension)
            shape = matrix.shape
            matrix.resize((shape[0], num_cells))

        if dimension == self.dimension:
            self._boundary[dimension + 1] = np.ones((num_cells, 1), dtype=np.int8)
        else:
            matrix = self.boundary(dimension + 1)
            shape = matrix.shape
            matrix.resize((num_cells, shape[1]))

    def boundary(self, dimension):
        r"""Return the boundary operator on chains of a given dimension"""
        return self._boundary[dimension]

    def cell(self, dimension, index):
        r"""Return the indices of the faces and the incidences to them"""
        if dimension == 0:
            return np.array([0]), np.array([+1])

        matrix = self._boundary[dimension]
        faces = matrix[:, index].nonzero()[0]
        incidence = np.array(matrix[faces, index].todense()).flatten()
        return faces, incidence

    def set_cell(self, dimension, index, faces, incidence):
        r"""Set the faces and incidences of a given cell of the mesh"""
        if dimension == 0:
            return

        D = self.boundary(dimension)
        D[:, index] = 0
        D[faces, index] = incidence

    def cells(self, dimension):
        r"""Yield all the cells of a given dimension"""
        num_cells = self.num_cells(dimension)
        return (self.cell(dimension, index) for index in range(num_cells))

    def compute_nonzero_boundary_products(self):
        r"""For each dimension `k`, compute the product of the boundary
        operators of dimension `k` and `k + 1`, and return all products
        that are non-zero"""
        results = []
        for k in range(self.dimension):
            P = self.boundary(k) @ self.boundary(k + 1)
            if matrix_norm(P, ord=1) != 0:
                results.append((k, P))

        return results
