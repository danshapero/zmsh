import numbers
import numpy as np
import z3


def symbolic_matrix(shape, name):
    # TODO: Make it work for arbitrary shape
    fn = np.vectorize(lambda i, j: z3.Int(f"{name}_{i}_{j}"))
    return np.fromfunction(fn, shape, dtype=int)


Abs = np.frompyfunc(lambda x: z3.If(x >= 0, x, -x), 1, 1)


class Transformation:
    def __init__(self, shape, name=None):
        self._matrix = symbolic_matrix(shape, name or "matrix")
        self._solver = z3.Solver()

    @property
    def matrix(self):
        r"""The local boundary matrix in the transformed topology"""
        return self._matrix

    @property
    def solver(self):
        r"""The `z3.Solver` instance to obtain the transformed topology"""
        return self._solver

    def constrain_range(self, minval=-1, maxval=+1):
        r"""Constrain all matrix entries so that `minval <= A[i, j] <= maxval`"""
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                self.solver.add(self.matrix[i, j] >= minval)
                self.solver.add(self.matrix[i, j] <= maxval)

    def constrain_num_faces(self, num_entries, columns=None):
        r"""Constrain the number of non-zero entries in each column of the matrix"""
        cols = columns or range(self.matrix.shape[1])
        if isinstance(num_entries, numbers.Integral):
            num_entries = len(cols) * (num_entries,)

        for col in cols:
            constraint = np.sum(Abs(self.matrix[:, col])) == num_entries[col]
            self.solver.add(constraint)

    def constrain_num_cofaces(self, num_entries, rows=None):
        r"""Constrain the number of non-zero entries in each row of the matrix"""
        rows = rows or range(self.matrix.shape[0])
        if isinstance(num_entries, numbers.Integral):
            num_entries = len(rows) * (num_entries,)

        for row in rows:
            constraint = np.sum(Abs(self.matrix[row, :])) == num_entries[row]
            self.solver.add(constraint)

    def constrain_boundary(self, face_matrix):
        r"""Constrain the product from the left with another matrix"""
        product = face_matrix @ self.matrix
        for i in range(product.shape[0]):
            for j in range(product.shape[1]):
                self.solver.add(product[i, j] == 0)

    def constrain_coboundary(self, coface_matrix):
        r"""Constrain the product from the right with another matrix"""
        product = self.matrix @ coface_matrix
        for i in range(product.shape[0]):
            for j in range(product.shape[1]):
                self.solver.add(product[i, j] == 0)

    def compute(self, dtype=np.int8):
        r"""Solve for and return the transformed boundary matrix"""
        if self.solver.check() == z3.unsat:
            return None

        model = self.solver.model()
        fn = np.vectorize(lambda i, j: model[self.matrix[i, j]].as_long())
        return np.fromfunction(fn, self.matrix.shape, dtype=dtype)
