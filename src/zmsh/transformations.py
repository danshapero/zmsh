import numpy as np


def split(D):
    r"""Return a chain complex obtained by splitting the sum of the top-
    dimensional cells on a newly-added vertex"""
    # Create the lower-dimensional boundary matrices
    num_vertices = D[0].shape[1]
    E = [np.ones((1, num_vertices + 1), dtype=np.int8)]
    for k in range(1, len(D) - 1):
        num_cells = D[k].shape[1]
        num_sub_faces, num_faces = D[k - 1].shape
        I = np.eye(num_faces, dtype=np.int8)
        Z = np.zeros((num_sub_faces, num_cells), dtype=np.int8)
        E_k = np.block([[D[k], I], [Z, -D[k - 1]]])
        E.append(E_k)

    # Create the top-dimensional boundary matrix, removing any empty cells
    C = D[-1] if len(D[-1].shape) == 1 else np.sum(D[-1], axis=1)
    E_n = np.vstack((np.diag(C), -D[-2] @ np.diag(C)))
    empty_cells = np.argwhere(np.all(E_n == 0, axis=0))
    E.append(np.delete(E_n, empty_cells, axis=1))

    # Nullify any cell that has zero coboundary
    for E_2, E_1 in zip(E[-1:0:-1], E[-2::-1]):
        empty_cells = np.argwhere(np.all(E_2 == 0, axis=1))
        E_1[:, empty_cells] = 0

    return E
