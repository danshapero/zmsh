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


def _nonzero_indices(A, axis=0):
    return np.flatnonzero(np.count_nonzero(A, axis=axis))


def identify_component(D, start_id, cell_ids, stopping_face_ids=np.array([])):
    r"""Return all cells among a set that are reachable from the start cell"""
    component = set()
    queue = {start_id}
    while queue:
        cell_id = queue.pop()
        face_ids = np.flatnonzero(D[:, cell_id])
        if len(face_ids) > 0:
            component.add(cell_id)
            face_ids = np.setdiff1d(face_ids, stopping_face_ids)
            neighbor_ids = _nonzero_indices(D[face_ids, :], axis=0)
            queue.update(set(neighbor_ids).intersection(cell_ids) - component)

    return np.array(sorted(list(component)))


def identify_components(D, cell_ids, stopping_face_ids=np.array([])):
    r"""Return all subsets of a given set of cells by mutual reachability"""
    components = []
    queue = set(cell_ids)
    while queue:
        start_id = queue.pop()
        component = identify_component(D, start_id, cell_ids, stopping_face_ids)
        if len(component) > 0:
            queue.difference_update(component)
            components.append(component)

    return components


def identify_separators(Ds):
    r"""Return tuples of all separators and their complements"""
    null_cell_ids = np.flatnonzero(Ds[-1] == 0)
    null_components = identify_components(Ds[-2], null_cell_ids)
    components = []
    for null_cell_ids in null_components:
        cell_ids = np.setdiff1d(np.arange(Ds[-2].shape[1]), null_cell_ids)
        face_ids = _nonzero_indices(Ds[-2][:, null_cell_ids], axis=1)
        complements = identify_components(Ds[-2], cell_ids, stopping_face_ids=face_ids)
        if len(complements) > 1:
            components.append((null_cell_ids, complements))

    return components


def bisect(Ds, components=None):
    r"""Return a chain complex obtained by dividing a cell in two along a set
    of faces to be determined"""
    if Ds[-1].shape[1] != 1:
        raise ValueError("Can only bisect a single top cell")

    if not components:
        components = identify_separators(Ds)
        num_seps = len(components)
        if num_seps != 1:
            raise ValueError("%s separators found, expected 1" % num_seps)

    separator_ids, (face_ids1, face_ids2) = components[0]
    D_1, D_2 = Ds[-2], Ds[-1]
    v_e1 = D_1[:, face_ids1]
    v_e2 = D_1[:, face_ids2]
    v_e3 = D_1[:, separator_ids]

    e1_p = D_2[face_ids1]
    e2_p = D_2[face_ids2]

    A = v_e3.T @ v_e3
    f = -v_e3.T @ v_e1 @ e1_p
    # TODO: This is really dodgy, the solution is computed in floating-point
    # and rounded to integer. We should try to concoct a case where it breaks
    # and then do the linear algebra using the Hermite normal form instead.
    e3_p = np.rint(np.linalg.solve(A, f)).astype(np.int8)

    z_12 = np.zeros((e1_p.shape[0], e2_p.shape[1]), dtype=np.int8)
    z_21 = np.zeros((e2_p.shape[0], e1_p.shape[1]), dtype=np.int8)
    E = np.block(
        [
            [e1_p, z_12],
            [z_21, e2_p],
            [+e3_p, -e3_p],
        ],
    )

    face_ids = np.concatenate((face_ids1, face_ids2, separator_ids))
    D = np.zeros_like(E)
    D[face_ids, :] = E
    return D
