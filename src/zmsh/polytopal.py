import functools
import itertools
from typing import List
import numpy as np
from numpy import flatnonzero as nonzero, count_nonzero
import scipy
from . import simplicial


Topology = List[np.ndarray]


def ones(shape):
    return np.ones(shape, np.int8)


def eye(shape):
    return np.eye(shape, dtype=np.int8)


def zeros(shape):
    return np.zeros(shape, dtype=np.int8)


def closure(D: Topology, cell_ids: np.ndarray) -> List[np.ndarray]:
    r"""Given the integer IDs of a set of top cells, return a list of arrays of
    these cells' faces, their faces' faces, etc."""
    dimension = len(D) - 1
    face_ids = [np.array(cell_ids)]
    for k in range(dimension, -1, -1):
        face_ids.append(nonzero(count_nonzero(D[k][:, face_ids[-1]], axis=1)))

    return face_ids[::-1]


def subcomplex(D: Topology, cells_ids: np.ndarray) -> Topology:
    r"""Given a list of arrays of the integer IDs of a closed set of cells,
    return the boundary matrices of the subcomplex"""
    return [
        d[face_ids, :][:, cell_ids]
        for d, face_ids, cell_ids in zip(D, cells_ids[:-1], cells_ids[1:])
    ]


def join_vertex(D: Topology) -> Topology:
    r"""Given a polytopal complex, return the boundary operators of the join
    with a single point, i.e. the topological cone"""
    n = len(D) - 1
    num_vertices = D[0].shape[1]
    E = [ones((1, num_vertices + 1))]
    for k in range(1, n + 1):
        num_cells = D[k].shape[1]
        num_sub_faces, num_faces = D[k - 1].shape
        I = eye(num_faces)
        Z = zeros((num_sub_faces, num_cells))
        E_k = np.block([[D[k], I], [Z, -D[k - 1]]])
        E.append(E_k)

    num_cells = D[n].shape[1]
    I = eye(num_cells)
    E_n = (-1) ** n * np.vstack((-I, D[n]))
    E.append(E_n)

    return E


def vertex_split(D: Topology) -> Topology:
    r"""Given a polytopal complex, return the split on a single vertex, which
    is equivalent to the top of the topological cone"""
    E = join_vertex(D)[:-2]
    C = np.sum(D[-1], axis=1)
    E_n = np.block([[np.diag(C)], [-D[-2] @ np.diag(C)]])
    empty_cells = np.argwhere(np.all(E_n == 0, axis=0))
    E.append(np.delete(E_n, empty_cells, axis=1))
    return E


def join_vertices(D: Topology) -> Topology:
    r"""Given a polytopal complex, return the boundary operators of the join
    with two points, i.e. the topological suspension"""
    n = len(D) - 1
    num_vertices = D[0].shape[1]
    E = [ones((1, num_vertices + 2))]
    for k in range(1, n + 1):
        num_cells = D[k].shape[1]
        num_sub_faces, num_faces = D[k - 1].shape
        I = eye(num_faces)
        Z1 = zeros((num_sub_faces, num_cells))
        Z2 = zeros((num_sub_faces, num_faces))
        E_k = np.block([[D[k], I, I], [Z1, -D[k - 1], Z2], [Z1, Z2, -D[k - 1]]])
        E.append(E_k)

    num_faces, num_cells = D[n].shape
    I = eye(num_cells)
    Z = zeros((num_faces, num_cells))
    E_n = (-1) ** n * np.block([[-I, +I], [+D[n], Z], [Z, -D[n]]])
    E.append(E_n)

    return E


def _mark_component(
    matrix: np.ndarray,
    marker: int,
    starting_cell_ids: np.ndarray,
    stopping_face_ids: np.ndarray,
    components: np.ndarray,
):
    queue = np.array(starting_cell_ids)
    while queue.size > 0:
        cell_id, queue = queue[0], queue[1:]
        components[cell_id] = marker

        face_ids = nonzero(matrix[:, cell_id])
        interior_face_ids = np.setdiff1d(face_ids, stopping_face_ids)
        neighbor_ids = nonzero(count_nonzero(matrix[interior_face_ids, :], 0))
        component_ids = nonzero(components == marker)
        unmarked_neighbor_ids = np.setdiff1d(neighbor_ids, component_ids)
        queue = np.union1d(queue, unmarked_neighbor_ids)


def mark_components(D: Topology, separator_ids: np.ndarray) -> np.ndarray:
    r"""Given a polytopal complex and the IDs of some faces known to be on a
    separator, find the rest of the separator and the remaining connected
    components and return a list of the associated markings"""
    D_1, D_2 = D[-2], D[-1]
    components = -np.ones(D_2.shape[0], dtype=int)

    # First, mark the connected component of the initial separators as 0
    cell_ids = nonzero(D_2)
    stop_face_ids = nonzero(count_nonzero(D_1[:, cell_ids], 1))
    _mark_component(D_1, 0, separator_ids, stop_face_ids, components)

    # Find all the faces of the separating cells; no connected component cxan
    # cross these faces
    cell_ids = nonzero(components == 0)
    face_ids = nonzero(count_nonzero(D_1[:, cell_ids], 1))

    # Mark all the remaining connected comonents
    marker = 1
    unmarked_cell_ids = nonzero(components == -1)
    while unmarked_cell_ids.size > 0:
        _mark_component(D_1, marker, [unmarked_cell_ids[0]], face_ids, components)
        marker += 1
        unmarked_cell_ids = nonzero(components == -1)

    return components


def face_split(D: Topology, components: np.ndarray) -> Topology:
    r"""Given a polytopal complex and a list of component markings (obtainable
    from the `mark_components` function), return the topology of the sub-
    divided complex split on the 0th component"""
    separator_ids = nonzero(components == 0)
    if separator_ids.size == 0:
        return D[-1]

    num_components = components.max()
    u = zeros((len(separator_ids), num_components))
    f = zeros((len(separator_ids), num_components))

    D_1, D_2 = D[-2], D[-1]
    S_F0 = D_1[:, separator_ids]
    A = S_F0.T @ S_F0

    for index in range(num_components):
        face_ids = nonzero(components == index + 1)
        S_F = D_1[:, face_ids]
        F_P = D_2[face_ids, :]
        f[:, [index]] = -S_F0.T @ S_F @ F_P

    # FIXME: This is really dodgy -- we're doing the linear solve in floating-
    # point and then rounding to integer. We should concoct a case where this
    # fails. The moral thing to do is to use Hermite normal form.
    u = np.rint(np.linalg.solve(A, f)).astype(np.int8)

    E_2 = zeros((D_2.shape[0], num_components))
    for index in range(num_components):
        face_ids = nonzero(components == index + 1)
        E_2[face_ids, index] = D_2[face_ids, 0]
        E_2[separator_ids, index] = u[:, index]

    return E_2


@functools.lru_cache(maxsize=10)
def standard_simplex(n: int) -> Topology:
    if n == 0:
        return [ones((1, 1))]
    return join_vertex(standard_simplex(n - 1))


def find_permutation_and_sign(A: np.ndarray, B: np.ndarray):
    r"""Return a permutation `p` and an array of signs `s` such that
    `A[:, p] @ diag(s) == B` if one exists"""
    if A.shape != B.shape:
        raise ValueError("Matrices must be the same shape!")

    # Handle the case of row of all 1s
    if np.array_equal(A, B):
        return np.arange(A.shape[1]), ones(A.shape[1])

    permutation = np.zeros(A.shape[1], dtype=int)
    signs = zeros(A.shape[1])
    for j, a_j in enumerate(A.T):
        matches = [
            k for k, b_k in enumerate(B.T) if np.array_equal(nonzero(a_j), nonzero(b_k))
        ]

        if len(matches) != 1:
            return None

        k, b_k = matches[0], B[:, matches[0]]
        if not (np.array_equal(a_j, b_k) or np.array_equal(a_j, -b_k)):
            return None

        permutation[j] = k
        signs[j] = np.sign(np.inner(a_j, b_k))

    return permutation, signs


def permutation_matrix(permutation: np.ndarray) -> np.ndarray:
    return eye(len(permutation))[:, permutation]


def find_isomorphism(As: Topology, Bs: Topology):
    r"""Given two sets `As` and `Bs` of boundary matrices, return permutations
    and sign flips that will transform `As` into `Bs` or return `None` if no
    such isomorphism exists"""
    if not len(As) == len(Bs):
        raise ValueError("Topologies must have the same dimension!")
    if not all([A.shape == B.shape for A, B in zip(As, Bs)]):
        raise ValueError("All boundary operators must have the same shape!")

    N = As[0].shape[0]
    permutations_and_signs = [(np.arange(N, dtype=int), ones(N))]

    for index, (A, B) in enumerate(zip(As, Bs)):
        permutation, sign = permutations_and_signs[-1]
        P = permutation_matrix(permutation)
        S = np.diag(sign)
        permutation_and_sign = find_permutation_and_sign(A, S @ P.T @ B)
        if permutation_and_sign is None:
            raise ValueError("No mapping found for dimension %d!" % index)
        permutations_and_signs.append(permutation_and_sign)

    return list(zip(*permutations_and_signs))


def orientation(D: Topology) -> int:
    r"""Given a polytopal complex that is assumed isomorphic to the standard
    simplex, return +1 if this complex is positively-oriented with the given
    vertex ordering or -1 if it is negatively oriented"""
    Δ = standard_simplex(len(D) - 1)
    permutations, signs = find_isomorphism(D, Δ)
    if not np.unique(signs[-1]).size == 1:
        raise ValueError("Top cells mapped non-trivially!")
    return signs[-1][0]


def to_simplicial(D: Topology) -> np.ndarray:
    r"""Given a polytopal complex that is assumed to be simplicial, return the
    array of vertex IDs in each simplex, ordered to be positive"""
    num_cells = D[-1].shape[1]
    dimension = len(D) - 1
    cells = np.zeros((num_cells, dimension + 1), dtype=np.uintp)
    for cell_id in range(num_cells):
        cells_ids = closure(D, [cell_id])
        Dc = subcomplex(D, cells_ids)
        vertex_ids = cells_ids[1]
        try:
            sign = orientation(Dc)
        except ValueError as exc:
            raise ValueError(
                "No isomorphism between cell %d and standard simplex!" % cell_id
            )

        if sign == -1:
            vertex_ids[:2] = vertex_ids[:2][::-1]
        cells[cell_id] = vertex_ids

    return cells


def _cell_counts_and_id_maps(simplices: np.ndarray):
    dimension = simplices.shape[1] - 1

    cell_counts = np.zeros(dimension + 1, dtype=int)
    cell_counts[0] = simplices.max() + 1
    cell_counts[dimension] = len(simplices)

    cell_id_maps = [{} for k in range(dimension)]
    cell_id_maps[0] = {(idx,): idx for idx in range(int(simplices.max()) + 1)}

    for simplex in simplices:
        for k in range(1, dimension):
            for face in itertools.combinations(tuple(np.sort(simplex)), k + 1):
                if face not in cell_id_maps[k]:
                    cell_id_maps[k][face] = cell_counts[k]
                    cell_counts[k] += 1

    return cell_counts, cell_id_maps


def from_simplicial(simplices: np.ndarray) -> Topology:
    r"""Given an array of vertex IDs of each simplex, return the equivalent
    set of boundary operators"""
    dimension = simplices.shape[1] - 1
    cell_counts, cell_id_maps = _cell_counts_and_id_maps(simplices)
    matrices = [ones((1, cell_counts[0]))] + [
        zeros((fcounts, ccounts))
        for fcounts, ccounts in zip(cell_counts[:-1], cell_counts[1:])
    ]

    for cell_id, cell in enumerate(simplices):
        for face in itertools.combinations(tuple(np.sort(cell)), dimension):
            face_id = cell_id_maps[dimension - 1][tuple(np.sort(face))]
            matrices[-1][face_id, cell_id] = simplicial.incidence(cell, face)

    for k in range(1, dimension):
        for cell, cell_id in cell_id_maps[k].items():
            for face in itertools.combinations(cell, k):
                face_id = cell_id_maps[k - 1][face]
                matrices[k][face_id, cell_id] = simplicial.incidence(cell, face)

    return matrices


def _polygon(num_vertices: int) -> Topology:
    d_0 = ones((1, num_vertices))
    edge = np.array([-1, +1] + [0] * (num_vertices - 2))
    d_1 = scipy.linalg.circulant(edge)
    d_2 = ones((num_vertices, 1))
    return [d_0, d_1, d_2]


class RandomPolygon:
    def __init__(self, num_vertices: int, rng: np.random.Generator):
        self._topology = _polygon(num_vertices)
        self._rng = rng
        self._nz = count_nonzero(self._topology[2], axis=0)

    @property
    def topology(self):
        return self._topology

    def step(self):
        d_0, d_1, d_2 = self._topology
        if self._nz.max() == 3:
            return

        cell_id = nonzero(self._nz > 3)[0]
        cells_ids = closure([d_0, d_1, d_2], [cell_id])
        f_0, f_1, f_2 = subcomplex([d_0, d_1, d_2], cells_ids)
        vertex_ids, edge_ids, poly_ids = cells_ids[1:]

        # Add an edge at random between two unconnected vertices
        vertex0 = self._rng.choice(list(range(len(vertex_ids))))
        G = f_1 @ f_1.T
        vertex1 = self._rng.choice(nonzero(G[vertex0, :] == 0))

        edge = zeros(len(vertex_ids))
        edge[[vertex0, vertex1]] = (-1, +1)
        f_1 = np.column_stack((f_1, edge))
        f_2 = np.vstack((f_2, zeros((1, 1))))

        # Subdivide the polygon along the newly-added edge
        edge_id = len(edge_ids)
        components = mark_components([f_0, f_1, f_2], [edge_id])
        g_2 = face_split([f_0, f_1, f_2], components)

        # Add the new polygons back into the global topology
        # FIXME: This is gross, stop being gross
        edge = zeros(d_1.shape[0])
        edge[[vertex_ids[vertex0], vertex_ids[vertex1]]] = (-1, +1)
        d_1 = np.column_stack((d_1, edge))
        d_2 = np.vstack((d_2, zeros((1, d_2.shape[1]))))

        d_2[edge_ids, cell_id] = g_2[:-1, 0]
        d_2[-1, cell_id] = g_2[-1, 0]

        column = zeros(d_2.shape[0])
        column[edge_ids] = g_2[:-1, 1]
        column[-1] = g_2[-1, 1]
        d_2 = np.column_stack((d_2, column))

        self._topology = [d_0, d_1, d_2]
        self._nz = count_nonzero(d_2, axis=0)

    def is_done(self):
        return self._nz.max() == 3

    def finalize(self):
        d_0, d_1, d_2 = self._topology
        p_1 = self._rng.permutation(d_1.shape[1])
        s_1 = self._rng.choice([+1, -1], size=d_1.shape[1])
        p_2 = self._rng.permutation(d_2.shape[1])
        s_2 = self._rng.choice([+1, -1], size=d_2.shape[1])

        e_1 = d_1[:, p_1] @ np.diag(s_1)
        e_2 = np.diag(s_1) @ d_2[p_1, :][:, p_2] @ np.diag(s_2)

        return [d_0, e_1, e_2]

    def run(self):
        while not self.is_done():
            self.step()
        return self.finalize()


def random_polygon(num_vertices: int, rng: np.random.Generator) -> Topology:
    return RandomPolygon(num_vertices, rng).run()
