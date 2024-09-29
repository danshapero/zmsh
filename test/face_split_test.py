import pytest
import numpy as np
import predicates
from zmsh import polytopal


def quadrilateral_d1():
    return np.array(
        [[-1, 0, 0, +1], [+1, -1, 0, 0], [0, +1, -1, 0], [0, 0, +1, -1]], dtype=np.int8
    )


def test_no_separator():
    d_1 = quadrilateral_d1()
    d_2 = np.ones((4, 1), dtype=np.int8)
    components = polytopal.mark_components([d_1, d_2], [])
    components_expected = np.ones(4, dtype=int)
    assert np.array_equal(components, components_expected)

    e_2 = polytopal.face_split([d_1, d_2], components)
    assert np.array_equal(d_2, e_2)


def test_one_edge_separator():
    column = np.array([-1, 0, +1, 0], dtype=np.int8)
    d_1 = np.column_stack((quadrilateral_d1(), column))
    d_2 = np.ones((5, 1), dtype=np.int8)
    d_2[-1] = 0
    components = polytopal.mark_components([d_1, d_2], [4])
    assert components[-1] == 0
    assert np.unique(components[[0, 1]]).size == 1
    assert np.unique(components[[2, 3]]).size == 1

    e_2 = polytopal.face_split([d_1, d_2], components)
    assert np.sum(np.abs(d_1 @ e_2)) == 0
    assert np.array_equal(np.sum(e_2, axis=1), d_2.flatten())


def test_two_edge_separator():
    row = np.zeros((1, 4), dtype=np.int8)
    columns = np.array([[-1, 0, 0, 0, +1], [0, 0, +1, 0, -1]], np.int8).T
    d_1 = np.column_stack((np.vstack((quadrilateral_d1(), row)), columns))
    d_2 = np.ones((6, 1), dtype=np.int8)
    d_2[[4, 5]] = 0
    separator_edge_ids = [4, 5]
    components = polytopal.mark_components([d_1, d_2], separator_edge_ids)
    assert (components[separator_edge_ids] == 0).all()

    e_2 = polytopal.face_split([d_1, d_2], components)
    assert np.sum(np.abs(d_1 @ e_2)) == 0
    assert np.array_equal(np.sum(e_2, axis=1), d_2.flatten())

    d_1[:, -1] *= -1
    e_2 = polytopal.face_split([d_1, d_2], components)
    assert np.sum(np.abs(d_1 @ e_2)) == 0
    assert np.array_equal(np.sum(e_2, axis=1), d_2.flatten())


def test_no_separator_with_hanging():
    row = np.zeros((1, 4), dtype=np.int8)
    column = np.array([-1, 0, 0, 0, +1], dtype=np.int8)
    d_1 = np.column_stack((np.vstack((quadrilateral_d1(), row)), column))
    d_2 = np.ones((5, 1), dtype=np.int8)
    d_2[-1] = 0
    components = polytopal.mark_components([d_1, d_2], [])
    assert np.unique(components).size == 1

    e_2 = polytopal.face_split([d_1, d_2], components)
    assert e_2.shape == (5, 1)
    assert e_2[4] == 0


def test_one_edge_separator_with_hanging():
    row = np.zeros((1, 4), dtype=np.int8)
    columns = np.array([[-1, 0], [0, -1], [+1, 0], [0, 0], [0, +1]], dtype=np.int8)
    d_1 = np.column_stack((np.vstack((quadrilateral_d1(), row)), columns))
    d_2 = np.ones((6, 1))
    d_2[[4, 5]] = 0
    separator_edge_id = 4
    hanging_edge_id = 5

    components = polytopal.mark_components([d_1, d_2], [separator_edge_id])
    assert components[separator_edge_id] == 0
    assert np.unique(components[[2, 3]]).size == 1

    e_2 = polytopal.face_split([d_1, d_2], components)
    assert np.sum(np.abs(d_1 @ e_2)) == 0
    assert np.array_equal(np.sum(e_2, axis=1), d_2.flatten())
    assert np.count_nonzero(e_2[hanging_edge_id, :]) == 0


def test_degenerate_boundary():
    d_1 = np.array([[-1, -1, -1], [+1, +1, +1]], dtype=np.int8)
    d_2 = np.array([[+1], [-1], [0]], dtype=np.int8)
    separator_edge_id = 2
    components = polytopal.mark_components([d_1, d_2], [separator_edge_id])
    assert components[separator_edge_id] == 0
    assert np.unique(components).size == 3

    e_2 = polytopal.face_split([d_1, d_2], components)
    assert np.sum(np.abs(d_1 @ e_2)) == 0

    # TODO: Couldn't it be a permutation of this also...
    e_2_expected = np.array([[+1, 0], [0, -1], [-1, +1]], dtype=np.int8)
    assert np.array_equal(e_2, e_2_expected)


def test_three_way():
    row = np.zeros((1, 4), dtype=np.int8)
    columns = np.array(
        [[+1, 0, 0], [0, +1, 0], [0, 0, +1], [0, 0, 0], [-1, -1, -1]], dtype=np.int8
    )
    d_1 = np.column_stack((np.vstack((quadrilateral_d1(), row)), columns))
    d_2 = np.ones((7, 1))
    d_2[4:] = 0
    separator_edge_ids = [4, 5]
    components = polytopal.mark_components([d_1, d_2], separator_edge_ids)
    assert (components[4:] == 0).all()
    assert np.unique(components[:4]).size == 3

    e_2 = polytopal.face_split([d_1, d_2], components)
    assert np.sum(np.abs(d_1 @ e_2)) == 0
    assert np.array_equal(np.sum(e_2, axis=1), d_2.flatten())


def test_three_tetrahedra():
    triangle = polytopal.standard_simplex(2)
    tetrahedron = polytopal.join_vertex(triangle)
    hypertetrahedron = polytopal.join_vertex(tetrahedron)
    top_cell_ids = [2, 3, 4]
    cells_ids = polytopal.closure(hypertetrahedron[:-1], top_cell_ids)
    D_0, D_1, D_2, D_3 = polytopal.subcomplex(hypertetrahedron[:-1], cells_ids)
    D_3 = D_3 @ np.diag(triangle[-1].flatten())
    E_3 = D_3 @ np.ones((3, 1), dtype=np.int8)

    separator_face_ids = np.flatnonzero(E_3 == 0)
    components = polytopal.mark_components([D_2, E_3], separator_face_ids)
    F_3 = polytopal.face_split([D_2, E_3], components)
    assert np.array_equal(D_3, F_3)


def test_null_edge():
    d_0 = np.ones((4, 1), dtype=np.int8)
    d_1 = np.array(
        [
            [-1, 0, 0, +1, 0, 0, 0],
            [+1, -1, 0, 0, 0, 0, -1],
            [0, +1, -1, 0, 0, 0, 0],
            [0, 0, +1, -1, 0, 0, +1],
        ],
        dtype=np.int8,
    )
    d_2 = np.array([+1, +1, +1, +1, 0, 0, 0], dtype=np.int8)[:, None]

    separator_face_ids = [6]
    components = polytopal.mark_components([d_1, d_2], separator_face_ids)
    assert (components[[4, 5]] == -1).all()
    e_2 = polytopal.face_split([d_1, d_2], components)
    assert e_2.shape[1] == 2
