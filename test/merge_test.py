import numpy as np
from zmsh import polytopal


def test_merging_two_triangles():
    d_0 = np.ones((1, 4), dtype=np.int8)
    d_1 = np.array(
        [
            [0, +1, -1, 0, 0],
            [-1, 0, +1, -1, 0],
            [+1, -1, 0, 0, +1],
            [0, 0, 0, +1, -1],
        ],
        dtype=np.int8,
    )
    d_2 = np.array([[+1, -1], [+1, 0], [+1, 0], [0, +1], [0, +1]], dtype=np.int8)

    s = polytopal.merge([d_0, d_1, d_2], face_ids=[0])
    s_expected = np.array([+1, +1], dtype=np.int8)
    assert np.array_equal(s, s_expected)

    d_1[:, 0] *= -1
    d_2[0, :] *= -1
    s = polytopal.merge([d_0, d_1, d_2], face_ids=[0])
    assert np.array_equal(s, s_expected)

    d_2[:, 1] *= -1
    s = polytopal.merge([d_0, d_1, d_2], face_ids=[0])
    s_expected = np.array([+1, -1], dtype=np.int8)
    assert np.array_equal(s, s_expected) or np.array_equal(-s, s_expected)


def test_merging_three_triangles():
    triangle = polytopal.standard_simplex(2)
    d_0, d_1, d_2 = polytopal.vertex_split(triangle)

    face_ids = np.flatnonzero(np.count_nonzero(d_2, axis=1) >= 2)
    assert len(face_ids) == 3
    s_1 = polytopal.merge([d_0, d_1, d_2], face_ids)
    assert s_1 is not None

    e_2 = d_2.copy()
    e_2[:, 0] *= -1
    s_2 = polytopal.merge([d_0, d_1, e_2], face_ids)
    s_2_expected = s_1.copy()
    s_2_expected[0] *= -1
    assert np.array_equal(s_2, s_2_expected) or np.array_equal(-s_2, s_2_expected)


def test_cant_merge_mace():
    d_0 = np.ones((1, 5), dtype=np.int8)
    d_1 = np.array(
        [
            [-1, 0, +1, 0, +1, 0, +1],
            [+1, -1, 0, -1, 0, -1, 0],
            [0, +1, -1, 0, 0, 0, 0],
            [0, 0, 0, +1, -1, 0, 0],
            [0, 0, 0, 0, 0, +1, -1],
        ],
        dtype=np.int8,
    )
    d_2 = np.array(
        [
            [+1, +1, +1],
            [+1, 0, 0],
            [+1, 0, 0],
            [0, +1, 0],
            [0, +1, 0],
            [0, 0, +1],
            [0, 0, +1],
        ],
        dtype=np.int8,
    )

    assert polytopal.merge([d_0, d_1, d_2], [0]) is None


def test_merging_random_polygon():
    rng = np.random.default_rng(seed=112358)
    num_trials = 20
    min_num_vertices = 4
    max_num_vertices = 10
    for num_vertices in range(min_num_vertices, max_num_vertices + 1):
        for trial in range(num_trials):
            d_0, d_1, d_2 = polytopal.random_polygon(num_vertices, rng)
            edge_ids = np.flatnonzero(np.count_nonzero(d_2, axis=1) > 1)
            s = polytopal.merge([d_0, d_1, d_2], edge_ids)
            e_2 = d_2 @ s
            assert np.max(np.abs(d_1 @ e_2)) == 0
            assert (s != 0).all()
