import numpy as np
from zmsh import polytopal, simplicial


def test_random_polygon():
    rng = np.random.default_rng(seed=1729)

    num_trials = 20
    min_num_vertices = 4
    max_num_vertices = 10
    for num_vertices in range(min_num_vertices, max_num_vertices + 1):
        for trial in range(num_trials):
            random_polygon = polytopal.RandomPolygon(num_vertices, rng)
            max_num_sides = [num_vertices]
            while not random_polygon.is_done():
                random_polygon.step()
                d_2 = random_polygon.topology[2]
                nz = np.count_nonzero(d_2, axis=0)
                max_num_sides.append(nz.max())

            assert all(
                [nz2 <= nz1 for nz1, nz2 in zip(max_num_sides[:-1], max_num_sides[1:])]
            )

            topology = random_polygon.finalize()
            simplices = polytopal.to_simplicial(topology)
