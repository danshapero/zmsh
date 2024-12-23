import pytest
import numpy as np
from zmsh.simplification import compute_plane


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_computing_planes(dimension):
    num_trials = 20
    for trial in range(num_trials):
        rng = np.random.default_rng(seed=1729)
        xs = rng.standard_normal((dimension, dimension))
        p = compute_plane(xs)
        for trial in range(num_trials):
            ts = rng.uniform(size=dimension)
            ts /= np.sum(ts)
            w = ts @ xs
            assert np.abs(p[0] + np.dot(p[1:], w)) < 1e-6
