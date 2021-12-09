from math import comb as binomial
import zmsh


def test_compositions():
    max_dimension = 4
    max_degree = max_dimension + 2
    for dimension in range(max_dimension):
        for degree in range(1, max_degree):
            num_compositions = binomial(degree + dimension, degree)
            compositions = zmsh.compositions(degree, dimension + 1)
            assert len(set(compositions)) == num_compositions

            for composition in compositions:
                assert len(composition) == dimension + 1
                assert sum(composition) == degree
