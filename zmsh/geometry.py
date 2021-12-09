import itertools


def compositions(number, size):
    r"""Compute the number of weak compositions of `number` of `size` integers

    From this [SO answer](https://stackoverflow.com/a/40540014)
    """
    m = number + size - 1
    last = (m,)
    first = (-1,)
    for t in itertools.combinations(range(m), size - 1):
        yield tuple(v - u - 1 for u, v in zip(first + t, t + last))
