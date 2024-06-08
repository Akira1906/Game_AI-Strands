import random
import itertools

def gen_random_combinations(iterable, r, seed=123):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return

    # Create a random generator with a specific seed
    rand_gen = random.Random(seed)

    # Generate permutation indices in a pseudorandom order
    indices = list(range(r))
    max_index = len(list(itertools.combinations(range(n), r)))
    order = list(range(max_index))
    rand_gen.shuffle(order)

    for index in order:
        # Generate the specific permutation from the permutation index
        indices = nth_combination(index, range(n), r)
        yield tuple(pool[i] for i in indices)

def nth_combination(index, iterable, r):
    'Equivalent to list(itertools.combinations(iterable, r))[index]'
    pool = tuple(iterable)
    n = len(pool)
    if r < 0 or r > n:
        raise ValueError
    c = 1
    k = min(r, n-r)
    for i in range(1, k+1):
        c = c * (n - k + i) // i
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError
    result = []
    while r:
        c, n, r = c*r//n, n-1, r-1
        while index >= c:
            index -= c
            c, n = c*(n-r+1)//n, n-1
        result.append(pool[-1-n])
    return tuple(result)

def test_gen_random_combinations():
    # Test case 1: Empty iterable
    iterable = []
    r = 3
    expected_output = []
    assert list(gen_random_combinations(iterable, r)) == expected_output

    # Test case 2: r is greater than the length of the iterable
    iterable = [1, 2, 3]
    r = 4
    expected_output = []
    assert list(gen_random_combinations(iterable, r)) == expected_output

    # Test case 3: r is equal to the length of the iterable
    iterable = [1, 2, 3]
    r = 3
    expected_output = [(1, 2, 3)]
    assert list(gen_random_combinations(iterable, r)) == expected_output

    # Test case 4: r is less than the length of the iterable
    iterable = [1, 2, 3, 4]
    r = 2
    expected_output = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert list(gen_random_combinations(iterable, r)) == expected_output

    # Test case 5: Custom seed
    iterable = [1, 2, 3]
    r = 2
    seed = 987
    expected_output = [(2, 3), (1, 3), (1, 2)]
    assert list(gen_random_combinations(iterable, r, seed)) == expected_output

test_gen_random_combinations()