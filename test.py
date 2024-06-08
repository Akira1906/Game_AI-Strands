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

