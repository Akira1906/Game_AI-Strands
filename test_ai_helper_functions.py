from ai_helper_functions import gen_random_combinations, gen_combinations
import random

def test_gen_random_combinations():
    # Test case 1: Empty iterable
    iterable = []
    r = 3
    expected_output = []
    assert sorted(list(gen_random_combinations(iterable, r))) == sorted(expected_output), f"expected_output: {expected_output}, actual_output: {list(gen_random_combinations(iterable, r))}"

    # Test case 2: r is greater than the length of the iterable
    iterable = [1, 2, 3]
    r = 4
    expected_output = []
    assert sorted(list(gen_random_combinations(iterable, r))) == sorted(expected_output), f"expected_output: {expected_output}, actual_output: {list(gen_random_combinations(iterable, r))}"

    # Test case 3: r is equal to the length of the iterable
    iterable = [1, 2, 3]
    r = 3
    expected_output = [(1, 2, 3)]
    assert sorted(list(gen_random_combinations(iterable, r))) == sorted(expected_output), f"expected_output: {expected_output}, actual_output: {list(gen_random_combinations(iterable, r))}"

    # Test case 4: r is less than the length of the iterable
    iterable = [1, 2, 3, 4]
    r = 2
    expected_output = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert len(sorted(list(gen_random_combinations(iterable, r)))) == len(sorted(expected_output)), f"expected_output: {expected_output}, actual_output: {list(gen_random_combinations(iterable, r))}"

test_gen_random_combinations()

import timeit

def test_gen_random_combinations_performance():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = 3

    # Measure the execution time of gen_random_combinations
    start_time = timeit.default_timer()
    for _ in range(1000):
        gen = gen_random_combinations(iterable, r)
        next(gen)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"rand_execution_time: {execution_time}")
    # Assert that the execution time is less than 1 second
    assert execution_time < 1.0, f"execution_time: {execution_time}"
    
def test_gen_combinations_performance():
    iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = 3

    # Measure the execution time of gen_random_combinations
    start_time = timeit.default_timer()
    for _ in range(1000):
        random.choice(list(gen_combinations(iterable, r)))
    end_time = timeit.default_timer()
    execution_time = end_time - start_time

    # Assert that the execution time is less than 1 second
    print(f"normal_execution_time: {execution_time}")
    assert execution_time < 5, f"execution_time: {execution_time}"

test_gen_random_combinations_performance()
test_gen_combinations_performance()