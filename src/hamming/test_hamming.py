"""
Tests different Hamming distance implementations.
"""

# Standard Library
import array
import random
from functools import partial

# 3rd party
import numpy as np
import pytest
from hamming.hamming import calculate
from hamming.hamming import calculate_c_style
from hamming.hamming import calculate_np_array


def build_random_test_data(n_sequences, length):
    """
    Generate `n_sequences` random test cases.

    Each case is a dictionary with following keys:
        "seq_1" -- binary 0|1 list of length `length`;
        "seq_2" -- binary 0|1 list derived from `seq_1`
            with a random number of mutations;
        "distance" -- number of mutations between `seq_1` and `seq_2`.
    """
    cases = []
    for _ in range(n_sequences):
        seq = [random.choice([0, 1]) for _ in range(length)]
        n_mutations = random.randint(0, length)
        mutated_positions = random.sample(list(range(length)), n_mutations)
        mutated_seq = list(seq)
        for position in mutated_positions:
            mutated_seq[position] = int(not mutated_seq[position])
        case = {'seq_1': seq, 'seq_2': mutated_seq, 'distance': n_mutations}
        cases.append(case)
    return cases


CASE = build_random_test_data(1, 100)[0]


@pytest.mark.parametrize('constructor,calculator', [
    (list, calculate),
    (list, calculate_c_style),
    (partial(array.array, 'b'), calculate_c_style),
    (partial(array.array, 'b'), calculate),
    (np.array, calculate_np_array),
    ], ids=['list-c_style', 'list-list_comp',
            'arrray.array-c_style', 'arrray.array-list_comp',
            'np.array'])
def test_hamming(constructor, calculator, benchmark):
    """
    Parametric data driven benchmark test for hamming function.
    """
    seq_1 = constructor(CASE['seq_1'])
    seq_2 = constructor(CASE['seq_2'])
    assert benchmark(calculator, seq_1, seq_2) == CASE['distance']
