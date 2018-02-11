"""
By counting the number of differences between two homologous DNA strands
taken from different genomes with a common ancestor, we get a measure of
the minimum number of point mutations that could have occurred on the
evolutionary path between the two strands.

This is called the 'Hamming distance'.
"""

# 3rd party
from numpy import absolute


def calculate_c_style(seq_1, seq_2):
    """
    Calculate Hamming distance between `seq_1` and `seq_2`.

    Non-pythonic implementation.
    """
    result = 0
    length = len(seq_1)
    for i in range(length):
        el_1 = seq_1[i]
        el_2 = seq_2[i]
        if el_1 != el_2:
            result += 1
    return result


def calculate(seq_1, seq_2):
    """
    Calculate Hamming distance between `seq_1` and `seq_2`.

    The function is type-agnostic (works with different kind of sequences).
    """
    return sum([a != b for a, b in zip(seq_1, seq_2)])


def calculate_np_array(seq_1, seq_2):
    """
    Calculate Hamming distance between numpy arrays `seq_1` and `seq_2`.
    """
    return absolute(seq_1 - seq_2).sum()
