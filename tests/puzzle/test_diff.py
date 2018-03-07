"""
Test image diff soluzions.
"""
import numpy as np

import pytest

from puzzle import puzzle


# Create arrays
SIZE = 800
TEST_TARGET = np.random.randint(0, 255, (SIZE, SIZE), np.int16)
TEST_ARRAY = np.random.randint(0, 255, (SIZE, SIZE), np.int16)

# Select a cell to swap
CELL_0 = tuple(np.random.randint(0, SIZE, (2,)).tolist())
CELL_1 = tuple(np.random.randint(0, SIZE, (2,)).tolist())

INITIAL_DIFF = ((TEST_TARGET - TEST_ARRAY) ** 2).sum()

DIFF_CELL_0 = (TEST_TARGET[CELL_0] - TEST_ARRAY[CELL_0]) ** 2
DIFF_CELL_1 = (TEST_TARGET[CELL_1] - TEST_ARRAY[CELL_1]) ** 2
LOCAL_START_DIFF = DIFF_CELL_0 + DIFF_CELL_1

puzzle.swap(TEST_ARRAY, 1, (CELL_0, CELL_1))

FINAL_DIFF = ((TEST_TARGET - TEST_ARRAY) ** 2).sum()


def full(target, ary, initial=0):
    """Compute diff in all array"""
    return ((TEST_TARGET - TEST_ARRAY) ** 2).sum()


def delta(target, ary, initial=0):
    """Compute diff using only changed cells"""
    return INITIAL_DIFF + (TEST_TARGET[CELL_0] - TEST_ARRAY[CELL_0]) ** 2 + (TEST_TARGET[CELL_1] - TEST_ARRAY[CELL_1]) ** 2 - LOCAL_START_DIFF


@pytest.mark.parametrize('func', [delta, full])
def test_diffs(benchmark, func):
    """Test different image diff functions."""
    assert benchmark(func, TEST_TARGET, TEST_ARRAY, INITIAL_DIFF) == FINAL_DIFF
