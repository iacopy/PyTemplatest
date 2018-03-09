"""
Test image diff soluzions.
"""
import numpy as np

import pytest

from puzzle import puzzle
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


SEED = 123456

seeds_ret = {
    37: 6939729215,
    123456: 6943999565,
}

np.random.seed(SEED)

# Create arrays
SHAPE = 800, 800
TEST_TARGET = np.random.randint(0, 255, SHAPE, np.int64)
TEST_ARRAY = np.random.randint(0, 255, SHAPE, np.int64)


def generate_random_swap_cells(shape, how_many):
    rv = []
    while len(rv) < how_many:
        while True:
            cell_0 = np.random.randint(shape[0]), np.random.randint(shape[1])
            cell_1 = np.random.randint(shape[0]), np.random.randint(shape[1])
            if cell_0 != cell_1:
                break
        rv.append((cell_0, cell_1))
    return rv


def rand_image_array(shape):
    return np.random.randint(0, 255, shape, np.int32)


@given(
    target=arrays(np.int32, (200, 100), elements=st.integers(0, 255)),
    ary=arrays(np.int32, (200, 100), elements=st.integers(0, 255)),
    cell_0=st.tuples(st.integers(0, 199), st.integers(0, 99)),
    cell_1=st.tuples(st.integers(0, 199), st.integers(0, 99)),
)
def test_diff_algs(target, ary, cell_0, cell_1):
    """
    Test that smart diff method return the same results
    than full one.

    """
    assume(cell_0 != cell_1)
    assert 0 <= target[cell_0] <= 256
    initial_diff = diff_full(target, ary)
    initial_cells_diff = diff_cells(target, ary, cell_0, cell_1)
    assert initial_cells_diff <= initial_diff

    puzzle.swap(ary, 1, cell_0, cell_1)

    final_cells_diff = diff_cells(target, ary, cell_0, cell_1)
    smart_diff = initial_diff - initial_cells_diff + final_cells_diff
    final_diff = diff_full(target, ary)

    assert smart_diff == final_diff, '{} != {} (actual)'.format(smart_diff, final_diff)


def diff_full(target, ary):
    """Compute diff in all array"""
    return ((target - ary) ** 2).sum()


def diff_cells(target, ary, cell_0, cell_1):
    # print('target[c0={}] = {}'.format(cell_0, target[cell_0]))
    # print('ary[c0={}] = {}'.format(cell_0, target[cell_0]))
    # print('target[c1={}] = {}'.format(cell_1, target[cell_1]))
    # print('ary[c1={}] = {}'.format(cell_1, target[cell_1]))
    rv = (target[cell_0] - ary[cell_0]) ** 2 + (target[cell_1] - ary[cell_1]) ** 2
    return rv


def workflow_full(target, ary, swap_cells):
    for cell_0, cell_1 in swap_cells:
        puzzle.swap(ary, 1, cell_0, cell_1)
        diff = diff_full(target, ary)
    return diff


def workflow_smart(target, ary, swap_cells):
    swap = puzzle.swap
    diff = diff_full(target, ary)
    for cell_0, cell_1 in swap_cells:
        dc0 = diff_cells(target, ary, cell_0, cell_1)
        swap(ary, 1, cell_0, cell_1)
        dc1 = diff_cells(target, ary, cell_0, cell_1)
        diff = diff - dc0 + dc1
    return diff


def workflow_smarter(target, ary, swap_cells):
    """
    Use the one-sized swap
    """
    swap = puzzle.swap_size_one
    diff = diff_full(target, ary)
    for cell_0, cell_1 in swap_cells:
        dc0 = diff_cells(target, ary, cell_0, cell_1)
        swap(ary, cell_0, cell_1)
        dc1 = diff_cells(target, ary, cell_0, cell_1)
        diff = diff - dc0 + dc1
    return diff


@pytest.mark.parametrize('workflow', [
    workflow_full, workflow_smart, workflow_smarter
])
def test_workflow(benchmark, workflow):
    shape = (1000, 1000)
    target = rand_image_array(shape)
    ary = rand_image_array(shape)
    swap_cells = generate_random_swap_cells(ary.shape, 2000)
    assert benchmark(workflow, target, ary, swap_cells)
