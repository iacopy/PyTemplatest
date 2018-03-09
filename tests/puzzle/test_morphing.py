# 3rd party
import numpy as np
import pytest

# My stuff
from puzzle import morphing
import animation

from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@pytest.mark.parametrize('first,second,expected', [
    ([[0, 0, 1], [1, 0, 2]], [[0, 0, 0], [0, 2, 2]], 6),
    ([[0, 4], [10, 2]], [[1, 1], [0, 2]], 110),
])
def test_square_diff_sum(first, second, expected):
    first = np.array(first, dtype=np.int16)
    second = np.array(second, dtype=np.int16)
    assert morphing.square_diff_sum(first, second) == expected


@pytest.mark.parametrize('source,target,expected', [
    (
        [[4, 3], [2, 1]],
        [[1, 2], [3, 4]],
        [[1, 2], [3, 4]],
    ),
    (
        [[180, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 20]],
        [[0, 0], [0, 0], [0, 180]],
    ),
    (
        [[200, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 100]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 200]],
    ),
])
def test_final_step(source, target, expected):
    steps, last_array = morphing.search_morph_steps(source, target)
    assert last_array.tolist() == expected
    animation.build_animation(
        list(morphing.apply_morphing_steps(np.array(source, np.uint8), steps)),
        'anim.gif')


@given(
    source=arrays(np.int16, (3, 3), elements=st.integers(0, 5)),
    target=arrays(np.int16, (3, 3), elements=st.integers(0, 5)),
    cell_0=st.tuples(st.integers(min_value=0, max_value=2), st.integers(min_value=0, max_value=2)),
    cell_1=st.tuples(st.integers(min_value=0, max_value=2), st.integers(min_value=0, max_value=2)),
)
def test_update_diff(source, target, cell_0, cell_1):
    assume(cell_0 != cell_1)
    # print(target)
    # print(source)
    cells = (cell_0, cell_1)
    # print(cells)

    global_diff_0 = morphing.square_diff_sum(target, source)
    # print('global_0 =', global_diff_0)
    start_diff, end_diff = morphing.update_diff(target, source, cells)
    # print('start_local_diff =', start_diff)
    # print('end_local_diff =', end_diff)
    updiff = global_diff_0 + (end_diff - start_diff)
    # print('updiff = {} + {} + {}'.format(global_diff_0, start_diff, end_diff))
    global_diff_1 = morphing.square_diff_sum(target, source)
    # print('global_1 =', global_diff_1)
    assert global_diff_1 == updiff, '{} != {}'.format(global_diff_1, updiff)
