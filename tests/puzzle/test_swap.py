"""
Test about swapping puzzle cells.
"""

# 3rd party
import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes
from hypothesis.extra.numpy import arrays

# My stuff
from puzzle import puzzle
from swap_testdata import ARY_4x4
from swap_testdata import ARY_4x4_S01
from swap_testdata import ARY_4x4_S01_S13
from swap_testdata import ARY_4x4x3
from swap_testdata import ARY_4x4x3_S01
from swap_testdata import ARY_4x4x3_S01_S13


@pytest.mark.parametrize('initial,intermediate,final', [
    [ARY_4x4, ARY_4x4_S01, ARY_4x4_S01_S13],
    [ARY_4x4x3, ARY_4x4x3_S01, ARY_4x4x3_S01_S13],
])
def test_simple_swap(initial, intermediate, final):
    """A couple of 2x2 swap tests.

    This test supports the more general 'test_double_swap' in which
    only the final result of a double swap (in many randomized cases) is
    checked, but the result of the first swap can't be checked.
    """
    ary = np.array(initial)

    puzzle.swap(ary, 2, (0, 0), (0, 1))

    # I'm using array.tolist for the comparison instead of np.array_equal
    # to better visualize in case of fail
    assert ary.tolist() == intermediate

    puzzle.swap(ary, 2, (0, 1), (1, 1))

    assert ary.tolist() == final


@st.composite
def swap_cases(draw):
    """
    Return a swap test case in a dict with these keys:
        ary -- array with random shape
        cell_size -- random cell size from 1 to half `ary` dim
        src_cell -- random swap source cell
        dst_cell -- random swap destination cell

    The 'composite' decorator is needed to use hypothesis strategies
    examples as input to other strategies (e.g. cell_size needs
    the array shape).
    """
    # i need a concrete shape (-> draw) to calculate its minimum
    shape = draw(array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=16))
    # minimum of dims is needed to have the max cell size (half of min dim)
    max_cell_size = max(int(min(shape) / 2), 1)
    cell_size = draw(st.integers(min_value=1, max_value=max_cell_size))
    #: is the shape of the puzzle in terms of its pieces (cells)
    subdivided_shape = int(shape[0] / cell_size), int(shape[1] / cell_size)
    n_cells = np.prod(subdivided_shape)

    ary = draw(arrays(np.int16, shape, elements=st.integers(
        min_value=0, max_value=255)))
    # To generate the cells coordinates that will be swapped, I decided to
    # use a 1D-index (which then I convert to 2D with np.unravel_index)
    src_item = draw(st.integers(min_value=0, max_value=n_cells - 1))
    dst_item = draw(st.integers(min_value=0, max_value=n_cells - 1))
    src_cell = np.unravel_index(src_item, subdivided_shape)
    dst_cell = np.unravel_index(dst_item, subdivided_shape)
    return dict(
        ary=ary, cell_size=cell_size, src_cell=src_cell, dst_cell=dst_cell
    )


@given(case=swap_cases())
def test_double_swap(case):
    """
    A double swap (within same cells) should finally result like a no-op,
    no matter which couple of cells are involved, or their sizes.

    The first swap result can't be exactly checked because positions and values
    are randomized. Furthermore, we can't even check that first swap
    modify no-matter-how the array, as it is not guaranteed in several cases
    (e.g. if swapping a cell with itself or the contents of selected cells are
    the same).

    Thus, a simple exact test for a single swap is needed too.
    """
    ary = case['ary']
    initial_array = ary.copy()

    #: size of puzzle pieces
    cell_size = case['cell_size']

    src_cell = case['src_cell']
    dst_cell = case['dst_cell']

    puzzle.swap(ary, cell_size, src_cell, dst_cell)
    puzzle.swap(ary, cell_size, src_cell, dst_cell)

    assert np.array_equal(ary, initial_array)


@given(
    ary=arrays(np.uint8, (8, 8), unique=True),
    src_cell=st.tuples(st.integers(min_value=0, max_value=7),
                       st.integers(min_value=0, max_value=7)),
    dst_cell=st.tuples(st.integers(min_value=0, max_value=7),
                       st.integers(min_value=0, max_value=7)),
)
def test_one_sized_cell_double_swap(ary, src_cell, dst_cell):
    """
    Complete single-sized cells double swap tests,
    assuming unique values and different cells.

    8 x 8 shaped array is chosen because 16 x 16 (max shape with
    0-255 unique values) results in exception:
    'hypothesis.errors.FailedHealthCheck: Data generation is extremely slow'
    """
    assume(src_cell != dst_cell)
    initial_array = ary.copy()

    # The first swap MUST give a different array
    puzzle.swap(ary, 1, src_cell, dst_cell)
    assert not np.array_equal(ary, initial_array)

    # The second swap must give the initial array
    puzzle.swap(ary, 1, src_cell, dst_cell)
    assert np.array_equal(ary, initial_array)
