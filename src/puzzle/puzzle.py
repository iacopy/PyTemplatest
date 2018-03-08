"""
Puzzle functions.
"""

# Standard Library
from copy import copy


def swap(ary, size, cell_0, cell_1):
    """
    Given an array `ary`, swap `cell_0` with `cell_1`.
    Each cell is `size` x `size`.
    """
    def iter_cell(cell, axis):
        """Return the ``cell`` ``axis`` iterator, given implicit size.

        axis=0 -> row, axis=1 -> col
        """
        return range(cell[axis] * size, cell[axis] * size + size)

    for y_0, y_1 in zip(iter_cell(cell_0, 0), iter_cell(cell_1, 0)):
        for (x_0, x_1) in zip(iter_cell(cell_0, 1), iter_cell(cell_1, 1)):
            # could not managed to do it without tmp
            tmp = copy(ary[y_0][x_0])
            ary[y_0][x_0] = ary[y_1][x_1]
            ary[y_1][x_1] = tmp


def swap_size_one(ary, index_0, index_1):
    """
    Swap ``ary`` items at ``index_0`` and ``index_1``.
    """
    tmp = copy(ary[index_0])
    ary[index_0] = ary[index_1]
    ary[index_1] = tmp
