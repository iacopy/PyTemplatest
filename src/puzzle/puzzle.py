"""
Puzzle functions.
"""

# Standard Library
from copy import copy


def swap(ary, size, cells):
    """
    Given an array `ary`, swap 2 `cells`.

    Each cell is `size` x `size`.
    """
    def iter_cell(cell, axis):
        """Return the ``cell`` ``axis`` iterator, given implicit size.

        axis=0 -> row, axis=1 -> col
        """
        return range(cell[axis] * size, cell[axis] * size + size)

    cell_0, cell_1 = cells
    for y_0, y_1 in zip(iter_cell(cell_0, 0), iter_cell(cell_1, 0)):
        for (x_0, x_1) in zip(iter_cell(cell_0, 1), iter_cell(cell_1, 1)):
            # could not managed to do it without tmp
            tmp = copy(ary[y_0][x_0])
            ary[y_0][x_0] = ary[y_1][x_1]
            ary[y_1][x_1] = tmp
