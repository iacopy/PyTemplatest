"""
Functions to randomize puzzles
and save them to disk.

Usage::

    python -m src.puzzle.rand_puzzle <image_file_path>
"""
# 3rd party
import numpy as np

from . import cli
from . import util
from .puzzle import swap


def rand_cell(shape):
    """
    Get a random 2D-index given a `shape`.
    """
    row = np.random.randint(0, shape[0])
    column = np.random.randint(0, shape[1])
    return row, column


def random_puzzle(ary, cell_size, swaps):
    """
    Create a random puzzle from an array representing a grayscale image.
    """
    initial_sum = ary.sum()

    subdivided_shape = (
        int(ary.shape[0] / cell_size),
        int(ary.shape[1] / cell_size)
    )

    for i in range(swaps):
        src_cell = rand_cell(subdivided_shape)
        dst_cell = rand_cell(subdivided_shape)
        swap(ary, cell_size, src_cell, dst_cell)

        assert ary.sum() == initial_sum, \
            'Checksum failed after {} swaps! {} -> {}'.format(
                i + 1, initial_sum, ary.sum())


def main(options):
    """
    Create a random puzzle from a source image.
    """
    ary = util.load_image(options.src)

    cell_size, swaps = cli.process_options(options, shape=ary.shape)

    # Let's avoid "ValueError: assignment destination is read-only"
    ary.flags['WRITEABLE'] = True

    if cell_size[0] != cell_size[1]:
        print('Warning: non-squared cell size not yet supported. Using ({0}, {0})'.format(
            cell_size[0]))
    random_puzzle(ary, cell_size[0], swaps)

    util.save_image(ary, options.dst)


if __name__ == '__main__':
    main(cli.get_options())
