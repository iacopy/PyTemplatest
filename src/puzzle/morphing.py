"""
Morph a source image in a target one, just swapping pixels.

The final result is not guaranteed to be equal to the target
but similar due to immutable pixel values.
"""
import os
import pickle
import sys

from time import time as get_time

import numpy as np

from . import puzzle
from . import rand_puzzle
from ..animation import animation
from .util import load_image_as_grayscale
from .util import save_image


from collections import namedtuple

Bbox = namedtuple('bbox', ['min_row', 'min_col', 'max_row', 'max_col'])


def square_diff_sum(ary_1, ary_2):
    """
    Represent a difference, distance, between 2 image arrays.
    """
    return ((ary_1 - ary_2) ** 2).sum()


def diff_cells(target, ary, cell_0, cell_1):
    rv = (target[cell_0] - ary[cell_0]) ** 2 + (target[cell_1] - ary[cell_1]) ** 2
    return rv


def rand_swap_cells(ary, bbox=None):
    """
    Return a random couple of cells within the shape.
    """
    if bbox is None:
        bbox = Bbox(min_row=0, min_col=0, max_row=ary.shape[0], max_col=ary.shape[1])

    row_0 = np.random.randint(bbox.min_row, bbox.max_row)
    col_0 = np.random.randint(bbox.min_col, bbox.max_col)
    cell_0 = row_0, col_0
    # optimize
    choices = [(row_0 + i, col_0 + j) for i in range(-1, 2) for j in range(-1, 2)]
    while True:
        cell_1 = choices.pop(np.random.randint(len(choices)))
        if (cell_1 != cell_0) and (0 <= cell_1[0] < bbox.max_row and 0 <= cell_1[1] < bbox.max_col):
            return (row_0, col_0), cell_1



def explore(target, ary, best_diff, bbox=None, sub_trials=1000):
    def expand_bbox(bbox):
        height, width = ary.shape
        return Bbox(
            min_row=max(0, bbox.min_row - 1),
            min_col=max(0, bbox.min_col - 1),
            max_row=min(height, bbox.max_row + 1),
            max_col=min(width, bbox.max_col + 1),
        )
    #print('explore\n', ary, 'to\n', target)
    swap = puzzle.swap_size_one

    best_ary = ary.copy()
    trials_steps = []
    trials_steps_pointer = 0
    diff = best_diff

    cell_0, cell_1 = rand_swap_cells(ary, bbox)
    cell_diff_0 = diff_cells(target, ary, cell_0, cell_1)

    for i in range(sub_trials):
        swap(ary, cell_0, cell_1)
        cell_diff_1 = (target[cell_0] - ary[cell_0]) ** 2 + (target[cell_1] - ary[cell_1]) ** 2
        diff = square_diff_sum(target, ary)
        trials_steps.append((cell_0, cell_1))
        if diff < best_diff:
            best_diff = diff
            trials_steps_pointer = len(trials_steps)
            best_ary = ary.copy()
        else:
            # enlarge search bbox
            bbox = expand_bbox(bbox)

        if best_diff == 0:
            break

        # Cells for next swap
        cell_0, cell_1 = rand_swap_cells(ary, bbox)
        cell_diff_0 = cell_diff_1

    #print('best_diff = {diff:,}, swaps: {swaps:,}'.format(swaps=trials_steps_pointer, diff=best_diff))
    return best_ary, trials_steps[:trials_steps_pointer], best_diff


def save_steps(steps):
    with open('steps.csv', 'a') as fp:
        tow = ''.join(
            ['{}\t{}\t{}\t{}\n'.format(cell_0[0], cell_0[1], cell_1[0], cell_1[1]) for (cell_0, cell_1) in steps]
        )
        fp.write(tow)


def get_bbox(cells):
    tmp = np.array(cells)
    return Bbox(*np.array([tmp.min(0), tmp.max(0) + 1]).flatten().tolist())


def search_morph_steps(ary, target, super_trials=1000, sub_trials=100, step=10):
    """
    Try to morph ``ary`` image into ``target`` one, just swapping pixels.
    """
    def steps2bbox(new_steps):
        if new_steps:
            return get_bbox(new_steps[-1])
        return Bbox(0, 0, target.shape[0] - 1, target.shape[1] - 1)

    # checksum
    source_sum = ary.sum()
    print('source_sum =', source_sum)
    print('target')
    print(target, target.dtype)
    print('array')
    print(ary, ary.dtype)

    best_diff = square_diff_sum(target, ary)
    print('initial diff =', best_diff)

    size = 1
    n_steps = 0
    new_steps = []
    t0 = get_time()
    try:
        while True:
            # start exploration from last step
            bbox = steps2bbox(new_steps)
            for i in range(super_trials):
                ary, new_steps, new_diff = explore(target, ary.copy(), best_diff, bbox, sub_trials)
                assert new_diff <= best_diff, '{} !<= {}'.format(new_diff, best_diff)
                best_diff = new_diff
                if new_steps:
                    #print('Successful exploration after {} trials'.format(i))
                    break

            if new_steps:
                n_steps += len(new_steps)
                save_steps(new_steps)
                if not n_steps % 10:
                    print('{:,} total steps saved'.format(n_steps))
                    save_image(ary, 'current_best.png')
                    print('Current best image updated')
                    print('{:.2f} steps / s'.format(n_steps / (get_time() - t0)))
                if best_diff == 0:
                    print('TOP')
                    break
            else:
                break
    except KeyboardInterrupt:
        print('Manual stop.')

    return ary


def apply_morphing_steps(ary, steps, n_frames=3600):
    frame_step = int(len(steps) / n_frames)
    yield ary.copy()
    j = 0
    for i, cells in enumerate(steps):
        j += 1
        print(i, cells)
        puzzle.swap(ary, 1, cells[0], cells[1])
        if j == group:
            yield ary.copy()
            j = 0


def save_animation(starter, steps_file):
    """
    Save a dict with keys 'source' and 'steps' in a file
    """
    cells = []
    with open(steps_file) as fp:
        for line in fp:
            cell_0_row, cell_0_col, cell_1_row, cell_1_col = tuple(map(int, line.split('\t')))
            cell_0 = cell_0_row, cell_0_col
            cell_1 = cell_1_row, cell_1_col
            cells.append((cell_0, cell_1))
    to_save = dict(starter=starter.tolist(), cells=cells)
    with open('anim_data.pkl', 'wb') as fp:
        pickle.dump(to_save, fp)



def main(target, source=None, dst='video.mp4'):
    try:
        os.remove('steps.csv')
    except FileNotFoundError:
        print('.')

    target = load_image_as_grayscale(target).astype(np.int32)
    print(target)
    print(target.dtype)
    save_image(target, 'target.png')

    if source is None:
        source = target.copy().flatten()
        source.sort()
        source = source.reshape(target.shape)
        n_swaps = np.prod(target.shape) * 2
        print('swapping target {} ({:,} swaps)...'.format(target.shape, n_swaps))
        rand_puzzle.random_puzzle(source, 1, n_swaps)
    else:
        source = load_image_as_grayscale(source).astype(np.int32)

    starter = source.copy()
    save_image(starter, 'starter.png')

    print('Finding steps...')
    ary = search_morph_steps(source.copy(), target.copy(), super_trials=1000, sub_trials=200)

    print('Saving things')
    save_image(ary, 'final_array.png')
    save_animation(starter, 'steps.csv')


if __name__ == '__main__':
    main(*sys.argv[1:])
