"""
Morph a source image in a target one, just swapping pixels.

The final result is not guaranteed to be equal to the target
but similar due to immutable pixel values.
"""
import os
import pickle
import sys
from collections import namedtuple
from random import choice
from time import time as get_time

import numpy as np

from animation import animation
from . import puzzle
from . import rand_puzzle
from .util import load_image_as_grayscale
from .util import save_image


Bbox = namedtuple('bbox', ['min_row', 'min_col', 'max_row', 'max_col'])
DIRECTIONS = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if (x, y) != (0, 0)]


def square_diff_sum(ary_1, ary_2):
    """
    Represent a difference, distance, between 2 image arrays.
    """
    return ((ary_1 - ary_2) ** 2).sum()


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
    """
    Try to find a long sequence of swap which reduce the
    difference from `ary` to `target`.
    """
    def expand_bbox(bbox):
        """Given a bbox, expands it by 1, top, left, right and bottom."""
        height, width = ary.shape
        return Bbox(
            min_row=max(0, bbox.min_row - 1),
            min_col=max(0, bbox.min_col - 1),
            max_row=min(height, bbox.max_row + 1),
            max_col=min(width, bbox.max_col + 1),
        )
    swap = puzzle.swap_size_one

    best_ary = ary.copy()
    trials_steps = []
    trials_steps_pointer = 0
    diff = best_diff

    direction = choice(DIRECTIONS)
    cell_0, cell_1 = rand_swap_cells(ary, bbox)

    for _ in range(sub_trials):
        try:
            swap(ary, cell_0, cell_1)
        except IndexError:
            break
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
        cell_0 = cell_1
        direction = choice(DIRECTIONS)
        cell_1 = (cell_0[0] + direction[0], cell_0[1] + direction[1])
        if min(cell_1) < 0:
            break
        try:
            ary[cell_1]
        except IndexError:
            break

    return best_ary, trials_steps[:trials_steps_pointer], best_diff


def save_steps(steps):
    """
    Persist steps in a csv in append mode.

    Useful to quickly save a bunch of steps every tot of time.
    """
    with open('steps.csv', 'a') as file:
        tow = ''.join([
            '{}\t{}\t{}\t{}\n'.format(cell_0[0], cell_0[1], cell_1[0], cell_1[1])
            for (cell_0, cell_1) in steps
        ])
        file.write(tow)


def get_bbox(cells):
    """
    Calculate the bounding box of given ``cells``.
    """
    tmp = np.array(cells)
    return Bbox(*np.array([tmp.min(0), tmp.max(0) + 1]).flatten().tolist())


def search_morph_steps(ary, target, super_trials=1000, sub_trials=100):
    """
    Try to morph ``ary`` image into ``target`` one, just swapping pixels.
    """
    def steps2bbox(new_steps):
        """
        Obtain the starting bbox from last ``new_steps`` (i.e. last cell).
        """
        if new_steps:
            return get_bbox(new_steps[-1])
        return Bbox(0, 0, target.shape[0] - 1, target.shape[1] - 1)

    # checksum
    source_sum = ary.sum()
    print('source_sum =', source_sum)
    best_diff = square_diff_sum(target, ary)
    print('initial diff =', best_diff)

    n_steps = 0
    new_steps = []
    t_0 = get_time()
    try:
        while True:
            # start exploration from last step
            bbox = steps2bbox(new_steps)
            for _ in range(super_trials):
                ary, new_steps, new_diff = explore(target, ary.copy(), best_diff, bbox, sub_trials)
                assert new_diff <= best_diff, '{} !<= {}'.format(new_diff, best_diff)
                best_diff = new_diff
                if new_steps:
                    break

            if new_steps:
                n_steps += len(new_steps)
                save_steps(new_steps)
                if not n_steps % 10:
                    print('{:,} total steps saved'.format(n_steps))
                    save_image(ary, 'current_best.png')
                    print('Current best image updated')
                    print('{:.2f} steps / s'.format(n_steps / (get_time() - t_0)))
                if best_diff == 0:
                    print('TOP')
                    break
            else:
                break
    except KeyboardInterrupt:
        print('Manual stop.')

    return ary


def save_animation(starter, steps_file, file_path='anim_data.pkl'):
    """
    Save a dict with keys 'source' and 'steps' in a file
    """
    cells = []
    with open(steps_file) as file:
        for line in file:
            cell_0_row, cell_0_col, cell_1_row, cell_1_col = tuple(map(int, line.split('\t')))
            cell_0 = cell_0_row, cell_0_col
            cell_1 = cell_1_row, cell_1_col
            cells.append((cell_0, cell_1))
    to_save = dict(starter=starter.tolist(), cells=cells)
    with open(file_path, 'wb') as file:
        pickle.dump(to_save, file)


def build_video(file_path, video_file_path='video.mov'):
    """
    Create a video from ``file_path`` stored steps.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    ary = np.array(data['starter'], np.uint8)
    n_steps = len(data['cells'])
    frame_step = int(n_steps / 600)
    frames = []
    for i, (cell_0, cell_1) in enumerate(data['cells']):
        puzzle.swap_size_one(ary, cell_0, cell_1)
        if not i % frame_step:
            frames.append(ary.copy())
    print('{:,} steps, frame step = {:,}, frames = {:,}'.format(n_steps, frame_step, len(frames)))
    animation.build_animation(frames, video_file_path)


def main(target, source=None):
    """
    Try to draw ``target`` by swaps, starting from ``source``
    or from a random image.
    """
    _, file_name = os.path.split(target)
    name, _ = os.path.splitext(file_name)

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
    ary = search_morph_steps(source.copy(), target.copy(), super_trials=20000, sub_trials=10000)

    print('Saving things')
    save_image(ary, 'final_array.png')
    anim_data_file_name = name + 'anim_data.pkl'
    save_animation(starter, 'steps.csv', anim_data_file_name)
    build_video(anim_data_file_name, name + '.mov')


if __name__ == '__main__':
    SOURCE = None
    ARGS = sys.argv[1:]
    if len(ARGS) == 2:
        TARGET, SOURCE = ARGS[0], ARGS[1]
    elif len(ARGS) == 1:
        TARGET = ARGS[0]
    else:
        raise Exception('Invalid arguments. Usage: python {} TARGET [SOURCE]'.format(__file__))
    main(TARGET, SOURCE)
