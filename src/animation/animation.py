"""
Animation example.
"""
import os

# 3rd party
import numpy as np
from imageio import mimwrite

from puzzle.util import save_image


def save_frames_images(frames, dst_dir='.'):
    """
    Save frames into file (used?).
    """
    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass
    old_frame = None
    for i, frame in enumerate(frames):
        if old_frame is not None:
            assert not np.array_equal(frame, old_frame), 'frame {} == frame {}'.format(i, i - 1)
        old_frame = frame.copy()
        save_image(frame, os.path.join(dst_dir, 'frame_{:04d}.jpg'.format(i)))
    print('{:,} frames saved'.format(len(frames)))


def build_animation(frames, dst):
    """
    Combine images into a gif or video, depending on ``dst`` extension.

    NB: video needs ffmpeg.
    """
    mimwrite(dst, frames)


FRAMES = []

for _ in range(100):
    FRAMES.append(np.random.randint(0, 255, (50, 50), np.uint8))

build_animation(FRAMES, 'output.gif')
