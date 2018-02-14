"""
Just import things.
"""

# Standard Library
import sys

# 3rd party
import numpy as np

# My stuff
from samples import sample_module

if __name__ == '__main__':
    print('numpy', np.version.full_version)
    print(np.__file__)
    try:
        print(sample_module.reverse_manually(sys.argv[1]))
    except IndexError:
        print('No args?')
