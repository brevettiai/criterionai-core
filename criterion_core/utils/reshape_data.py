from itertools import repeat, chain, islice
import numpy as np


def tile2d(x, grid=(10, 10)):
    """
    Function to tile numpy array to plane, eg. images along the first axis
    :param x: numpy array to tile
    :param grid: size of the grid in tiles
    :return: tiled
    """
    generator = chain((x[i] for i in range(len(x))), repeat(np.zeros_like(x[0])))
    return np.vstack([np.hstack(list(islice(generator, grid[0]))) for _ in range(grid[1])])
