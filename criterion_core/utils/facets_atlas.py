import math
from criterion_core.utils.reshape_data import tile2d
from criterion_core.utils import path
from PIL import Image
import time
import numpy as np
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import sharedctypes
import json
import os
from tensorflow.python.lib.io import file_io


def fill_array(args):
    idx, sample, atlas_shape = args
    try:
        img = Image.open(sample["path"])
        img.thumbnail(atlas_shape[1:3])
        tmp = np.ctypeslib.as_array(shared_array).reshape(atlas_shape)
        tmp[idx] = img
    except Exception:
        pass

def create_atlas(samples, thumbnail_size=(64,64), channels=3):
    atlas_size = int(math.ceil(math.sqrt(len(samples))))
    if channels > 1:
        atlas_shape = len(samples), *thumbnail_size, channels
    else:
        atlas_shape = len(samples), *thumbnail_size

    # Build atlas
    shared_array = None

    def _init(a):
        global shared_array
        shared_array = a

    _array = np.ctypeslib.as_ctypes(np.zeros(atlas_shape, np.uint8).flatten())
    _array = sharedctypes.RawArray(_array._type_, _array)

    with Pool(initializer=_init, initargs=(_array,)) as pool:
        ret = pool.map(fill_array, [(i, x, atlas_shape) for i, x in enumerate(samples)], 10)
    atlas = np.ctypeslib.as_array(_array).reshape(atlas_shape)
    atlas = tile2d(atlas, (atlas_size, atlas_size))
    return atlas


def build_facets(samples, output_path, atlas_param=None):
    atlas_param = atlas_param or {}

    with file_io.FileIO(path.join(output_path, 'facets.json'), 'w') as fp:
        json.dump(list(samples), fp)

    atlas = create_atlas(samples, **atlas_param)
    Image.fromarray(atlas).save(path.join(output_path, 'spriteatlas.jpeg'))


if __name__ == '__main__':
    from criterion_core import load_image_datasets
    from criterion_core.utils import sampletools
    output_path = r'C:\data\novo\jobdirs\image_classification35'

    datasets = [
        dict(bucket=r'C:\data\novo\data\22df6831-5de9-4545-af17-cdfe2e8b2049.datasets.criterion.ai',
             id='22df6831-5de9-4545-af17-cdfe2e8b2049',
             name='test')
    ]

    dssamples = load_image_datasets(datasets)

    samples = list(sampletools.flatten_dataset(dssamples))

    build_facets(samples, output_path)
