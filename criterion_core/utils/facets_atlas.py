import math
from criterion_core.utils.reshape_data import tile2d
from criterion_core.utils import path
from PIL import Image
import time
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import sharedctypes
import json
import os
from . import io_tools
import cv2

def fill_array(args):
    idx, sample, atlas_shape = args
    try:
        buffer = io_tools.read_file(sample["path"])
        cv2_img = cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)
        img = Image.fromarray(cv2_img, 'RGB' if cv2_img.ndim==3 else 'L')
        img.thumbnail(atlas_shape[1:3])
        tmp = np.ctypeslib.as_array(shared_array).reshape(atlas_shape)
        tmp[idx] = img
    except Exception:
        print(Exception)
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

    io_tools.write_file(path.join(output_path, 'facets.json'), json.dumps(list(samples)))

    atlas = create_atlas(samples, **atlas_param)
    jpeg_created, buffer = cv2.imencode(".jpeg", atlas)
    assert jpeg_created
    io_tools.write_file(path.join(output_path, 'spriteatlas.jpeg'), bytes(buffer))
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
