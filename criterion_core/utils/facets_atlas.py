import math
from criterion_core.utils.reshape_data import tile2d
from criterion_core.utils import path
from criterion_core.utils import data_generator

import numpy as np
import json
import os
from . import io_tools
import cv2


def create_atlas(facet_gen):
    images = [None] * len(facet_gen)
    for ii in range(len(facet_gen)):
        images[ii], y = facet_gen[ii]
    images = np.squeeze(np.concatenate(images)*255).astype(np.uint8)
    atlas_size = int(math.ceil(math.sqrt(len(images))))
    atlas = tile2d(images, (atlas_size, atlas_size))
    return atlas


def build_facets(facet_gen, output_path, facet_key='', **atlas_param):
    facet_dive = 'facets{}.json'.format(facet_key)
    facet_sprite = 'spriteatlas{}.jpeg'.format(facet_key)

    io_tools.write_file(path.join(output_path, facet_dive), json.dumps(list(facet_gen.samples)))
    atlas = create_atlas(facet_gen)
    atlas = cv2.cvtColor(atlas, cv2.COLOR_BGR2RGB)
    jpeg_created, buffer = cv2.imencode(".jpeg", atlas)
    assert jpeg_created
    io_tools.write_file(path.join(output_path, facet_sprite), bytes(buffer))
    return facet_dive, facet_sprite

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

    build_facets(samples, output_path, thumbnail_size=(64, 64), color_mode="greyscale")
