from .utils import io_tools
from .utils import path
import urllib.parse
import mimetypes
import os
import logging

log = logging.getLogger(__name__)

def load_dataset(dataset, category_depth=1, filter=None, samples=None, category_map=None, force_categories=False):
    """
    Load a dataset using the last n directories as category
    :param folder: top folder of dataset to walk
    :param category_depth: Depth of category extraction
    :param filter: Process files, return None to ignore
    :param samples: dict of existing samples to extend
    :return: samples dict with new samples on the format {cat1:[samples...], cat2:...]}
    """
    
    if samples is None:
        samples = {}

    category_map = {} if category_map is None else category_map

    for root, _, files in io_tools.walk(dataset["bucket"]):
        folders = path.get_folders(path.join(root, "dummy.ext"), dataset["bucket"])
        category = (folders if len(folders) else [None])[-1] if category_depth == 1 else "/".join(folders[-category_depth:])
        #category = category.lower() if isinstance(category, str) else category
        category = category_map.get(category, category)

        if (force_categories and category not in list(category_map.values())) or category is None:
            continue

        for file in files:
            full_path = path.join(root, file)
            sample = dict(path=full_path) if filter is None else filter(full_path, category)

            if sample is not None and mimetypes.guess_type(file)[0].startswith('image/'):
                sample["dataset"] = dataset['name']
                sample["bucket"] = dataset['bucket']
                sample["dataset_id"] = dataset['id']
                params = urllib.parse.urlencode({"path": path.join(root, file).replace("gs://",'')}, safe='')
                sample["url"] = "https://app.criterion.ai/download?{params}".format(params=params)
                samples.setdefault(category if isinstance(category, str) else "/".join(category), []).append(sample)
    return samples

def filter_file_by_ending(ftypes):
    """
    Filter function filtering by file ending
    :param ftypes: whitelist of filetypes
    :return:
    """

    def _filter(file_path, category):
        ftype = file_path.rsplit('.', 1)[-1]
        if ftype.lower() in ftypes:
            return dict(path=file_path, category=(category,) if isinstance(category, str) else tuple(category))
        else:
            return None

    return _filter


def load_image_datasets(datasets, class_map=None, force_categories=False):
    """
    Load a list of images from list of datasets
    :param datasets:
    :return:
    """
    datasets_ = {}

    for d in datasets:
        log.debug("Loading dataset %r" % d['id'])
        datasets_[d['id']] = load_dataset(d,
                                     filter=filter_file_by_ending({'bmp', 'jpeg', 'jpg', 'png'}),
                                     category_map=class_map,
                                     force_categories=force_categories)
        log.debug([(k, len(v)) for k, v in datasets_[d['id']].items()])

    return datasets_
