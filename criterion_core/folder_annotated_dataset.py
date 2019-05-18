from .utils import io_tools
import mimetypes
import os

def load_dataset(dataset, name=None, category_depth=1, filter=None, samples=None, category_map=None, force_categories=False):
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

    sep_ = "/" if "://" in dataset["bucket"] else os.path.sep

    for root, _, files in io_tools.walk(dataset["bucket"]):
        dir_ = root[:-1] if root.endswith(sep_) else root
        category = dir_.rsplit(sep_, category_depth)

        category = category[-1] if category_depth == 1 else category[1:]

        category = category_map.get(category, category)
        if force_categories and category not in list(category_map.values()):
            continue

        for file in files:
            path = sep_.join([root, file])
            sample = dict(path=path) if filter is None else filter(path, category)

            if sample is not None and len(category)>0 and mimetypes.guess_type(file)[0].startswith('image/'):
                sample["dataset"] = dataset['name']
                sample["id"] = dataset['id']
                samples.setdefault(category, []).append(sample)
    return samples

def filter_file_by_ending(ftypes):
    """
    Filter function filtering by file ending
    :param ftypes: whitelist of filetypes
    :return:
    """

    def _filter(path, category):
        ftype = path.rsplit('.', 1)[-1]
        if ftype.lower() in ftypes:
            return dict(path=path, category=category)
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
        datasets_[d['id']] = load_dataset(d,
                                      d['name'],
                                     filter=filter_file_by_ending({'bmp', 'jpeg', 'jpg', 'png'}),
                                     category_map=class_map,
                                     force_categories=force_categories)

    return datasets_
