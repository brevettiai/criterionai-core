import os
from fs import open_fs
from functools import reduce

def partition(items, func):
    return reduce(lambda x, y: x[not func(y)].append(y) or x, items, ([], []))


def walk(top):
    items = file_io.list_directory_v2(top)
    folders, files = partition(items, lambda x: x.endswith("/"))
    yield top, folders, files
    for folder in folders:
        yield from walk(os.path.join(top, folder))


def load_dataset(dataset, name=None, category_depth=1, filter=None, samples=None, category_map=None, force_categories=False):
    """
    Load a dataset using the last n directories as category
    :param folder: top folder of dataset to walk
    :param category_depth: Depth of category extraction
    :param filter: Process files, return None to ignore
    :param samples: dict of existing samples to extend
    :return: samples dict with new samples on the format {cat1:[samples...], cat2:...]}
    """
    
    file_sys = open_fs(dataset["path"])

    if samples is None:
        samples = {}

    category_map = {} if category_map is None else category_map

    for root, _, files in file_sys.walk():
        category = root.rsplit('/', category_depth)
        category = category[-1] if category_depth == 1 else category[1:]

        category = category_map.get(category, category)
        if force_categories and category not in category_map:
            continue

        for file in files:
            path = '/'.join([root, file.name])
            sample = dict(path=path) if filter is None else filter(path, category)

            if not sample is None:
                sample["dataset"] = dataset['name']
                sample["file_sys"] = file_sys
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
                                     category_map=class_map)

    return datasets_
