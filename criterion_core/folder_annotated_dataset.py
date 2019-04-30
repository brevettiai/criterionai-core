import os
from fs import open_fs
from functools import reduce
import asyncio
import aioftp
import mimetypes
import logging

logging.getLogger('aioftp').setLevel(logging.WARNING)


def partition(items, func):
    return reduce(lambda x, y: x[not func(y)].append(y) or x, items, ([], []))

def walk(top):
    items = file_io.list_directory_v2(top)
    folders, files = partition(items, lambda x: x.endswith("/"))
    yield top, folders, files
    for folder in folders:
        yield from walk(os.path.join(top, folder))

def walk_ftp(host, user, password):
    return list(asyncio.get_event_loop().run_until_complete(asyncio.wait((_walk_ftp(host, user, password), )))[0])[0].result()

async def _walk_ftp(host, user, password, port=21):
    files = []
    async with aioftp.ClientSession(host, port, user, password) as client:
        for path, info in (await client.list(recursive=True)):
            if info["type"] == "file":
                files.append(('/'.join(path.parts[:-1]), path))
    return files

async def get_mp3(host, login, password, port=21):
    files = []
    async with aioftp.ClientSession(host, port, login, password) as client:
        for path, info in (await client.list(recursive=True)):
            if info["type"] == "file":
                print("Downloading: ", path)
                #await client.download(path, "test.heic", write_into=True)
                files.append(path)
    return files

def load_dataset(dataset, name=None, category_depth=1, filter=None, samples=None, category_map=None, force_categories=False):
    """
    Load a dataset using the last n directories as category
    :param folder: top folder of dataset to walk
    :param category_depth: Depth of category extraction
    :param filter: Process files, return None to ignore
    :param samples: dict of existing samples to extend
    :return: samples dict with new samples on the format {cat1:[samples...], cat2:...]}
    """
    #dataset["bucket"]
    user, password = dataset['bucket'].lstrip('ftp://').split('@')[0].split(':')
    connect_info = {'host': "ftp.criterion.ai", 'user': user, 'password': password}
    
    file_paths = walk_ftp(connect_info['host'], user, password)
    if samples is None:
        samples = {}

    category_map = {} if category_map is None else category_map

    for root, file_name in file_paths:
        files = [file_name]
        category = root.rsplit('/', category_depth)
        category = category[-1] if category_depth == 1 else category[1:]

        category = category_map.get(category, category)
        if force_categories and category not in category_map:
            continue

        for file in files:
            path = '/'.join([root, file.name])
            sample = dict(path=path) if filter is None else filter(path, category)

            if not sample is None and len(category)>0 and mimetypes.guess_type(file.name)[0].startswith('image/'):
                sample["dataset"] = dataset['name']
                sample["id"] = dataset['id']
                samples.setdefault(category, []).append(sample)
    return samples, connect_info

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
    connections_ = {}

    for d in datasets:
        datasets_[d['id']], connections_[d['id']] = load_dataset(d,
                                      d['name'],
                                     filter=filter_file_by_ending({'bmp', 'jpeg', 'jpg', 'png'}),
                                     category_map=class_map)

    return datasets_, connections_
