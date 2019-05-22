import os
import shutil
from tensorflow.python.lib.io import file_io
from functools import reduce


def movedir(srcdir, targetdir):
    """
    Move contents of directory on linux and windows file systems
    :param srcdir:
    :param targetdir:
    :return:
    """
    for file in os.listdir(srcdir):
        shutil.move(os.path.join(srcdir, file), os.path.join(targetdir, file))


def join(*paths):
    """
    Join os paths and urls
    :param paths:
    :return:
    """
    sp = paths[0].split('://', 1)
    if len(sp) == 2:
        # Found protocol
        pathsep = '/'
        paths = list(paths)
        for i, p in enumerate(paths[:-1]):
            if p[-1] == pathsep:
                paths[i] = p[:-1]
        return pathsep.join(paths)
    else:
        return os.path.join(*paths)


def partition(items, func):
    """
    Partition a list based on a function
    :param items: List of items
    :param func: Function to partition after
    :return:
    """
    return reduce(lambda x, y: x[not func(y)].append(y) or x, items, ([], []))


def gswalk(top):
    """
    Walk on google cloud storage
    :param top: Root directory
    :return:
    """
    items = file_io.list_directory_v2(top)
    folders, files = partition(items, lambda x: x.endswith("/"))
    yield top, folders, files
    for folder in folders:
        yield from gswalk(os.path.join(top, folder))


def walk(top):
    """
    walk local and gcs folders
    :param top: Root directory
    :return:
    """
    if top.startswith("gs://"):
        yield from gswalk(top)
    else:
        yield from os.walk(top)
