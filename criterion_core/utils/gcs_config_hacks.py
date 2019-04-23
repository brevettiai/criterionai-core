import os
from tensorflow.python.lib.io import file_io
import builtins
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

old_open = builtins.open

# NB: not all modes are compatible; should handle more carefully.
# Probably should be reported on
# https://github.com/tensorflow/tensorflow/issues/4357
def new_open(name, mode='r', buffering=-1, *args, **kwargs):
    #if not args or not kwargs:
    #    old_open(name, mode, buffering, *args, **kwargs)
    if False and 'U' in mode:
        file_io.recursive_create_dir('tmp')
        file_io.copy(name, os.path.join('tmp', os.path.split(name)[-1]), overwrite=True)
        return old_open(name, mode, buffering)
    else:
        return file_io.FileIO(name, mode.replace('U', ''))

builtins.open = new_open
