import os
import asyncio
import aiofiles
import threading
from threading import current_thread
from threading import Thread
import time
import sys
import multiprocessing
from . import gcs_io, path


def read_file(file_path):
    if "gs://" in file_path:
        return gcs_io.gcs_read(file_path)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as fp:
            output = fp.read()
        return output


def write_file(file_path, content):
    if "gs://" in file_path:
        return gcs_io.gcs_write(file_path, content)

    if os.path.exists(os.path.dirname(os.path.abspath(file_path))):
        if isinstance(content, (bytes, bytearray)):
            with open(file_path, 'wb') as fp:
                output = fp.write(content)
        else:
            with open(file_path, 'w') as fp:
                output = fp.write(content)
        return output


def make_dirs(folder_path):
    if "gs://" in folder_path:
        return gcs_io.gcs_write(path.join(folder_path, ''), '')

    return os.makedirs(folder_path, exist_ok=True)


def walk(folder_path):
    if "gs://" in folder_path:
        return gcs_io.gcs_walk(folder_path)

    if os.path.exists(folder_path):
        return os.walk(folder_path)


async def _async_read_file(fn):
    async with aiofiles.open(fn, mode='rb') as stream:
        output = await stream.read()
    return output


async def _async_download_batch(img_files_batch):
    futures = [_async_read_file(blob["path"]) for blob in img_files_batch]
    return await asyncio.gather(*futures)


def aio_download_batch(img_files_batch):
    loop = gcs_io.get_loop()
    buffers = list(loop.run_until_complete(asyncio.wait((_async_download_batch(img_files_batch), )))[0])[0].result()
    return buffers


def download_batch(img_files_batch):
    if "gs://" in img_files_batch[0]["path"]:
        return gcs_io.download_batch(img_files_batch)

    return aio_download_batch(img_files_batch)