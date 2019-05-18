import os
import asyncio
import aiohttp
from gcloud.aio.storage import Storage
from . import path
import threading

threadLocal = threading.local()


def get_loop():
    loop = getattr(threadLocal, 'loop', None)
    if loop is None:
        loop = asyncio.new_event_loop()
        threadLocal.loop = loop
        asyncio.set_event_loop(loop)
    return loop


def _gcs_operation(bucket_name, operation_name, service_file: str, *args, **kwargs):
    async def async_operation():
        conn = aiohttp.TCPConnector(limit_per_host=30)
        async with aiohttp.ClientSession(connector=conn) as session:
            st = Storage(service_file=service_file, session=session)
            gcs_method = getattr(st, operation_name)
            async_output = await gcs_method(bucket_name, *args, **kwargs)
            return async_output
    loop = get_loop()
    output = list(loop.run_until_complete(asyncio.wait((async_operation(), )))[0])[0].result()
    return output


def _get_gcs_from_path(blob_path):
    assert "gs://" in blob_path
    bucket_name, sep_, blob = blob_path.lstrip("gs://").partition('/')
    return bucket_name, blob


def gcs_write(blob_path, content,
              service_file: str=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')):
    bucket_name, blob  = _get_gcs_from_path(blob_path)
    _gcs_operation(bucket_name, "upload", service_file, object_name=blob, file_data=content)


def gcs_read(blob_path, params=None, service_file: str=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')):
    bucket_name, blob  = _get_gcs_from_path(blob_path)
    return _gcs_operation(bucket_name, "download", service_file, object_name=blob, params=params)


def gcs_walk(folder_path, content_filter: str = "image",
             service_file: str = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')):
    bucket_name, blob = _get_gcs_from_path(folder_path)
    assert (len(blob) == 0), "Only walk bucket root is support: use parameter content_filter for filtering instead"
    outputs = []
    params = None
    while True:
        blobs = _gcs_operation(bucket_name, "list_objects", service_file, params=params)
        outputs.append([blob for blob in blobs['items'] if content_filter in blob.get("contentType", '')])
        if "nextPageToken" in blobs:
            params = dict(pageToken=blobs["nextPageToken"])
            continue
        else:
            break
    yield from [(path.join("gs://", bucket_name, ff[0]), '', [ff[1]]) for ff in [ff["name"].rsplit("/", 1) for output in outputs for ff in output]]


async def download_file(file_path, st):
    bucket_name, blob = _get_gcs_from_path(file_path)
    for retries in range(10):
        try:
            byte_buffer = await st.download(bucket_name, blob, timeout=100)
            return byte_buffer
        except aiohttp.ClientResponseError:
            pass
    raise aiohttp.ClientResponseError


async def async_download_batch(img_files_batch, service_file: str = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')):
    conn = aiohttp.TCPConnector(limit_per_host=32)
    async with aiohttp.ClientSession(connector=conn) as session:
        st = Storage(service_file=service_file, session=session)
        futures = [download_file(blob['path'], st) for blob in img_files_batch]
        buffers = await asyncio.gather(*futures)
    return buffers


def download_batch(img_files_batch):
    loop = get_loop()
    buffers = list(loop.run_until_complete(asyncio.wait((async_download_batch(img_files_batch), )))[0])[0].result()
    return buffers, [blob['category'] for blob in img_files_batch]