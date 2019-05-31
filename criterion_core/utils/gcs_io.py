import os
import asyncio
import aiohttp
from gcloud.aio.storage import Storage
import google.auth
from google.cloud import storage
from . import path
import threading
import logging
import datetime

logging.getLogger("asyncio").setLevel(level=logging.INFO)
logging.getLogger("urllib3").setLevel(level=logging.INFO)
logging.getLogger("google.auth.transport.requests").setLevel(level=logging.INFO)

threadLocal = threading.local()


def get_signed_url(bucket_id, blob_path, expiration=datetime.datetime.utcnow() + datetime.timedelta(days=365)):
    credentials, project = google.auth.default()
    storage_client = storage.Client(project, credentials)
    auth_request = requests.Request()
    signing_credentials = storage_client._credentials
    bucket = storage_client.get_bucket(bucket_id)
    blob = bucket.blob(blob_path)
    signed_url = blob.generate_signed_url(expiration=expiration, method='GET', credentials=signing_credentials,
                                          version="v2")
    return signed_url


def get_loop():
    if threading.current_thread() is threading.main_thread():
        loop = asyncio.get_event_loop()
    else:
        loop = getattr(threadLocal, 'loop', None)

    if loop is None:
        print(threading.current_thread())
        loop = asyncio.new_event_loop()
        threadLocal.loop = loop
        asyncio.set_event_loop(loop)
    return loop


def _get_gcs_from_path(blob_path):
    assert "gs://" in blob_path
    bucket_name, sep_, blob = blob_path.lstrip("gs://").partition('/')
    return bucket_name, blob


def gcs_write(blob_path, content):
    credentials, project = google.auth.default()
    storage_client = storage.Client(project, credentials)
    bucket_name, blob_name  = _get_gcs_from_path(blob_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name=blob_name)
    return blob.upload_from_string(data=content)


def gcs_read(blob_path):
    credentials, project = google.auth.default()
    storage_client = storage.Client(project, credentials)
    bucket_name, blob_name  = _get_gcs_from_path(blob_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name=blob_name)
    return blob.download_as_string()


def gcs_walk(blob_path):
    credentials, project = google.auth.default()
    storage_client = storage.Client(project, credentials)
    bucket_name, prefix  = _get_gcs_from_path(blob_path)
    bucket = storage_client.get_bucket(bucket_name)
    yield from [(path.join("gs://", bucket_name, output.name.rsplit("/", 1)[0]), '', [output.name.rsplit("/", 1)[1]]) for output in bucket.list_blobs(prefix=prefix)]


async def download_file(file_path, st):
    bucket_name, blob = _get_gcs_from_path(file_path)
    for retries in range(10):
        try:
            byte_buffer = await st.download(bucket_name, blob, timeout=100)
            return byte_buffer
        except aiohttp.ClientResponseError:
            pass
    raise aiohttp.ClientResponseError


async def async_download_batch(img_files_batch):
    conn = aiohttp.TCPConnector(limit_per_host=32)
    async with aiohttp.ClientSession(connector=conn) as session:
        st = Storage(session=session)
        futures = [download_file(blob['path'], st) for blob in img_files_batch]
        buffers = await asyncio.gather(*futures)
    return buffers


def download_batch(img_files_batch):
    loop = get_loop()
    buffers = list(loop.run_until_complete(asyncio.wait((async_download_batch(img_files_batch), )))[0])[0].result()
    return buffers
