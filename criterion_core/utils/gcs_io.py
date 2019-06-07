import asyncio
import aiohttp
from gcloud.aio.storage import Storage
import google.auth
from google.cloud import storage
from urllib.parse import unquote
import threading
import logging
import backoff

logging.getLogger("asyncio").setLevel(level=logging.INFO)
logging.getLogger("urllib3").setLevel(level=logging.INFO)
logging.getLogger("google.auth.transport.requests").setLevel(level=logging.INFO)

threadLocal = threading.local()


def get_loop():
    loop = getattr(threadLocal, 'loop', None)
    if threading.current_thread() is threading.main_thread():
        try:
            loop = asyncio.get_event_loop()
        except:
            pass

    if loop is None:
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
    return ((unquote(output.public_url).replace("https://storage.googleapis.com/", "gs://").rsplit("/", 1)[0], '', [output.name.rsplit("/", 1)[-1]]) for output in bucket.list_blobs(prefix=prefix))


@backoff.on_exception(backoff.expo, aiohttp.ClientResponseError, max_tries=5)  # type: ignore
async def download_file(file_path, st):
    bucket_name, blob = _get_gcs_from_path(file_path)
    byte_buffer = await st.download(bucket_name, blob, timeout=100)
    return byte_buffer


async def async_download_batch(img_files_batch):
    conn = aiohttp.TCPConnector(limit_per_host=32)
    async with aiohttp.ClientSession(connector=conn) as session:
        st = Storage(session=session)
        futures = [download_file(blob['path'], st) for blob in img_files_batch]
        buffers = await asyncio.gather(*futures)
    return buffers


def download_batch(img_files_batch):
    loop = get_loop()
    buffers = list(loop.run_until_complete(asyncio.wait((async_download_batch(img_files_batch), ), timeout=300))[0])[0].result()
    return buffers
