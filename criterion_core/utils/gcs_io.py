import os
import queue
import multiprocessing
import asyncio

import aiohttp
from gcloud.aio.storage import Storage
from tensorflow.python.lib.io import file_io


def gcs_operation(bucket_name, operation_name, service_file: str, *args, **kwargs):
    bucket_name = bucket_name.replace('gs://', '')
    if not os.path.exists(service_file):
        file_io.copy("gs://security.criterion.ai/urlsigner.json", service_file)

    async def async_operation():
        conn = aiohttp.TCPConnector(limit_per_host=30)
        async with aiohttp.ClientSession(connector=conn) as session:
            st = Storage(service_file=service_file, session=session)
            gcs_method = getattr(st, operation_name)
            async_output = await gcs_method(bucket_name, *args, **kwargs)
            return async_output
    loop = asyncio.get_event_loop()
    output = list(loop.run_until_complete(asyncio.wait((async_operation(), )))[0])[0].result()
    return output


def gcs_write(bucket_name, blob, content,
              service_file: str = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'urlsigner.json')):
    gcs_operation(bucket_name, "upload", service_file, object_name=blob, file_data=content)


def gcs_read(bucket_name, blob,
             service_file: str = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'urlsigner.json')):
    return gcs_operation(bucket_name, "download", service_file, object_name=blob)


def gcs_walk(bucket_name, content_filter: str = "image",
             service_file: str = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'urlsigner.json')):
    blobs = gcs_operation(bucket_name, "list_objects", service_file)
    output = [blob for blob in blobs['items'] if content_filter in blob.get("contentType", '')]
    return output


def walk(bucket_name, content_filter: str = "image",
             service_file: str = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'urlsigner.json')):
    files = gcs_walk(bucket_name, content_filter, service_file)
    yield from [(ff[0], '', [ff[1]]) for ff in [ff["name"].rsplit("/", 1) for ff in files]]


class GcsBatchDownloader(multiprocessing.Process):
    def __init__(self, q_in, q_done, service_file="urlsigner.json"):
        multiprocessing.Process.__init__(self)
        if not os.path.exists(service_file):
            file_io.copy("gs://security.criterion.ai/urlsigner.json", service_file)
        self.service_file = service_file
        self.q_in = q_in
        self.q_done = q_done
        self.loop = None
        self.start()

    @staticmethod
    async def download_file(bucket_name, blob, st):
        for retries in range(10):
            try:
                byte_buffer = await st.download(bucket_name, blob, timeout=100)
                return byte_buffer
            except aiohttp.ClientResponseError:
                pass
        raise aiohttp.ClientResponseError

    async def download_loop(self):
        conn = aiohttp.TCPConnector(limit_per_host=30)
        async with aiohttp.ClientSession(connector=conn) as session:
            st = Storage(service_file=self.service_file, session=session)
            ii = 0
            while True:
                ii += 1
                blobs = self.q_in.get()
                futures = [self.download_file(blob['bucket'], blob['path'], st) for blob in blobs]
                buffers = await asyncio.gather(*futures)
                self.q_done.put((buffers, [blob['category'] for blob in blobs]))
                self.q_in.task_done()
                self.q_in.put(blobs) # Adding batch to queue again, if multiple querys on "same" batch

    @property
    def loop(self):
        return self._loop

    @loop.setter
    def loop(self, value):
        self._loop = value

    @loop.getter
    def loop(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def reset_queues(self):
        while True:
            try:
                self.q_in.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self.q_done.get_nowait()
            except queue.Empty:
                break

    def run(self):
        self.loop.run_until_complete(asyncio.wait((self.download_loop(),)))