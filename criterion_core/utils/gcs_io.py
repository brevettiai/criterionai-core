import os
import queue
import multiprocessing
import asyncio

import aiohttp
from gcloud.aio.storage import Storage
from tensorflow.python.lib.io import file_io


def gcs_operation(bucket_name, operation_name, service_file: str, *args, **kwargs):
    bucket_name = bucket_name.replace('gs://', '').replace('/', '')
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
    def __init__(self, service_file="urlsigner.json", max_queue_size=10):
        multiprocessing.Process.__init__(self)
        self.service_file = service_file
        self.q = multiprocessing.JoinableQueue()
        self.q_downloaded = multiprocessing.JoinableQueue(maxsize=max_queue_size)
        self.loop = None
        self.daemon = True
        self.start()


    def update_queue(self, img_files, batch_size, num_batches, indices):
        self.reset_queues()
        for index in range(num_batches):
            # Generate indexes of the batch
            batch_indices = indices[index*batch_size:(index+1)*batch_size]
            # Find list of IDs
            img_files_batch = [img_files[k] for k in batch_indices]
            self.q.put(img_files_batch)

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
            while True:
                blobs = self.q.get()
                futures = [self.download_file(blob['bucket'], blob['path'], st) for blob in blobs]
                buffers = await asyncio.gather(*futures)
                self.q_downloaded.put((buffers, [blob['category'] for blob in blobs]))
                self.q.task_done()
                self.q.put(blobs) # Adding batch to queue again, if multiple querys on "same" batch

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
                self.q.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self.q_downloaded.get_nowait()
            except queue.Empty:
                break

    def cancel(self):
        self.reset_queues()
        self.q.cancel_join_thread()
        self.q_downloaded.cancel_join_thread()

    def run(self):
        self.loop.run_until_complete(asyncio.wait((self.download_loop(),)))
