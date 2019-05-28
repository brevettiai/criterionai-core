import asyncio
import aiohttp
from gcloud.aio.storage import Storage
import aiofiles
import aiohttp
import threading
from threading import current_thread
from threading import Thread
import time
import sys
import multiprocessing
from google.cloud import storage
import google.auth

threadLocal = threading.local()

class mini_batch_downloader:
    def __init__(self, ftp_connections, id):
        self.ftp_connections = ftp_connections
        self.id = id

    def get_sessions(self):
        return getattr(threadLocal, 'sessions_{}'.format(self.id), {})
    
    def set_sessions(self, sessions):
        setattr(threadLocal, 'sessions_{}'.format(self.id), sessions)

    async def connect(self):
        self.disconnect()
        sessions = self.get_sessions()
        for kk, vv in self.ftp_connections.items():
            client = sessions[kk] = aioftp.Client(socket_timeout=60, path_timeout=60)
            for connection_attempt in range(10):
                try:
                    await client.connect(vv['host'])
                    await client.login(vv['user'], vv['password'])
                    ll = await client.list("/")
                    print("connection workding - first element: ", ll[0])
                except aioftp.errors.StatusCodeError:
                    pass
                else:
                    break
            if connection_attempt >= 9:
                print("Failed connecting", connection_attempt)
        print(self.id, threading.currentThread().getName(), "Connected in thread ", threading.currentThread().getName())

        self.set_sessions(sessions)
        return self

    def __del__(self, *err):
        self.disconnect()

    def disconnect(self):
        sessions = self.get_sessions()
        for client in sessions.values():
            client.close()
        self.set_sessions({})

    async def download(self, q):
        sessions = self.get_sessions()
        while True:
            t = {}
            t0 = time.time()
            ftp_id, path, outp = await q.get()
            t['name'] = path
            t["item"] = time.time()-t0
            if ftp_id is None:
                await q.put((ftp_id, path, outp))
                break
            for download_attempt in range(10):
                try:
                    async with aiofiles.open(outp, 'wb') as out_file, sessions[ftp_id].download_stream(path) as stream:
                        t["opened"] = time.time()-t0
                        async for block in stream.iter_by_block():
                            await out_file.write(block)
                except aioftp.errors.StatusCodeError:
                    print(self.id, threading.currentThread().getName(), "Falied, retrying")
                    pass
                except asyncio.TimeoutError:
                    print(self.id, threading.currentThread().getName(), "Download TimeOut")
                    raise
                except ConnectionResetError:
                    print(self.id, threading.currentThread().getName(), "Connection is reset - trying to reconnect", path)
                    await asyncio.sleep(10.0)
                    await self.connect()
                except:
                    print(self.id, threading.currentThread().getName(), "Unexpected error:", sys.exc_info()[0])
                    raise
                else:
                    break
            t["downloaded"] = time.time()-t0
            if download_attempt >= 9:
                print(self.id, threading.currentThread().getName(), "Failed download", download_attempt)

            q.task_done()
            t["done"] = time.time()-t0
            print(self.id, threading.currentThread().getName(), t)


async def queue_files(files, q):
    for ff in files:
        await q.put(ff)
 
    await q.put((None, None, None))

class batch_downloader():
    def __init__(self, ftp_connections, async_num):
        self.async_num = async_num
        #self.downloaders = [mini_batch_downloader(ftp_connections, ii) for ii in range(async_num)]
        print("Creating downloaders in thread ", threading.currentThread().getName())
    def get_loop(self):
        loop = getattr(threadLocal, 'loop', None)
        if loop is None:
            loop = asyncio.new_event_loop()
            threadLocal.loop = loop
            asyncio.set_event_loop(loop)
            [mbd.disconnect() for mbd in self.downloaders]
            loop.run_until_complete(asyncio.wait(tuple([mbd.connect() for mbd in self.downloaders])))
        return loop

    async def download_batch(self, files, loop):
        q = asyncio.Queue()
        fill_queue = asyncio.ensure_future(queue_files(files, q))
        self.credentials, self.project = google.auth.default()
        async with aiohttp.ClientSession() as session:
            storage = Storage(service_file=self.credentials, session=session)
            storage.download(bucket_name, object_name)
        download_files = [asyncio.ensure_future(self.downloaders[ii].download(q)) for ii in range(self.async_num)]
        await asyncio.gather(fill_queue, *download_files)

class threaded_downloader(multiprocessing.Process):
    def __init__(self, q, q_done, **kwargs):
        multiprocessing.Process.__init__(self)
        self.q = q
        self.q_done = q_done
        self.kwargs = kwargs
        self.start()

    def run(self):
        bd = batch_downloader(**self.kwargs)
        loop = bd.get_loop()
        while True:
            files = self.q.get()
            loop.run_until_complete(asyncio.wait((bd.download_batch([ff[:3] for ff in files], loop),)))
            self.q_done.put(files)
            self.q.task_done()


class download_f():
    def __init__(self, ftp_connections, async_num):
        self.async_num = async_num
        #self.downloaders = [mini_batch_downloader(ftp_connections, ii) for ii in range(async_num)]

        print("Creating downloaders in thread ", threading.currentThread().getName())
    def get_loop(self):
        loop = getattr(threadLocal, 'loop', None)
        if loop is None:
            loop = asyncio.new_event_loop()
            threadLocal.loop = loop
            asyncio.set_event_loop(loop)
            [mbd.disconnect() for mbd in self.downloaders]
            loop.run_until_complete(asyncio.wait(tuple([mbd.connect() for mbd in self.downloaders])))
        return loop

def download_batch(bucket_name, object_name):
    loop = asyncio.get_event_loop()
    out = loop.run_until_complete(asyncio.wait(async_download(bucket_name, object_name)))
    return out

async def async_download(bucket_name, object_name):
    credentials, project = google.auth.default()
    async with aiohttp.ClientSession() as session:
        storage = Storage(service_file=credentials, session=session)
        res = await storage.download(bucket_name, object_name)
    return res
