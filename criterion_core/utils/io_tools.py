import asyncio
import aioftp
import aiofiles
import threading
from threading import current_thread

threadLocal = threading.local()

class mini_batch_downloader:
    def __init__(self, ftp_connections, id):
        self.ftp_connections = ftp_connections
        self.id = id

    def get_sessions(self):
        return getattr(threadLocal, f'sessions_{self.id}', {})
    
    def set_sessions(self, sessions):
        setattr(threadLocal, f'sessions_{self.id}', sessions)

    async def connect(self):
        self.disconnect()
        sessions = self.get_sessions()
        for kk, vv in self.ftp_connections.items():
            client = sessions[kk] = aioftp.Client(socket_timeout=60, path_timeout=60)
            for connection_attempt in range(10):
                try:
                    await client.connect(vv['host'])
                    await client.login(vv['user'], vv['password'])
                except aioftp.errors.StatusCodeError:
                    pass
                else:
                    break            

        self.set_sessions(sessions)
        return self

    def __del__(self, *err):
        self.disconnect()

    def disconnect(self):
        sessions = self.get_sessions()
        for client in sessions.values():
            client.close()
        self.set_sessions({})

    async def download(self, files):
        sessions = self.get_sessions()
        for ftp_id, path, outp in files:
            for download_attempt in range(10):
                try:
                    async with aiofiles.open(outp, 'wb') as out_file, sessions[ftp_id].download_stream(path) as stream:
                        async for block in stream.iter_by_block():
                            await out_file.write(block)
                except aioftp.errors.StatusCodeError:
                    pass
                else:
                    break            

class batch_downloader():
    def __init__(self, ftp_connections, async_num):
        self.async_num = async_num
        self.downloaders = [mini_batch_downloader(ftp_connections, ii) for ii in range(async_num)]

    def download_batch(self, files):
        loop = getattr(threadLocal, 'loop', None)
        if loop is None:
            loop = asyncio.new_event_loop()
            threadLocal.loop = loop
            asyncio.set_event_loop(loop)
            [mbd.disconnect() for mbd in self.downloaders]
            loop.run_until_complete(asyncio.wait(tuple([mbd.connect() for mbd in self.downloaders])))
        tasks = tuple(self.downloaders[ii].download([ff for ff in files[ii::self.async_num]]) for ii in range(self.async_num))
        loop.run_until_complete(asyncio.wait(tasks))
