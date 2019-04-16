from tensorflow import keras
import numpy as np
from math import ceil
import asyncio
import aiofiles
import imageio
from types import CoroutineType

class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, batch_size=32, shuffle=True, max_epoch_samples=np.inf):
        self.indices = np.arange(len(samples))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = samples
        self.max_epoch_samples = max_epoch_samples
        self.on_epoch_end()
        self.loop = asyncio.get_event_loop()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return ceil(len(self.indices) / self.batch_size)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)

    @staticmethod
    async def get_sample(sample):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        print("Loading")
        async with aiofiles.open(sample['image']['path'], mode='rb') as f:
            img = await f.read(f)
            #img = await imageio.imread(f)
        print("Loaded")

        return img

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        samples = self.samples[indices]

        batch = [self.get_sample(s) for s in samples]

        if isinstance(batch[0], CoroutineType):
            futures = asyncio.gather(*batch)
            batch = self.loop.run_until_complete(futures)


        # Generate data

        return samples
