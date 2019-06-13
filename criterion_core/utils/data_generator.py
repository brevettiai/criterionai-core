# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import logging

import cv2
import numpy as np
from tensorflow import keras

from criterion_core.utils.augmentation import get_augmentation_pipeline
from . import io_tools

log = logging.getLogger(__name__)
COLOR_MODE = {1: cv2.IMREAD_GRAYSCALE,
              3: cv2.IMREAD_COLOR}


class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, classes=None, rois=[], augmentation=None, target_shape=(224, 224, 1), batch_size=32,
                 shuffle=True, target_mode="classification", max_epoch_samples=np.inf,
                 interpolation='linear', anti_aliasing=False):

        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rois = rois
        self.target_mode = target_mode
        self.augmentation = get_augmentation_pipeline(augmentation, target_shape, rois, interpolation=interpolation,
                                                      anti_aliasing=anti_aliasing)
        self.target_shape = target_shape
        self.indices = None
        self.interpolation = interpolation
        class_space = set(s['category'] for s in samples)
        self.classes = classes or sorted(
            set(item for sublist in class_space for item in sublist if item != "__UNLABELED__"))
        self.label_space = self.categorical_encoder(self.classes, class_space)
        self.color_mode = COLOR_MODE[target_shape[2]]
        self.max_epoch_samples = max_epoch_samples
        self.on_epoch_end()

    @staticmethod
    def categorical_encoder(classes, class_space):
        output = np.eye(len(classes), len(classes))
        space = {}
        for v in class_space:
            if len(v) > 0:
                try:
                    space[v] = np.vstack([output[classes.index(x)] for x in set(v)]).sum(0)
                except ValueError as ex:
                    log.exception("Data for generator contains labels not in accepted classes")
                    space[v] = np.full(len(classes), np.nan)
            else:
                space[v] = np.zeros(len(classes))
        return space

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_samples = min(self.max_epoch_samples, len(self.samples))
        return int(np.ceil(float(num_samples) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, samples_batch):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        buffers = io_tools.download_batch(samples_batch)
        # Initialization
        X = np.zeros((len(buffers),) + self.target_shape)
        for ii, buffer in enumerate(buffers):
            img = cv2.imdecode(np.frombuffer(buffer, np.uint8), flags=self.color_mode)
            if self.color_mode == cv2.IMREAD_COLOR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_t = self.augmentation.augment_image(img)

            if np.ndim(img_t) == 2:
                img_t = img_t[..., None]

            X[ii] = img_t

        return X

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        samples = [self.samples[k] for k in batch_indices]
        X = self.__data_generation(samples)
        if self.target_mode == "classification":
            y = np.stack([self.label_space[s['category']] for s in samples])
            return X, y
        elif self.target_mode == "input":
            return X, X
        elif self.target_mode == "samples":
            return X, samples
