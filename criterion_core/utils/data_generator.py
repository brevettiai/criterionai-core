# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import logging

import cv2
import numpy as np
from tensorflow import keras

from criterion_core.utils.augmentation import get_augmentation_pipeline
from . import io_tools

log = logging.getLogger(__name__)

COLOR_MODES = {"greyscale": 1,
               "bayer": 3,
               "rgb": 3}

COLOR_MODE_CV2_IMREAD = {
    "greyscale": cv2.IMREAD_GRAYSCALE,
    "bayer": cv2.IMREAD_GRAYSCALE,
    "rgb": cv2.IMREAD_COLOR
}


class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, classes=None, rois=[], augmentation=None, target_shape=(224, 224), batch_size=32,
                 shuffle=True, target_mode="classification", max_epoch_samples=np.inf,
                 interpolation='linear', anti_aliasing=False, color_mode="rgb"):

        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rois = rois
        self.target_mode = target_mode
        self.indices = None
        self.interpolation = interpolation
        class_space = set(s['category'] for s in samples)
        self.classes = classes or sorted(
            set(item for sublist in class_space for item in sublist if item != "__UNLABELED__"))
        self.label_space = self.categorical_encoder(self.classes, class_space)
        self.color_mode = color_mode
        self.target_shape = (int(target_shape[0]), int(target_shape[1]), COLOR_MODES[self.color_mode])
        self.augmentation = get_augmentation_pipeline(augmentation, self.target_shape, rois, interpolation=interpolation,
                                                      anti_aliasing=anti_aliasing)
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
        Xmask = np.zeros(len(buffers), dtype=np.bool)
        for ii, buffer in enumerate(buffers):
            try:
                img = cv2.imdecode(np.frombuffer(buffer, np.uint8), flags=COLOR_MODE_CV2_IMREAD[self.color_mode])
                if self.color_mode == "rgb":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif self.color_mode == "bayer":
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)

                img_t = self.augmentation.augment_image(img)

                if np.ndim(img_t) == 2:
                    img_t = img_t[..., None]

                X[ii] = img_t
                Xmask[ii] = True
            except:
                log.warning("Error loading %r" % samples_batch[ii])

        return X, Xmask

    def get_batch(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        samples = [self.samples[k] for k in batch_indices]
        X, Xmask = self.__data_generation(samples)
        return np.array(samples)[Xmask], X[Xmask]

    def __getitem__(self, index):
        'Generate one batch of data'
        samples, X = self.get_batch(index)
        if self.target_mode == "classification":
            y = np.stack([self.label_space[s['category']] for s in samples])
            return X, y
        elif self.target_mode == "input":
            return X, X
