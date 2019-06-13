#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import os
import numpy as np
import cv2

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from criterion_core.utils.augmentation import get_augmentation_pipeline

from . import io_tools
from . import image_proc


COLOR_MODE = {1:cv2.IMREAD_GRAYSCALE,
              3:cv2.IMREAD_COLOR}


class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_files, classes=None, rois=[], augmentation=None, target_shape=(224, 224, 1), batch_size=32,
                 shuffle=True, target_mode="classification", max_epoch_samples=np.inf,
                 interpolation='linear', anti_aliasing=False):
        'Initialization'
        self.img_files = img_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rois = rois
        self.target_mode = target_mode
        self.augmentation = get_augmentation_pipeline(augmentation, target_shape, rois, interpolation=interpolation, anti_aliasing=anti_aliasing)
        self.target_shape = target_shape
        self.indices = None
        if classes is None:
            self.label_space = sorted(list(set([img_f['category'] for img_f in img_files])))
        else:
            self.label_space = classes
        # one hot encode according to indices given in classes, or sorted list if classes are not specified
        self.enc = self.label_encoder
        self.color_mode = COLOR_MODE[target_shape[2]]
        self.max_epoch_samples = max_epoch_samples
        self.on_epoch_end()

    def label_encoder(self, x):
        return to_categorical([self.label_space.index(xi) for xi in x], num_classes=len(self.label_space))

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_imgs = min(self.max_epoch_samples, len(self.img_files))
        return int(np.ceil(float(num_imgs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.img_files))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, img_files_batch):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        buffers = io_tools.download_batch(img_files_batch)
        # Initialization
        X = np.zeros((len(buffers), ) + self.target_shape)
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
        img_files_batch = [self.img_files[k] for k in batch_indices]
        X = self.__data_generation(img_files_batch)
        if self.target_mode=="classification":
            categories = [blob['category'] for blob in img_files_batch]
            y = self.enc(categories)
            return X, y
        elif self.target_mode=="input":
            return X, X
        elif self.target_mode == "samples":
            return X, img_files_batch
