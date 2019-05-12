#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import os
import threading
import argparse
import numpy as np
import mimetypes
from scipy.misc import imresize
import imageio

import json, ast
from sklearn.model_selection import GroupKFold, train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import pandas as pd
import altair as alt
import uuid
from queue import Queue
import multiprocessing
from .aio_tools import threaded_downloader

def process_img(img_f, rois, target_shape, augmentation):
    with open(img_f, 'rb') as ff:
        img = imageio.imread(ff)
    img_roi = sel_rois(img, rois)
    scale_fraction = min((float(target_shape[ii]) / img_roi.shape[ii] for ii in range(2)))
    img_rescaled = imresize(img_roi, scale_fraction, interp='bilinear')
    if len(img_rescaled.shape) < 3:
        img_rescaled = img_rescaled.reshape(img_rescaled.shape + (1, ))
    img_rescaled = np.pad(img_rescaled, [(0, target_shape[0]-img_rescaled.shape[0]), (0, target_shape[1]-img_rescaled.shape[1]), (0, 0)], mode='constant', constant_values=0)
    if target_shape[-1] == 1 and img_rescaled.shape[-1] == 3:
        img_rescaled = img_rescaled.mean(-1, keepdims=True)
    if augmentation is not None:
        img_rescaled = augmentation.random_transform(img_rescaled)
    if target_shape[-1] == 3 and img_rescaled.shape[-1] == 1:
        img_rescaled = img_rescaled.repeat(3, 2)
    return img_rescaled.astype(np.float32) / 255.0


class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_files, dsconnections, classes=None, rois=[], augmentation=None, target_shape=(224, 224, 1), batch_size=32, shuffle=True, max_epoch_samples=np.inf, name="Train"):
        'Initialization'
        self.q = multiprocessing.JoinableQueue()
        self.q_downloaded = multiprocessing.JoinableQueue()
        self.download_threads = [threaded_downloader(self.q, self.q_downloaded, ftp_connections=dsconnections, async_num=4) for thr in range(1)]
        self.img_files = img_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rois = rois
        self.augmentation = augmentation
        self.target_shape = target_shape

        if classes is None:
            self.label_space = sorted(list(set([img_f['category'] for img_f in img_files])))
        else:
            self.label_space = classes
        # one hot encode according to indices given in classes, or sorted list if classes are not specified
        self.enc = lambda x: to_categorical([self.label_space.index(xi) for xi in x], num_classes=len(self.label_space))
        self.max_epoch_samples = max_epoch_samples
        self.on_epoch_end()
        self.name = name

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_imgs = min(self.max_epoch_samples, len(self.img_files))
        return int(np.ceil(float(num_imgs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
        # Schedule downloads to queue

        for index in range(len(self)):
            # Generate indexes of the batch
            indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
            # Find list of IDs
            img_files_temp = [self.img_files[k] for k in indices]
            files = [(ff['id'], ff['path'], str(uuid.uuid4()) + os.path.splitext(ff['path'])[-1], ff) for ff in img_files_temp]
            self.q.put(files)

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Generate data
        files = self.q_downloaded.get()
        # Initialization
        X = np.zeros((len(files), ) + self.target_shape)
        for ii, img_f in enumerate(files):
            X[ii] = process_img(img_f[2], self.rois, self.target_shape, self.augmentation)
            os.remove(img_f[2])
        y = self.enc([img_f[3]['category'] for img_f in files])
        self.q_downloaded.task_done()
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.__data_generation()
        return X, y

def parse_augmentation_args(unparsed):
    augmentation_parser = argparse.ArgumentParser()
    augmentation_parser.add_argument('--width-shift-range', type=float, help="Image augmentation for training", default=0.05)
    augmentation_parser.add_argument('--height-shift-range', type=float, help="Image augmentation for training", default=0.05)
    augmentation_parser.add_argument('--zoom-range', type=float, help="Image augmentation for training", default=0.02)
    aug_config, _ = augmentation_parser.parse_known_args(unparsed)
    return aug_config

def get_augmentation(aug_config):
    augmentation = image.ImageDataGenerator(
             horizontal_flip=aug_config.flip_horizontal,
             vertical_flip=aug_config.flip_vertical,
             rotation_range=aug_config.rotation_angle,
             width_shift_range=aug_config.horizontal_translation_range,
             height_shift_range=aug_config.vertical_translation_range,
             shear_range=aug_config.shear_range,
             zoom_range=aug_config.scaling_range)
    return augmentation

def sel_rois(img, rois):
    img_out = img
    if len(rois) == 1:
        roi = rois[0]
        img_out = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    if len(rois) > 1:
        img_out = [img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]] for roi in rois]
        img_out = np.concatenate(img_out)
    return img_out
