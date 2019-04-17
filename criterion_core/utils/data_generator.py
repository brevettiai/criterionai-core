#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import os
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

def parse_training_args(unparsed):
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_fraction', type=float, nargs='*', help='fraction of data used for validation', required=False, default=0.15)
    parser.add_argument('--development_fraction', type=float, nargs='*', help='fraction of data used for development set', required=False, default=0.1)
    parser.add_argument('--epochs', type=int, help='Number of epochs for optimization', required=False, default=100)
    parser.add_argument('--early-stopping-patience', type=int, help='Patience with no improvement in validation score during optimization', default=10)
    parser.add_argument('--max-epoch-samples', type=int, help='Number of samples in an epoch for optimization', required=False, default=4096)
    parser.add_argument('--batch-size', type=int, help='Number of samples in batch', required=False, default=64)
    parser.add_argument('--seed', help='Seed the datasplit and batch generators', default=42)
    config_train, unparsed_train = parser.parse_known_args(unparsed)
    return config_train, unparsed_train

def parse_image_class_settings(settings):
    # Image processing related parameters and formatting
    input_shape = [settings["imageHeight"], settings["imageWidth"]]  + [3 if settings["type"]=='rgb' else 1]
    img_format = settings.get('img_format')
    rois = json.loads(json.dumps(ast.literal_eval(settings["inputRois"]))) # Hack to read array with tuples [(a, b), (c, d)]
    # Training specific parameters that specifies the image classifier model output, and how to parse the input datasets
    class_map = json.loads(json.dumps(ast.literal_eval(settings["classMap"].strip(' ')))) # Hack to convert "{'a': 1}" to '{"a": 1}'
    # "classes" is intended to be the different classes / categories that the classifier can identify
    # "train_folders" holds names of all folders that will (and can) used for 
    if len(class_map) > 1:
        class_map_inv = {ff: kk for kk, vv in class_map.items() for ff in vv}
    else:
        class_map_inv = {}

    critical_classes = [cw for cw in settings['classWeights'].replace(' ', '').split(',') if len(cw)>0]
    return input_shape, img_format, rois, class_map, class_map_inv, critical_classes

def parse_category_info(class_map, critical_classes, folders):
    if len(class_map) > 1:
        classes = sorted(list(class_map.keys()))
    else:
        classes = sorted(folders)
        class_map = {cl: [cl] for cl in classes}
    classes = [cl for cl in classes if cl in critical_classes] + [cl for cl in classes if cl not in critical_classes]

    train_folders = np.concatenate([sorted(class_map[kk]) for kk in classes]).tolist() 
    all_classes = classes + sorted([ff for ff in folders if ff not in train_folders])
    
    class_indices = {cl: ii for ii, cl in enumerate(classes)}

    return classes, class_indices, train_folders, all_classes

def process_img(img_f, rois, target_shape, augmentation):
    with  img_f['file_sys'].open(img_f['path'], 'rb') as ff:
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
    def __init__(self, img_files, classes=None, rois=[], augmentation=None, target_shape=(224, 224, 1), batch_size=32, shuffle=True, max_epoch_samples=np.inf):
        'Initialization'
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_imgs = min(self.max_epoch_samples, len(self.img_files))
        return int(np.ceil(float(num_imgs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.img_files))
        self.i = 0
        if self.shuffle == True:
            np.random.shuffle(self.indices)
        

    def __data_generation(self, img_files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        # Generate data
        X = np.zeros((len(img_files), ) + self.target_shape)
        for ii, img_f in enumerate(img_files):
            X[ii] = process_img(img_f, self.rois, self.target_shape, self.augmentation)
        y = self.enc([img_f['category'] for img_f in img_files])
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        self.i = index
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(img_files_temp)

        return X, y

    def __iter__(self):
        # Iterators are iterables too.
        # Adding this functions to make them so.
        return self

    def __next__(self):
        if self.i < len(self):
            i = self.i
            self.i += 1
            return self[i]
        else:
            raise StopIteration()


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

def split_data_develop(objs, config, log):
    log.info("Splitting dataset in training and development set...")
    labels = [vi['label'] for vi in objs]
    class_i, cnt_i = np.unique(labels, return_counts=True)
    no_split = []
    for ii, cl in enumerate(class_i):
        if cnt_i[ii] <= 1:
            no_split += np.where(np.array(labels)==cl)[0].tolist()
    split = np.ones(len(objs), dtype=np.bool)
    split[no_split] = False
    train, devel = train_test_split(np.array(objs)[split].tolist(), test_size=config.development_fraction, stratify=np.array(labels)[split], random_state=config.seed, shuffle=True)
    train += np.array(objs)[no_split].tolist()
    return train, devel

def split_data_groups(objs, config, log):
    labels = [vi['label'] for vi in objs]
    groups = [vi['batch'] for vi in objs]
    n_splits = min(int(np.ceil(1.0 / config.validation_fraction)), len(np.unique(groups)))


    if n_splits > 1:
        log.info("Splitting groupwise")
        gkf = GroupKFold(n_splits=n_splits)
        gkf.random_state = config.seed
        gkf.shuffle = True
        train_set, test_set = [(train, test) for train, test in gkf.split(objs, y=labels, groups=groups)][0]
    else:
        train_set = np.arange(len(objs))
        test_set = []
    test = np.array(objs)[test_set].tolist()
    train, devel = split_data_develop(np.array(objs)[train_set], config, log)
    validation = devel + test
    log.info("Done splitting dataset in training and validation!")
    return train, validation, {'develop': devel, 'test': test}

def split_training_data(train_folders, objects, config_train, class_map_inv, log):
    # For training, optimization and evaluation purposes, the training data is split in disjunct sets of "train" objects and "valid" objects
    # "valid" objects are again drawn from 2 groups:
    #           1. (development set) simply a fraction of the training data, drawn from the same distribution
    #           2. (validation set) data from separate "batches" (top level folder), that is assumed to be more independent, and prevent overfitting
    # "test" objects are objects that do not have a class assigned, combined with the validation set.
    train_k = {}
    validation_k = {}
    val_sets = {}
    test_objects = []
    for kk, vv in objects.items():
        train_objects = [obj for obj in vv if obj['label'] in train_folders]
        test_objects += [obj for obj in vv if obj['label'] not in train_folders]
        train_k[kk], validation_k[kk], val_sets[kk] = split_data_groups(train_objects, config_train, log)

    train = np.concatenate([tt for tt in train_k.values()])
    valid = np.concatenate([vv for vv in validation_k.values()])
    
    summary_fields = ['path', 'dataset', 'batch', 'label', 'class']
    summary_dict = {kk:[obj[kk] for obj_set in [train, valid, test_objects] for obj in obj_set] for kk in summary_fields}
    summary_dict.update({'split': [split for split, obj_set in zip(['train', 'valid', 'test'], [train, valid, test_objects]) for ii in range(len(obj_set))]})
    
    file_data_df = pd.DataFrame(summary_dict, columns=summary_fields + ['split']).set_index('path').sort_index()

    return train, valid, test_objects, file_data_df


def sel_rois(img, rois):
    img_out = img
    if len(rois) == 1:
        roi = rois[0]
        img_out = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    if len(rois) > 1:
        img_out = [img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]] for roi in rois]
        img_out = np.concatenate(img_out)
    return img_out


def make_folder_summary(objects, test_classes, key='class'):
    folders = {}
    ordered_test_classes = [':'.join([str(ii), ci]) for ii, ci in enumerate(test_classes)]
    for obj in objects:
        path_key = ':'.join([obj['dataset'], obj['batch']])
        if folders.get(path_key) is None:
            folders[path_key] = {cl: 0 for cl in test_classes}
        folders[path_key][obj[key]] += 1
    folder_keys = sorted(folders.keys())
    folder_info = np.array([(fk, ordered_test_classes[ii], folders[fk][cl]) for fk in folder_keys for ii, cl in enumerate(test_classes)])
    folder_data = pd.DataFrame({'Dataset:folder/': folder_info[:, 0],
                        'class_name': folder_info[:, 1],
                        'count': folder_info[:, 2].astype(int)})
    folder_summary = alt.Chart(folder_data).mark_bar().encode(x='sum(count)', y='Dataset:folder/', color='class_name').configure_axis(labelLimit=1000)

    folder_table = {'Dataset:folder/': folder_keys}
    folder_table.update({ordered_test_classes[ii]: [folders[kk][cl] for kk in folder_keys] for ii, cl in enumerate(test_classes)})
    folder_table_pd = pd.DataFrame(folder_table, columns=['Dataset:folder/']+ordered_test_classes)

    folder_json = folder_summary.to_json().replace('"_0"', '"Dataset:folder/"') # Ugly hack to remedy error in encoding
    return folder_data, folder_table_pd, folder_json