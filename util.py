from datetime import datetime
from os import listdir
from os.path import isfile, join

import os
import librosa
import librosa.display
import glob
import random

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from livelossplot import PlotLossesKeras
import itertools
from tensorflow.keras.utils import Sequence
from scipy.signal import spectrogram


def plot_confusion_matrix(cm, unique_labels, show=True, output=None,
                          title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() + 1).astype(str))
    plt.yticks(tick_marks)

    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)

    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if output is not None:
        plt.savefig(output)
    if show:
        plt.show()
    plt.close()
    return output



# Necessário na minha máquina. Estava ocorrendo um erro devido à GPU e esse código resolveu.
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
        
def balance_data(X, labels):
    """Balances data duplicating some instances of the minority classes"""
    balX = None
    baly = None
    unique, counts = np.unique(labels, return_counts=True)
    max_inst = max(counts)
    # For every class duplicate the necessary amount
    for label in unique:
        data_of_label = []
        for x, y in zip(X, labels):
            if y == label:
                data_of_label.append(x)
        data_of_label = np.asarray(data_of_label)
        extra = data_of_label[np.random.choice(data_of_label.shape[0],
                                               max_inst,
                                               replace=True), :]
        if balX is not None:
            balX = np.stack((balX, extra), axis = 0) 
            baly = np.stack((baly, np.asarray([label for i in range(len(extra))])))
        else:
            balX = extra
            baly = np.asarray([label for i in range(len(extra))])
    balX = np.concatenate(balX)
    baly = np.concatenate(baly)
    return balX, baly

os.makedirs("features/", exist_ok=True)

def extract_features(file_name, duration=None, max_pad_len=None):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    feature = os.path.splitext(os.path.basename(file_name))[0] + ".npy"
    if os.path.isfile(os.path.join("./features/", feature)):
        return np.load(os.path.join("./features/", feature))
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=duration) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if max_pad_len is not None:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None 
    np.save(os.path.join("./features/", feature), mfccs)
    return mfccs


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = spectrogram(audio,
                                     fs=sample_rate,
                                     window='hann',
                                     nperseg=nperseg,
                                     noverlap=noverlap,
                                     detrend=False)
    return freqs, times, np.log(spec.astype(np.float32) + eps)


def save_spec_img(out, spec, fmt='png'):
    result = Image.fromarray((spec * 255.0).astype(np.uint8))
    result.save(out + '.' + fmt)
    

def spec_load_npy(path, remove_bad_file=True):
    try: 
        spc = np.load(path).reshape(SPEC_SHAPE_HEIGTH,
                                    SPEC_SHAPE_WIDTH,
                                    CHANNELS)
        spc = normalize(spc)
    except Exception as ex:
        if remove_bad_file:
            if not os.path.isfile(path):
                print(f'Removing file {path}: is not a file',
                      cur_frame=currentframe(), mtype='W')
            else:
                os.remove(path)
        return np.zeros(shape=(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH, CHANNELS))
    return spc


def spec_load_img(path, remove_bad_file=True):
    try: 
        spc = np.array(Image.open(path)).reshape(SPEC_SHAPE_HEIGTH,
                                                 SPEC_SHAPE_WIDTH,
                                                 CHANNELS)
        spc = normalize(spc)
    except Exception as ex:
        if remove_bad_file:
            if not os.path.isfile(path):
                print(f'Removing file {path}: is not a file',
                      cur_frame=currentframe(), mtype='W')
            else:
                os.remove(path)
        return np.zeros(shape=(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH, CHANNELS))
    return spc


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 


def wav_to_specdata(path,
                    normalize_pixels=True,
                    reshape_to_image=False,
                    duration=8):
    b, sr = librosa.load(path, duration=duration)
    _, _, spc = log_specgram(b, sr)    
    spc = spc.astype('float32')
    if reshape_to_image:
        spc = spc.reshape(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH)
        spc = spc.reshape(SPEC_SHAPE_HEIGTH, SPEC_SHAPE_WIDTH, 1)
    if normalize_pixels:
        spc = scale(spc, 0, 1)
    return spc

def extract_specs(file_name, duration, max_pad_len=None):
    features_path = f'features/spectrograms/{duration}/'
    os.makedirs(features_path, exist_ok=True)
    feature_name = f'{os.path.splitext(os.path.basename(file_name))[0]}.npy'
    feature_path = os.path.join(features_path, feature_name)
    if (os.path.isfile(feature_path)):
        return np.load(feature_path)
    spec = wav_to_specdata(file_name, normalize_pixels=True)
    if max_pad_len is not None:
        pad_width = max_pad_len - spec.shape[1]
        spec = np.pad(spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    np.save(feature_path, spec)
    return spec


def balance_metadata(X, labels):
    """Balances data duplicating some instances of the minority classes"""
    balX = None
    baly = None
    unique, counts = np.unique(labels, return_counts=True)
    max_inst = max(counts)
    # For every class duplicate the necessary amount
    for label in unique:
        data_of_label = []
        for x, y in zip(X, labels):
            if y == label:
                data_of_label.append(x)
        data_of_label = np.asarray(data_of_label)
        extra = data_of_label[np.random.choice(len(data_of_label),
                                               max_inst,
                                               replace=True)]
        if balX is not None:
            balX = np.stack((balX, extra), axis=0) 
            baly = np.stack((baly, np.asarray([label for i in range(len(extra))])))
        else:
            balX = extra
            baly = np.asarray([label for i in range(len(extra))])
    balX = np.concatenate(balX)
    baly = np.concatenate(baly)
    return balX, baly


class Generator(Sequence):
    """
    Generator for data batching. See keras.utils.Sequence.
    """
    def __init__(self, paths, labels, batch_size: int, loader_fn: callable,
                 shuffle: bool = True, expected_shape=None, loader_kw=None,
                 not_found_ok=False, shape=None):
        """
        Creates a new generator.
        
        Args:
            paths (list like): List containing paths.
            labels (list like): List of the respective data labels.
            batch_size (int): Batch size.
            loader_fn (callable): Function for data loading.
            shuffle (bool, optional): If True, will shuffle the data before.
                Defaults to True.
            expected_shape (tuple, optional): If not None, it will check each
                shape of the data loaded. Defaults to None.
            loader_kw (dict, optional): Key arguments to pass on to the loader.
                Defaults to None.
            not_found_ok (bool, optional): Choose to load another instance if
                the loader fails to find a file. Defaults to False.
            shape (tuple): Reshapes the original data loaded to a new shape.
        """
        assert len(paths) > 0
        self._paths = paths
        self._labels = labels
        self._batch_size = batch_size
        self._loaderkw = loader_kw if loader_kw else {}
        self._loader = loader_fn
        self._expected_shape = expected_shape
        self._not_found_ok = not_found_ok
        self._shape = shape

        if shuffle:
            dataset = list(zip(self._paths, self._labels))
            random.shuffle(dataset)
            self._paths, self._labels = zip(*dataset)

    def _get_random_instance(self):
        i = np.random.randint(0, len(self._paths))
        return self._paths[i], self._labels[i]

    def __getitem__(self, index):
        paths = self._paths[(index*self._batch_size):
                            ((index+1)*self._batch_size)]
        labels = self._labels[(index*self._batch_size):
                              ((index+1)*self._batch_size)]
        paths_and_labels = list(zip(paths, labels))
        # Fill batches
        x = []
        y = []
        threshold = 0
        for path_label in paths_and_labels:
            if self._not_found_ok:
                try:
                    data = self._loader(path_label[0], **self._loaderkw)
                    x.append(data)
                    y.append(path_label[1])
                except FileNotFoundError as fnf:
                    print(f'File {path_label[0]} not found ({str(fnf)})')
                    # If not found, append a new path to load
                    p, l = self._get_random_instance()
                    paths_and_labels.append((p, l))
                    # Increase a threshold value to avoid infinite loops
                    threshold += 1

                    if threshold == 10:
                        # (threshold can be any value)
                        raise RuntimeError(
                            'Threshold value reached. Error when '
                            'trying to read the files provided '
                            '(not able to fill the batch).')
                    continue
            else:  
                data = self._loader(path_label[0], **self._loaderkw)
                x.append(data)
                y.append(path_label[1])
            if (self._expected_shape is not None and x[-1].shape !=
                    self._expected_shape):
                print(f'Expected shape {self._expected_shape} when loading '
                      f'{path_label[0]}. But found shape of {x[-1].shape} '
                      'instead')
                # If the last read data is not in the expected shape
                p, l = self._get_random_instance()
                paths_and_labels.append((p, l))
                # Increase a threshold value to avoid infinite loops
                threshold += 1
                # Remove the last instance
                x.pop()
                y.pop()
                # If all data was tried to be read, raise an exception
                if threshold == self._batch_size:
                    raise RuntimeError('Threshold value reached. Error when '
                                       'trying to read the files provided '
                                       '(not able to fill the batch).')
                continue
        if self._shape is not None:
            x = np.reshape(np.asarray(x), (self._batch_size, *self._shape))
            return [x], np.asarray(y)
        return [np.asarray(x)], np.asarray(y)

    def __len__(self):
        return int(np.floor(len(self._paths) / self._batch_size))