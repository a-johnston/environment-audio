#!/usr/bin/env python3
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np
import math
import random
import os


class WavData:
    """Object for manipulating and analyzing WAV file data
    """

    def __init__(self, filename=None, fs=None, data=None):
        """Loads the given file or creates an instance with the given sampling
           frequency and data.
        """
        if filename:
            self.fs, self.data = wavfile.read(filename)
        else:
            self.fs = fs
            self.data = data

    def duration(self):
        """Returns the duration of this WAV in seconds
        """
        return len(self.data) / self.fs

    def slice(self, start, end):
        """Returns a slice of this WAV given start and end offsets in seconds
        """
        return WavData(
            fs=self.fs,
            data=self.data[math.ceil(start*self.fs):math.floor(end*self.fs)],
        )

    def get_samples(self, sample_duration, include_tail=False):
        """Returns a list of sample_duration second slices from this sample. If
           sample_duration is None, [self] is returned.
        """
        if sample_duration is None:
            return [self]

        start = 0.0
        segments = []

        while start + sample_duration <= self.duration():
            segments.append(self.slice(start, start + sample_duration))
            start += sample_duration

        if include_tail:
            segments.append(self.slice(start, self.duration()))

        return segments

    def fft(self):
        """Computes the FFT for the contained data
        """
        return fft(self.data)


class Dataset:
    """Object for loading and accessing a specified dataset
    """

    @staticmethod
    def load_wavs(data_folder='data', split=0.9, sample_length=5.0):
        """Loads the given dataset and performs a training/testing split using
           the given percentage of total data.

           Loaded WAV files are split into examples of duration sample_length
           if provided. If sample_length is None, the WAV files are used as the
           examples.
        """
        data = _load_labeled_data(data_folder, sample_length)

        training = []
        testing = []

        for label in data:
            y = data[label][0]
            l = data[label][1]

            random.shuffle(l)

            split1 = l[:math.floor(len(l) * split)]
            split2 = l[math.floor(len(l) * split):]

            training += [(x, y) for x in split1]
            testing += [(x, y) for x in split2]

        return Dataset(training, testing)

    @staticmethod
    def mock(num_per_label=[300, 300], split=0.9):
        """Generates a mock dataset with the specified number of each label.
           Each example has a single feature which is its class label.
        """
        training = []
        testing = []

        labels = list(range(len(num_per_label)))
        m = _map_label_to_one_hot(labels)

        for label in labels:
            y = m[label]
            l = [(y, y)] * num_per_label[label]

            training += l[:math.floor(len(l) * split)]
            testing += l[math.floor(len(l) * split):]

        return Dataset(training, testing)

    def __init__(self, training, testing):
        self._training = training
        self._testing = testing

    def training(self):
        """Shuffles the training data and returns [[examples], [labels]] where
           examples[i] corresponds to labels[i].
        """
        random.shuffle(self._training)
        return np.array(list(zip(*self._training)))

    def testing(self):
        """Returns the testing data as [[examples], [labels]] where examples[i]
           corresponds to labels[i].
        """
        return np.array(list(zip(*self._testing)))

    def x_shape(self):
        """Returns the shape of an example in this dataset. For example, the
           test.wav example with 2 second samples has shape (88200,).
        """
        return self._training[0][0].shape

    def y_shape(self):
        """Returns the shape of a label in this dataset. For example, three
           unique labels mapped to a one-hot vector would yield (3,).
        """
        return self._training[0][1].shape


def _load_labeled_data(data_folder, sample_length):
    """Returns a list of labels and examples given a data_folder with the
       structure:

            data_folder
                label1
                    wav1.wav
                    wav2.wav
                label2
                    wav3.wav

        List order may not be consistent between runs.
    """
    label_folders = os.listdir(data_folder)
    m = _map_label_to_one_hot(label_folders)

    labeled_data = {}

    for label in label_folders:
        files = os.listdir(os.path.join(data_folder, label))
        files = [os.path.join(data_folder, label, x) for x in files]
        samples = [WavData(f).get_samples(sample_length) for f in files]
        samples = [[sample.fft() for sample in example] for example in samples]
        labeled_data[label] = (m[label], sum(samples, []))

    return labeled_data


def _map_label_to_one_hot(labels):
    m = {}

    for i in range(len(labels)):
        onehot = np.zeros(len(labels))
        onehot[i] = 1
        m[labels[i]] = onehot

    return m
