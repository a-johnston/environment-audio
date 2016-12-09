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

    def fft(self, step_size=1):
        """Computes the FFT for the contained data
        """
        step_size = int(step_size)
        if step_size <= 1:
            step_size = 1
        print('fft resolution: %s' % (len(fft(self.data)[::step_size])))
        return fft(self.data)[::step_size] # python indexing magic

class Dataset:
    """Object for loading and accessing a specified dataset
    """

    @staticmethod
    def load_wavs(data_folder='data', split=0.9, sample_length=5.0, cross_validation=5, downsampling=1):
        """Loads the given dataset and performs a training/testing split using
           the given percentage of total data.

           Loaded WAV files are split into examples of duration sample_length
           if provided. If sample_length is None, the WAV files are used as the
           examples.
        """
        data = _load_labeled_data(data_folder, sample_length, downsampling)

        training = []
        if cross_validation <= 1:
            cross_validation = 1 # aka no cross validation
        cv_training = [None] * cross_validation
        testing = []

        for label in data:
            y = data[label][0]  # List[List[int]] # one-hot version of label
            true_class_label = y
            l = data[label][1]  # The full array of all examples with this label
            examples_for_label = l

            random.shuffle(l)

            split1 = l[:int(math.floor(len(l) * split))]  # the slice of the array for training
            split2 = l[int(math.floor(len(l) * split)):]  # the slice of the array for testing

            training += [(x, y) for x in split1]
            # conceptually, training is a list of tuples, where the first value is the fft and the second is the label

            testing += [(x, y) for x in split2]

        current_cv_set = 0
        random.shuffle(training)
        for label_id, item in enumerate(training):
            one_hot_label = item[1]
            fft = item[0]
            if cv_training[current_cv_set] is None:
                cv_training[current_cv_set] = []
            cv_training[current_cv_set].append(item) 

            current_cv_set += 1
            current_cv_set = current_cv_set % len(cv_training)

        print('data preservation:\n%s\n%s\nThese two should be equal' % (training[0][0][0], cv_training[0][0][0][0],))
        # this means that cv_training is a list of training sets that can be used

        return Dataset(training, testing, cv_training)

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

    def __init__(self, training, testing, cv_training=None):
        self._training = training
        if cv_training is None:
            self._cv_training = training
        else:
            self._cv_training = cv_training
            print('set cv training data')
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
           test.wav example with 2 second samples has shape 88200.
        """
        return self._training[0][0].shape[0]

    def y_shape(self):
        """Returns the shape of a label in this dataset. For example, three
           unique labels mapped to a one-hot vector would yield 3.
        """
        return self._training[0][1].shape[0]


def _load_labeled_data(data_folder, sample_length, downsampling=1):
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
        samples = [[sample.fft(downsampling) for sample in example] for example in samples]
        labeled_data[label] = (m[label], sum(samples, []))

    return labeled_data


def _map_label_to_one_hot(labels):
    m = {}

    for i in range(len(labels)):
        onehot = np.zeros(len(labels))
        onehot[i] = 1
        m[labels[i]] = onehot

    return m
