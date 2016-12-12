#!/usr/bin/env python3
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np
import math
import random
import os


def _kahan_range(start, stop, step):
    assert step > 0.0
    total = start
    compo = 0.0
    while total < stop:
        yield total
        y = step - compo
        temp = total + y
        compo = (temp - total) - y
        total = temp


class WavData:
    """Object for manipulating and analyzing WAV file data
    """

    def __init__(self, filename=None, fs=None, data=None):
        """Loads the given file or creates an instance with the given sampling
           frequency and data.
        """
        if filename:
            self.fs, self.data = wavfile.read(filename)

            if self.fs == 96000:
                pass
                # self.data = []
            else:
                print('Loaded {}'.format(filename))
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
            data=self.data[int(start*self.fs):int(end*self.fs)],
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

    def fft(self, target_freq=441):
        """Computes the FFT for the wav data. Only returns the real component.
        """
        if target_freq > self.fs:
            print('WARN: upsampling is unimplemented')
        data = fft(self.data)
        if target_freq > 1:
            target_length = self.duration() * target_freq
            n = self.fs / target_freq
            data = [sum(data[int(i):int(i + n)]) / n for i in _kahan_range(0.0, len(data), n)]
            data = data[:int(target_length)]
        return np.log(data[:len(data) // 2]).real

class Dataset:
    """Object for loading and accessing a specified dataset
    """

    @staticmethod
    def load_wavs(data_folder='data', split=None, sample_length=1.0, cross_validation=5):
        """Loads the given dataset and performs a training/testing split using
           the given percentage of total data.

           Loaded WAV files are split into examples of duration sample_length
           if provided. If sample_length is None, the WAV files are used as the
           examples.
        """
        data = _load_labeled_data(data_folder, sample_length)

        training = []
        testing = [] if split else None

        for label in data:
            y = data[label][0]  # List[List[int]] # one-hot version of label
            l = data[label][1]  # The full array of all examples with this label

            print('Loaded {} samples for label {}'.format(len(l), label))

            # Shuffle to deal with changes across longer recordings
            random.shuffle(l)

            if split:
                split1 = l[:int(math.floor(len(l) * split))]  # training data from this label
                split2 = l[int(math.floor(len(l) * split)):]  # testing data from this label

                # conceptually, these are lists of tuples, where the first value
                # is the fft and the second is the label
                training += [(x, y) for x in split1]
                testing += [(x, y) for x in split2]
            else:
                training += [(x, y) for x in l]

        return Dataset(training, testing, cross_validation)

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

    def __init__(self, training, testing, cross_validation=None):
        self._raw_training = training
        self._testing = testing

        cross_validation = int(cross_validation) if cross_validation and cross_validation >= 1 else 1
        self._training = [[] for _ in range(cross_validation)]
        self.i = 0

        if (len(self._raw_training) % cross_validation) != 0:
            print('WARN: cross validation folds not all equal')
        if len(self._raw_training) < cross_validation:
            print('WARN: need more training data')

        for i in range(len(training)):
            self._training[i % cross_validation].append(training[i])

    def _vstack(self, data):
        examples = np.vstack([x[0] for x in data])
        labels = np.vstack([x[1] for x in data])

        return examples, labels

    def training(self):
        """Shuffles the training data and returns [[examples], [labels]] where
           examples[i] corresponds to labels[i].
        """
        self.i = (self.i + 1) % len(self._training)

        if self._testing:
            data = self._training[0]
        else:
            data = np.concatenate(self._training[:self.i] + self._training[self.i+1:])

        random.shuffle(data)

        return self._vstack(data)

    def testing(self):
        """Returns the testing data as [[examples], [labels]] where examples[i]
           corresponds to labels[i].

           If the dataset was given k for k-fold cross validation, returns all
           examples not in the i'th fold.
        """
        data = self._training[self.i]
        if self._testing:
            data = self._testing
        else:
            data = self._training[self.i]

        return self._vstack(data)

    def x_shape(self):
        """Returns the shape of an example in this dataset. For example, the
           test.wav example with 2 second samples has shape 88200.
        """
        return self._training[0][0][0].shape[0]

    def y_shape(self):
        """Returns the shape of a label in this dataset. For example, three
           unique labels mapped to a one-hot vector would yield 3.
        """
        return self._training[0][0][1].shape[0]


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
