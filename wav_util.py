#!/usr/bin/env python3
from scipy.fftpack import fft
from scipy.io import wavfile
import math


class WavData:
    """Object for manipulating and analyzing WAV file data
    """

    def __init__(self, filename):
        """Loads the given file and uses its sampling frequency and data
        """
        self.fs, self.data = wavfile.read(filename)

    def __init__(self, fs, data):
        """Initializes with the given sampling frequency and data
        """
        self.fs = fs
        self.data = data

    def get_duration(self):
        """Returns the duration of this WAV in seconds
        """
        return len(self.data) / self.fs

    def slice(self, start, end):
        """Returns a slice of this WAV given start and end offsets in seconds
        """
        data = self.data[math.ceil(start * self.fs):math.floor(end * self.fs)]
        return WavData(self.fs, data)

    def fft(self):
        """Computes the FFT for the contained data
        """
        return fft(self.data)
