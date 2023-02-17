import collections
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path


class BaseMsFileReader(ABC):
    """
    An abstract base class for reading mass-spec files
    """
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    @abstractmethod
    def read_chromatograms(self):
        raise NotImplementedError()

    @abstractmethod
    def read_spectra(self, ms_level=None):
        raise NotImplementedError()

    @abstractmethod
    def read_spectrum(self, scan_index):
        raise NotImplementedError()

    @abstractmethod
    def read_chromatogram(self, chrom_index):
        raise NotImplementedError()        

    @abstractproperty
    def num_spectra(self):
        raise NotImplementedError()

    @abstractproperty
    def num_chromatograms(self):
        raise NotImplementedError() 