import collections
from abc import ABC, abstractmethod, abstractproperty


class IsolationWindow:

    def __init__(self, mz, lower_offset, upper_offset):
        self.mz = mz
        self.lower_offset = lower_offset
        self.upper_offset = upper_offset

    @property
    def min_mz(self):
        return self.mz - self.lower_offset

    @property
    def max_mz(self):
        return self.mz + self.upper_offset        

    def contains(self, mz):
        return (self.min_mz <= mz) & (mz <= self.max_mz)


class BaseSpectrum(ABC):

    def __init__(self, scan_index):
        self.scan_index = scan_index

    @abstractmethod
    def get_peaks(self):
        # yield (mz, intensity)
        raise NotImplementedError()

    @abstractmethod
    def get_ms_level(self):
        raise NotImplementedError()

    @abstractmethod
    def get_retention_time(self):
        raise NotImplementedError()        

    def get_scan_index(self):
        return self.scan_index

    @abstractmethod
    def find_peak(self, min_mz, max_mz):
        # return the most intense peak in the given range
        raise NotImplementedError()

    def find_peak_with_tolerance(self, mz, tolerance):
        tol = tolerance.get_mz_tolerance(mz)
        return self.find_peak(mz-tol, mz+tol)

    @abstractmethod
    def find_all_peaks(self, min_mz, max_mz):
        raise NotImplementedError()

    
    @abstractmethod
    def get_isolation_window(self):
        raise NotImplementedError()        


class BaseChromatogram(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_peaks(self):
        raise NotImplementedError()

    @abstractmethod
    def get_isolation_window(self, q=1):
        raise NotImplementedError()
    