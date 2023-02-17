import collections
from pyopenms import MSSpectrum as PyOpenMsSpectrum
from mstorch.data.mass_spec.base import BaseSpectrum, IsolationWindow
from scipy.signal import find_peaks

Peak = collections.namedtuple('Peak', ['mz', 'intensity'])


class PyOpenMsSpectrumWrapper(BaseSpectrum):

    def __init__(self, ms_exp, scan_index):
        super().__init__(scan_index)
        self.ms_exp = ms_exp
        self._spec = ms_exp.getMetaData().getSpectrum(scan_index)
        self._ms_spec = None

    @property
    def ms_spec(self):
        if self._ms_spec is None:
            if self._spec.getType() == 2:
                # to centroid profile spectrum
                profile_spec = self.ms_exp.getSpectrum(self.scan_index)
                x, y = profile_spec.get_peaks()
                peak_indexes = find_peaks(y)[0]
                profile_spec.set_peaks( (x[peak_indexes], y[peak_indexes]) )
                profile_spec.setType(1) # it's centroid spec
                self._ms_spec = profile_spec
            else:
                self._ms_spec = self.ms_exp.getSpectrum(self.scan_index)
        return self._ms_spec

    def get_peaks(self):
        peaks_ = self.ms_spec.get_peaks()
        peaks = [
                Peak(peaks_[0][idx], peaks_[1][idx]) for idx in range(len(peaks_[0]))
            ]
        return peaks

    def get_isolation_window(self):
        if self.get_ms_level() < 2:
            return ValueError('Not a MS2 spectrum')

        precursors = self._spec.getPrecursors()
        assert len(precursors) == 1

        precursor = precursors[0]
        return IsolationWindow(
                    precursor.getMZ(),
                    precursor.getIsolationWindowLowerOffset(),
                    precursor.getIsolationWindowUpperOffset())        


    def get_ms_level(self):
        return self._spec.getMSLevel()
    
    def get_retention_time(self):
        return self._spec.getRT()

    def __get_peak(self, index):
        if index >= 0:
            pk = self.ms_spec[index]
            return Peak(pk.getMZ(), pk.getIntensity())
        return None

    def find_peak_with_tolerance(self, mz, tolerance):
        index = self.ms_spec.findNearest(mz, tolerance.get_mz_tolerance(mz))
        return self.__get_peak(index)

    def find_peak(self, min_mz, max_mz):
        center_mz = (min_mz+max_mz)*0.5
        tol_mz = center_mz - min_mz
        # index = self.ms_spec.findHighestInWindow(center_mz, tol_mz, tol_mz)
        index = self.ms_spec.findNearest(center_mz, tol_mz, tol_mz)
        return self.__get_peak(index)

    def find_all_peaks(self, min_mz, max_mz):
        raise NotImplementedError()
    
    def find_intensity(self, min_mz, max_mz):
        pk = self.find_peak(min_mz, max_mz)
        return 0 if pk is None else pk.intensity
            
