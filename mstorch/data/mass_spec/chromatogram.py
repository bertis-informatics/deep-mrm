import collections
from mstorch.data.mass_spec.base import BaseChromatogram, IsolationWindow
from pyopenms import MSChromatogram as PyOpenMsChromatogram

ChromPeak = collections.namedtuple('ChromPeak', ['retention_time', 'intensity'])


class PyOpenMsChromatogramWrapper(BaseChromatogram):

    def __init__(self, ms_exp, chrom_index):
        super().__init__()
        self.chrom_index = chrom_index
        self.ms_exp = ms_exp
        self._chrom = self.ms_exp.getChromatogram(chrom_index)
        self._ms_chrom = self._chrom

    @property
    def ms_chrom(self):
        if self._ms_chrom is None:
            self._ms_chrom = self.ms_exp.getChromatogram(self.chrom_index)
        return self._ms_chrom

    def get_peaks(self):
        peaks_ = self.ms_chrom.get_peaks()
        peaks = [ChromPeak(peaks_[0][idx], peaks_[1][idx]) 
                    for idx in range(len(peaks_[0]))]
        return peaks        
    
    def get_isolation_window(self, q=1):
        assert q in (1, 3)
        ion = self._chrom.getPrecursor() if q == 1 else self._chrom.getProduct()
        return IsolationWindow(
                    ion.getMZ(),
                    ion.getIsolationWindowLowerOffset(),
                    ion.getIsolationWindowUpperOffset())
