from pyopenms import OnDiscMSExperiment, MSExperiment, MzMLFile
from mstorch.data.mass_spec.reader.base import BaseMsFileReader
from mstorch.data.mass_spec.spectrum import PyOpenMsSpectrumWrapper
from mstorch.data.mass_spec.chromatogram import PyOpenMsChromatogramWrapper


class MzMLFileReader(BaseMsFileReader):

    def __init__(self, file_path, in_memory=False):
        super().__init__(file_path)

        if in_memory:
            exp = MSExperiment()
            MzMLFile().load(str(self.file_path), exp)
        else:
            exp = OnDiscMSExperiment()
            _ = exp.openFile(str(self.file_path))
        self.exp = exp

    def _get_experiment(self):
        return self.exp

    def read_chromatograms(self):
        exp = self._get_experiment()
        num_chroms = exp.getNrChromatograms()
        for chrom_idx in range(num_chroms):
            yield PyOpenMsChromatogramWrapper(exp, chrom_idx)

    def read_spectra(self, ms_level=None):
        exp = self._get_experiment()
        num_spectra = exp.getNrSpectra()
        for scan_index in range(num_spectra):
            spec = exp.getSpectrum(scan_index)
            if ms_level is not None and spec.getMSLevel() != ms_level:
                continue
            yield PyOpenMsSpectrumWrapper(exp, scan_index)

    def read_chromatogram(self, chrom_index):
        exp = self._get_experiment()
        return PyOpenMsChromatogramWrapper(exp, chrom_index)

    def read_spectrum(self, scan_index):
        exp = self._get_experiment()
        return PyOpenMsSpectrumWrapper(exp, scan_index)
    
    @property
    def num_chromatograms(self):
        exp = self._get_experiment()
        return exp.getNrChromatograms()
    
    @property
    def num_spectra(self):
        exp = self._get_experiment()
        return exp.getNrSpectra()


def test():
    from pathlib import Path
    mzml_path = Path('/mnt/c/Users/jungk/Documents/Dataset/PRM_P100/mzml/F20131112_LINCS_MCF7-Rep1-01_anisomycin_01.mzML')
    ms_reader = MzMLFileReader(mzml_path)
    print(ms_reader.num_spectra)
    print(ms_reader.num_chromatograms)
