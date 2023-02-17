from mstorch.data.mass_spec.reader.base import BaseMsFileReader


class XcaliburReader(BaseMsFileReader):
    
    def read_chromatograms(self):
        raise NotImplementedError()
    
    def read_spectra(self, ms_level=None):
        raise NotImplementedError()
    
    def read_spectrum(self, scan_index):
        raise NotImplementedError()

    def read_chromatogram(self, chrom_index):
        raise NotImplementedError()        
    
    def num_spectra(self):
        raise NotImplementedError()

    def num_chromatograms(self):
        raise NotImplementedError() 