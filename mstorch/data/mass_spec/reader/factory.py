from pathlib import Path

from mstorch.enums import MassSpecDataFormat
from mstorch.data.mass_spec.reader.mzml_reader import MzMLFileReader
from mstorch.data.mass_spec.reader.xcalibur_reader import XcaliburReader

class MassSpecDataReaderFactory(object):

    @classmethod
    def get_mass_spec_reader(cls, file_path, in_memory=False):
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f'{file_path} does not exist')
        
        file_format = cls.find_mass_spec_type(file_path)

        if file_format == MassSpecDataFormat.MzML:
            ms_reader = MzMLFileReader(file_path, in_memory)
        elif file_format == MassSpecDataFormat.XcaliburRaw:
            ms_reader = XcaliburReader(file_path)
        else:
            raise NotImplementedError()
        return ms_reader


    @classmethod
    def find_mass_spec_type(cls, file_path):
        
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        if file_ext == '.raw':
            return MassSpecDataFormat.XcaliburRaw
        elif file_ext == '.wiff':
            return MassSpecDataFormat.Wiff
        elif file_ext == '.mzml':
            return MassSpecDataFormat.MzML
        elif file_ext == '.d':
            return MassSpecDataFormat.MassHunter
        else:
            raise ValueError('Unrecognizable file format')
        
        
        
