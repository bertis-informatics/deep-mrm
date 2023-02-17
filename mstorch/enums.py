from enum import Enum, IntEnum

class DeviceType(Enum):
    CPU = 'cpu'
    GPU = 'gpu'
    TPU = 'tpu'

class SplitMethod(Enum):
    KFOLD = 'kfold'
    HOLDOUT = 'holdout'
    HOLDOUT_PER_CLASS = 'holdout_per_class'


class PartitionType(Enum):
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'

class ToleranceUnit(IntEnum):
    PPM = 0
    DA = 1
    MZ = 2

class MassSpecDataFormat(Enum):
    XcaliburRaw = 'Thermo Scientific Raw'
    MzML = 'HUPO-PSI mzML' # https://github.com/HUPO-PSI/mzML
    Wiff = 'Sciex Wiff'
    MassHunter = 'Agilent .d'
