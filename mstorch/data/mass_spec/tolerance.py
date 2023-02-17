from mstorch.enums import ToleranceUnit



class Tolerance(object):
    def __init__(self, value, unit=ToleranceUnit.PPM):
        assert isinstance(unit, ToleranceUnit)
        self.value = value
        self.unit = unit

    def get_mz_tolerance(self, mz):
        if self.unit == ToleranceUnit.PPM:
            return mz*self.value*1e-6
        elif self.unit == ToleranceUnit.MZ:
            return self.value
        else:
            raise NotImplementedError()

    def get_dalton_tolerance(self, mz, charge):
        if self.unit == ToleranceUnit.DA:
            return self.value
        else:
            return self.get_mz_tolerance(mz)*charge

    def is_within(self, mz1, mz2):
        tol = self.get_mz_tolerance(mz1)
        return abs(mz2-mz1) < tol

    

    