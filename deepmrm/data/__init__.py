from mstorch.utils.data.collate import SelectiveCollation
from deepmrm.constant import XIC_KEY, TIME_KEY, TARGET_KEY

obj_detection_collate_fn = SelectiveCollation(
                            exclusion_keys=[TARGET_KEY, 
                                            TIME_KEY, 
                                            XIC_KEY,
                                            'manual_peak_quality']
                        )