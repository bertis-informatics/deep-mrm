import numpy as np
from .calibration_curve import generate_calibration_curves

def extract_deepmrm_ratio(row, 
                          heavy_is_reference=False,
                          selected_xic=True):
    if len(row['boxes']) < 1:
        return np.nan
    # if row['scores'][0] < boundary_score_th:
    #     return np.nan
    
    if selected_xic:
        if heavy_is_reference:
            ratio = row['light_area'][0]/row['heavy_area'][0]
        else:
            ratio = row['heavy_area'][0]/row['light_area'][0] 
    else:
        if heavy_is_reference:
            ratio = row['light0_area'][0]/row['heavy0_area'][0]
        else:
            ratio = row['heavy0_area'][0]/row['light0_area'][0]     
    return ratio
