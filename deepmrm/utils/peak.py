from turtle import back
from scipy.signal import find_peaks, peak_prominences
import numpy as np

# peak bounday detection
def detect_peak_boundary(time, xic, rt, window_len=21):

    # find peaks
    peaks, _ = find_peaks(xic)

    # find a peak at the given retention time
    peak_idx = np.searchsorted(time[peaks], rt)
    i0 = peaks[peak_idx-1]
    i1 = peaks[peak_idx]
    peak_idx = peak_idx-1 if rt - time[i0] < time[i1] - rt else peak_idx
    peak_idx = peaks[peak_idx]

    # find peak's base positions 
    # prominences, left_bases, right_bases = peak_prominences(xic, [peak_idx], wlen=window_len)
    # start_idx, end_idx = left_bases[0], right_bases[0]
    start_idx = 0
    for i in range(peak_idx-1, 0, -1):
        if xic[i] > xic[i+1]:
            start_idx = i+1
            break

    end_idx = len(xic) - 1
    for i in range(peak_idx+1, len(xic)-1):
        if xic[i] < xic[i+1]:
            end_idx = i
            break

    return start_idx, end_idx



def calculate_peak_area_with_time(time, intensity, start_time, end_time, return_xic_seg=False):
    # Calculate peak area according to the way Skyline does
    # https://skyline.ms/announcements/home/support/thread.view?entityId=4b24cd1b-ab9a-102e-87a2-4c1490ad0666&_docid=thread%3A4b24cd1b-ab9a-102e-87a2-4c1490ad0666
 
    # Determine sample points to estimate the intensity in the peak boundary
    step = (time[-1] - time[0])/(len(time) - 1)
    num = max(int(1.2 * ((end_time-start_time) / step)), 3)
    x = np.linspace(start_time, end_time, num)

    # Estimate peak intensity
    y = np.interp(x, time, intensity)

    # Background estimatio
    background = min(y[0], y[-1])*(x[-1] - x[0])
    
    # Peak integration
    peak_area = np.trapz(y, x)

    if return_xic_seg:
        return peak_area, background, y

    return peak_area, background


def calculate_peak_area_with_index(time, intensity, start_idx, end_idx):
    
    x = time[start_idx:end_idx+1]
    y = intensity[start_idx:end_idx+1]

    # Background estimatio
    background = min(y[0], y[-1])*(x[-1] - x[0]) 
    
    # Peak integration
    peak_area = np.trapz(y, x)

    return peak_area, background