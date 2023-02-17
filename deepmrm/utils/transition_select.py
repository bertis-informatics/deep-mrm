import pandas as pd
import numpy as np


def find_best_transition_index(time, xic, start_time, end_time):

    if xic.shape[1] < 3:
        return list(range(xic.shape[1]))

    step = (time[-1] - time[0])/(len(time) - 1)
    num = max(int(1.2 * ((end_time-start_time)/step)), 3)
    x = np.linspace(start_time, end_time, num)

    # create mean heavy peak shape profile
    heavy_y = np.array([np.interp(x, time, intensity) for intensity in xic[1, :, :]])
    light_y = np.array([np.interp(x, time, intensity) for intensity in xic[0, :, :]])
        
    # avoid divide-by-zero
    heavy_y[heavy_y == 0] = 1e-9
    light_y[light_y == 0] = 1e-9

    # mean_heavy_profile = np.mean([y/np.max(y) for y in heavy_y], axis=0)
    mean_heavy_profile = np.mean(heavy_y, axis=0)
    b = mean_heavy_profile / np.linalg.norm(mean_heavy_profile)
    a = light_y / np.linalg.norm(light_y, axis=1).reshape(-1, 1)
    a2 = heavy_y / np.linalg.norm(heavy_y, axis=1).reshape(-1, 1)
    peak_similarity = np.dot(a, b)

    # heavy and light pair similarity
    pair_similarity = np.diag(np.dot(a, a2.T))
    
    # find the best light transition 
    sim_cutoff = np.mean(peak_similarity) - 1*np.std(peak_similarity)
    m1 = peak_similarity > sim_cutoff

    # find the best pair
    sim_cutoff = np.mean(pair_similarity) - 1*np.std(pair_similarity)
    m2 = pair_similarity > sim_cutoff
    selected_index = np.where(m1 & m2)[0]

    # [TODO] check the consistency of light/heavy ratio
    if len(selected_index) < 1:
        selected_index = np.where(m1)[0]

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(x, heavy_y[0])
    # plt.plot(x, heavy_y[1])
    # plt.plot(x, heavy_y[2])
    # plt.plot(x, heavy_rep_y)
    # plt.savefig('temp/mean_peak_shape.jpg')

    return list(selected_index)