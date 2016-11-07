import numpy as np
import scipy.io as sio

def extract_load(file):
    work_dir = '/Users/Near/Desktop/MESSL/mvdr_test/dev/mask/ideal_complex/data/'
    # load masks data
    masks = sio.loadmat(work_dir + 'F02_01BC020F_STR.mat')['data'][0][0][0]
    return masks