import os
import sys

def prep_test_lists(base_dir):
    # given directories, creates lists of all the files (with paths) that will be used by keras
    # (once fed to  pred_data_xyz)
    # this function ensure that the filenames match to ensure proper order when fitting model
    # mask_type = {'ideal_amplitude', 'phase_sensitive', 'ideal_complex'}
    # this function is meant to be used on CHiME data only
    # noisy_spects_dir point to the directory in which the sub-directories contain matlab .mat files of noisy spectrograms
    # clean_spect_dir points to the cleaned spectrograms
    # (masks_dir points to the masks)
    # Note: this function does not check if noisy_spects_dir contain actual spectrograms, just '.mat' files with reasonable path
    # build file lists
    train_list = [path+'/'+file for path,_,files in sorted(os.walk(base_dir)) for file in sorted(files) if (file.endswith('.mat'))]
    return train_list