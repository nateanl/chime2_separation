import numpy as np
import os
import warnings
import scipy.io as sio
import time


def prep_data_SpMa(train_list, input_shape=(100, 50, 513), start=0):
    ### prepares the data for Keras
    # keras_inputs will hold noisy spectrograms
    # keras_targets will hold desired masks
    # masks_list, spects_list should have the corresponding filenames in the same order
    # input_shape will define the shape of the data: (sample_num, input_length, features) (must all be positive)
    # start=n allows the user to start later in the lists    
    # the spectrogram data will be normalized to [-1..1]
    
    sample_num, input_length, features = input_shape
    
    finished = False
    num_proc_files = 0
    
    keras_inputs = None
    keras_targets = None
    
    # safety valve for runtime 
    max_allowed = 300 #180s = 3 minutes
    start_time = time.clock()
    time_used = 0
    
    while not(finished):
                
        # check amount of time used, and exit loop with warning if time exceeds limit
        if time_used >= max_allowed:
            warnings.warn("Time limit exceeded, returning as is!")
            break
            
        # load next masks and spectrogram, if fail return what we have with warning
        try:
            noisy = sio.loadmat(train_list[num_proc_files+start])['data'][0][0][0]
            clean = sio.loadmat(train_list[num_proc_files+start])['data'][0][0][1]
            mask = sio.loadmat(train_list[num_proc_files+start])['data'][0][0][2]
        except:
            warnings.warn("Not enough files, returning as is!")
            finished = True
            break
        
        # extract useful variables
        freq_num, frame_num = mask.shape
        # for CHIME3, should be (513, ~100:400, 6-7) 7 only if CH0 included 
        # freq_num will be input_dim for keras model
        # nb_channels is the number of mics for CHIME3

        # swap axes frequency_number and frame number
        mask = mask.swapaxes(0,1)
        clean = clean.swapaxes(0,1)
        noisy = noisy.swapaxes(0,1)
        
        # pad end of data to fit ('wrap' = add frames from the beginning)
        # this sort of works for input_length > frame_num but leads to some repetitions
        amount_to_pad = input_length - (frame_num % input_length)
        mask = np.pad(mask, ((0,amount_to_pad),(0,0)), 'wrap')
        # convert spectrogam values to decibel from complex
        clean = np.pad(20.0*np.log10(abs(clean)), ((0,amount_to_pad),(0,0)), 'wrap')
        noisy = np.pad(20.0 * np.log10(abs(noisy)), ((0, amount_to_pad), (0, 0)), 'wrap')
        # normalize per spect batch (per .mat file)
        clean /= np.max(abs(clean))
        noisy /= np.max(abs(noisy))
        # reshape to keras desired shape
        # -1 allows for automatic dimension calculation
        clean = clean.reshape(-1, input_length, features)
        noisy = noisy.reshape(-1, input_length, features)
        temp_keras_targets = np.concatenate((clean, noisy), axis=2)
        temp_keras_inputs = noisy.reshape(-1, input_length, features)
        # add to arrays to return 
        if keras_targets is None:
            keras_targets = temp_keras_targets
        else:
            keras_targets = np.concatenate((keras_targets, temp_keras_targets), axis=0)
            
        if keras_inputs is None:
            keras_inputs = temp_keras_inputs
        else:
            keras_inputs = np.concatenate((keras_inputs, temp_keras_inputs), axis=0)
        # check if we are done
        if len(keras_inputs) >= sample_num: finished=True
        # increment the number of processed files
        num_proc_files += 1
        
        # update time_used
        time_used = time.clock() - start_time
        
    # return the files in the right format
    if keras_inputs is None:
        return (None, None, 0)
    else:
        return (keras_inputs[:sample_num], keras_targets[:sample_num], num_proc_files)