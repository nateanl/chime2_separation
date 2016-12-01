# Train an LSTM to take single channel noisy input spectrogram and produce a mask as output.
import numpy as np
import os
import warnings
import scipy.io as sio
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, Dense, Convolution1D, Activation
from keras.layers.normalization import BatchNormalization
import time
import random
import string
import nilib as ni
import tensorflow as tf
import h5py
base_dir = '/Users/Near/Desktop/mask/'
save_dir = '/Users/Near/Desktop/exp/'


# print experiment time for logging

run =0

def my_loss(y_pred, y_true):
    #0 is clean, 1 is noisy
    split0, split1 = tf.split(2, 2, y_true)
    split2, split3 = tf.split(2,2,y_pred)
    return tf.reduce_mean(tf.square(split2*split1 - split0*split1))

print "Building model: input->LSTM:1024->Dense:513=output :: optimizer=RMSprop,loss=binary_crossentropy"
# define sequential model
model = Sequential()
# the 1st LSTM layer
model.add(BatchNormalization(input_shape = (50,513), epsilon=1e-6, weights=None))
model.add(LSTM(input_dim=513, input_length=None, output_dim=1026, return_sequences=True))

# output layer
model.add(TimeDistributed(Dense(output_dim=513)))
model.add(Activation("sigmoid"))
model.compile(optimizer='RMSprop', loss='mse')

train_list = ni.prep_CHiME2_lists(base_dir, mask_type='ideal_amplitude')
print len(train_list)
num_proc_files =0
start_from_file = 0
while num_proc_files<len(train_list):
    print "Running experiment."
    start_time = time.strftime('%Y-%m-%d %T')
    print start_time
    # create new experiment folder
    print "Creating new folder for this experiment in:", save_dir
    newexp_folder_path = save_dir + '/' + "exp_" + start_time + '/'
    os.makedirs(newexp_folder_path)

    keras_inputs, keras_targets, num_proc_files = ni.prep_data_SpMa(train_list, input_shape=(1000, 50, 513), start=start_from_file)
    start_from_file = start_from_file + num_proc_files
    print keras_inputs.shape, num_proc_files
    nb_epoch = 10
    batch_size = 128
    print "Beginning fit: nb_epoch={0}, batch_size={1}".format(nb_epoch, batch_size)
    hist = model.fit(keras_inputs, keras_targets, nb_epoch=nb_epoch, batch_size= batch_size, verbose=2, shuffle=True)

    # save model to file every 1 run
    model_filename = "run_{0:04d}_model_architecture.json".format(run)
    model_wights_name = "run_{0:04d}_model_weights.h5".format(run)
    print "saving model to {0}".format(model_filename)
    #model.save(newexp_folder_path + model_filename)
    model_json = model.to_json()
    with open(newexp_folder_path + model_filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(newexp_folder_path+model_wights_name)
    print("Saved model to disk")

    # save history to text file
    hist_filename = "run_{0:04d}_keras_history.txt".format(run)
    print "saving history to {0}".format(hist_filename)
    with open(newexp_folder_path + hist_filename, "w") as text_file:
        text_file.write("epoch: {}\n".format(hist.epoch))
        text_file.write("loss: {}\n".format(hist.history["loss"]))
    run+=1
    pred = model.predict(keras_inputs, batch_size=batch_size)
    np.save(newexp_folder_path+'pred', pred)
    np.save(newexp_folder_path+'true', keras_targets)