import nilib as ni
import scipy.io as sio
import numpy as np
import keras.backend as K
base_dir = '/scratch/near/test/mask/'
model_dir = '/scratch/near/exp_second_my_loss_ideal_amplitude/exp_2016-12-04 17:56:49/'
save_dir = '/scratch/near/test_prediction/'
test_list = ni.prep_test_lists(base_dir)
print len(test_list)
#load the latest model

def my_loss(y_pred, y_true):
    #0 is clean, 1 is noisy
    mask, noisy = tf.split(2, 2, y_true)
    split2, split3 = tf.split(2,2,y_pred)
    # return tf.reduce_mean(tf.square(split2*split1 - split0*split1))
    return K.mean(K.square(split2 * noisy - mask*noisy),axis=-1)
print "loading model...."
from keras.models import model_from_json
import tensorflow as tf
tf.python.control_flow_ops = tf
jsonfile = open(model_dir+'run_0033_model_architecture.json')
model = jsonfile.read()
loaded_model = model_from_json(model)
loaded_model.load_weights(model_dir+'run_0033_model_weights.h5')
print "model loaded"
print "......................"
print "start predicting...."
for file in test_list:
    noisy = sio.loadmat(base_dir+file)['data'][0][0][0]
    noisy = noisy.swapaxis(0,1)
    test_input = noisy.reshape(1,-1,513)
    test_input = 20*np.log10(abs(test_input))
    test_output = loaded_model.predict(test_input)
    test_output = test_output.reshape(-1,1026)
    mask = test_output[:,0:513]
    clean = noisy * mask
    sio.savemat(save_dir+file, {'clean': clean})

print "finish"