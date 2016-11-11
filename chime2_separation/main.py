import process
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, Dense, Convolution1D

mask_dir = '/Users/Near/Desktop/MESSL/mvdr_test/dev/mask/ideal_complex/data/'
spect_dir = '/Users/Near/Desktop/MESSL/mvdr_test/dev/spectrogram/data/'

masks_list = process.extract_load(mask_dir)
spects_list = process.extract_load(spect_dir)


(X,y,nm_f) = process.prep_data_for_keras(masks_list, spects_list, input_shape=(50, 150, 513), start=0)

print "Building model: input->LSTM:1024->Dense:513=output :: optimizer=RMSprop,loss=binary_crossentropy"
# define sequential model
model = Sequential()
# the 1st LSTM layer
model.add(LSTM(input_dim=513, input_length=None, output_dim=1024, return_sequences=True))
# output layer
model.add(TimeDistributed(Dense(output_dim=513)))
model.compile(optimizer='RMSprop',loss='binary_crossentropy')

hist = model.fit(X, y, nb_epoch=100, batch_size=128, verbose=2, shuffle=True)