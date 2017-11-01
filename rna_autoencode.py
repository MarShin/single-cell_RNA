from keras.layers import Input, Dense, merge, Dropout, Activation
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.regularizers import l2


import matplotlib.pyplot as plt
import numpy as np
import data_handler as dh



# print x_test[0:5]

encoding_dim = 5000
l2_penalty_ae = 1e-2
noise_factor = 0.5

def denoise(x_train, x_test):
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    print '////'
    print 'complete denoise'
    print x_test_noisy[0:5]

    return x_train_noisy, x_test_noisy


def trainAE(denoise_ae=True):

    x_train_target, x_test_target = dh.load_data()

    if denoise_ae:
        x_train_noisy, x_test_noisy = denoise(x_train_target, x_test_target)

    # this is our input placeholder
    input_dim = x_train_noisy.shape[1]
    input_img = Input(shape=(input_dim,))

    encoded = Dense(encoding_dim, activation='relu')(input_img)
    encoded = Dense(2500, activation='relu', kernel_regularizer=l2(l2_penalty_ae))(encoded)

    decoded = Dense(5000, activation='relu', kernel_regularizer=l2(l2_penalty_ae))(encoded)
    decoded = Dense(input_dim, activation='sigmoid' ,kernel_regularizer=l2(l2_penalty_ae))(encoded)

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='adadelta', loss='mse')
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train_target,
                    epochs=50,
                    batch_size=50,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test_target),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), EarlyStopping(
                            monitor='val_loss', patience=25,  mode='auto')])

    autoencoder.save('autoencoder.h5')

    return autoencoder

    # note that we take them from the *test* set
    # encoded_imgs = encoder.predict(x_test)
    # decoded_imgs = decoder.predict(encoded_imgs)
    # return decoded_imgs

if __name__ == "__main__":
    # x_train, x_test = denoise()
    result = trainAE()
