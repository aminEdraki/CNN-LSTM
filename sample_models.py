from keras.models import Model
from keras.models import Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D, SpatialDropout1D, Conv2D,
                          MaxPooling2D, Flatten, Reshape, AveragePooling2D, Permute, ConvLSTM2D, Bidirectional)
from keras.initializers import RandomUniform
from keras import regularizers
import scipy.io
import numpy as np


def gabor_init(shape, dtype=None):
    w = scipy.io.loadmat('gabor_weights.mat')['w']
    return w


# LSTM 1 LAYER
def basic_lstm1(batch_input_shape=(None, 98, 350), batch_normalization=True):
    # LSTM input:
    # (batch_size, timesteps, input_dim)
    model = Sequential()
    model.add(Permute((2, 1), batch_input_shape=batch_input_shape))

    model.add(LSTM(128, batch_input_shape=batch_input_shape, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # model.add(Dropout(0.3))
    # model.add(Dense(64, activation='relu',
    #                 kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    model.add(Dense(31, activation='softmax'))
    return model


# LSTM 3 LAYER
def basic_lstm3(batch_input_shape=(None, 98, 39), batch_normalization=True):
    # LSTM input:
    # (batch_size, timesteps, input_dim)
    model = Sequential()
    model.add(Permute((2, 1), batch_input_shape=batch_input_shape))

    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())

    model.add(Dropout(0.1))
    model.add(Dense(31, activation='softmax',
                    kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    return model


def cnn_lstm(init_with_gabor=True, train_cnn=False, cnn_pooling=False, pool_size=(2, 2),
             input_shape=(23, 98, 1)):

    model = Sequential()
    if init_with_gabor:
        model.add(Conv2D(25, (69, 39), data_format='channels_last', activation='relu', input_shape=input_shape,
                         padding='same', kernel_initializer=gabor_init, trainable=train_cnn))
    else:
        model.add(Conv2D(25, (20, 40), data_format='channels_last', activation='relu', input_shape=input_shape,
                         padding='same', trainable=train_cnn, kernel_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l2(0.01)))

    model.add(MaxPooling2D(pool_size, padding='same'))
    model.add(Dropout(0.1))
    # model.add(Conv2D(25, (40, 20), data_format='channels_last', activation='relu', padding='same'))
    # model.add(Dropout(0.05))
    # model.add(MaxPooling2D(pool_size, padding='same'))
    #
    # model.add(Conv2D(15, (20, 15), data_format='channels_last', activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size, padding='same'))
    # model.add(Dropout(0.05))

    model.add(Permute((2, 1, 3)))
    a = model.output_shape
    model.add(Reshape((a[1], a[2]*a[3])))

    model.add(LSTM(128, batch_input_shape=(None, 98, None), return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(128, batch_input_shape=(None, 98, None), return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(128, batch_input_shape=(None, 98, None), return_sequences=True))

    # model.add(Dropout(0.1))
    model.add(LSTM(128, batch_input_shape=(None, 98, None), return_sequences=False))
    # model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(31, activation='softmax'))
    return model
