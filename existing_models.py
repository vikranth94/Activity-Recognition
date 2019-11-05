import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Permute, Reshape


# useful functions

def reshape_data(X_tr, X_va, X_tst, network_type):
    _, win_len, dim = X_tr.shape
    print(network_type)

    if network_type == 'CNN' or network_type == 'ConvLSTM':
        # make it into (frame_number, dimension, window_size, channel=1) for convNet
        X_tr = np.swapaxes(X_tr, 1, 2)
        X_va = np.swapaxes(X_va, 1, 2)
        X_tst = np.swapaxes(X_tst, 1, 2)

        X_tr = np.reshape(X_tr, (-1, dim, win_len, 1))
        X_va = np.reshape(X_va, (-1, dim, win_len, 1))
        X_tst = np.reshape(X_tst, (-1, dim, win_len, 1))

    elif network_type == 'MLP':
        X_tr = np.reshape(X_tr, (-1, dim * win_len))
        X_va = np.reshape(X_va, (-1, dim * win_len))
        X_tst = np.reshape(X_tst, (-1, dim * win_len))

    return X_tr, X_va, X_tst


# Models

def model_MLP(dim, win_len, num_classes, num_hidden_mlp=256, p=0.3, batchnorm=True, dropout=True):
    model = Sequential(name='MLP')
    model.add(Dense(num_hidden_mlp, activation='relu',
                    input_shape=(dim * win_len,), name='dense_1'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_1'))
    if dropout:
        model.add(Dropout(p, name='Drop_1'))
    model.add(Dense(num_hidden_mlp, activation='relu', name='dense_2'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_2'))
    if dropout:
        model.add(Dropout(p, name='Drop_2'))
    model.add(Dense(num_classes, activation='softmax', name='dense_out'))
    return model


def model_CNN(dim, win_len, num_classes, num_feat_map=64, p=0., batchnorm=True, dropout=True):
    model = Sequential(name='CNN')
    model.add(Conv2D(num_feat_map, kernel_size=(1, 3),
                     activation='relu',
                     input_shape=(dim, win_len, 1),
                     padding='same', name='Conv_1'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_1'))
    model.add(MaxPooling2D(pool_size=(1, 2), name='Max_pool_1'))
    if dropout:
        model.add(Dropout(p, name='Drop_1'))
    model.add(Conv2D(num_feat_map, kernel_size=(1, 3),
                     activation='relu', padding='same', name='Conv_2'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_2'))
    model.add(MaxPooling2D(pool_size=(1, 2), name='Max_pool_2'))
    if dropout:
        model.add(Dropout(p, name='Drop_2'))
    model.add(Flatten(name='Flatten_1'))
    model.add(Dense(32, activation='relu'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_3'))
    if dropout:
        model.add(Dropout(p, name='Drop_3'))
    model.add(Dense(num_classes, activation='softmax', name='dense_out'))
    return model


def model_LSTM(dim, win_len, num_classes, num_hidden_lstm=32, p=0.3, batchnorm=True, dropout=True):
    model = Sequential(name='LSTM')
    model.add(LSTM(num_hidden_lstm,
                   input_shape=(win_len, dim),
                   return_sequences=True, name='Lstm_1'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_1'))
    if dropout:
        model.add(Dropout(p, name='Drop_1'))
    model.add(LSTM(num_hidden_lstm, return_sequences=False, name='Lstm_2'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_2'))
    if dropout:
        model.add(Dropout(p, name='Drop_2'))
    model.add(Dense(num_classes, activation='softmax', name='dense_out'))
    return model


def model_ConvLSTM(dim, win_len, num_classes, num_feat_map=64, p=0.3, batchnorm=True, dropout=True):
    model = Sequential(name='ConvLSTM')
    model.add(Conv2D(num_feat_map, kernel_size=(1, 3),
                     activation='relu',
                     input_shape=(dim, win_len, 1),
                     padding='same', name='Conv_1'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_1'))
    model.add(MaxPooling2D(pool_size=(1, 2), name='Max_pool_1'))
    if dropout:
        model.add(Dropout(p, name='Drop_1'))
    model.add(Conv2D(num_feat_map, kernel_size=(1, 3),
                     activation='relu', padding='same', name='Conv_2'))
    if batchnorm:
        model.add(BatchNormalization(name='Bn_2'))
    model.add(MaxPooling2D(pool_size=(1, 2), name='Max_pool_2'))
    if dropout:
        model.add(Dropout(p, name='Drop_2'))
    model.add(Permute((2, 1, 3), name='Permute_1'))  # for swap-dimension
    model.add(Reshape((-1, num_feat_map * dim), name='Reshape_1'))
    model.add(LSTM(32, return_sequences=False, stateful=False, name='Lstm_1'))
    if dropout:
        model.add(Dropout(p, name='Drop_3'))
    model.add(Dense(num_classes, activation='softmax', name='dense_out'))
    return model
