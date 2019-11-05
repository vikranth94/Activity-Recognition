import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import itertools


def get_details(name):
    if (name == 'PAM2'):
        num_classes = 12
        sensors = ['acc', 'gyr', 'mag']
        locations = ['wrist', 'ankle', 'chest']
        label_names = ['Lying', 'Sitting', 'Standing', 'Walking',
                       'Running', 'Cycling', 'Nordic_walking', 'Ascending_stairs',
                       'Descending_stairs', 'Vacuum_cleaning', 'Ironing', 'Rope_jumping']
        f_hz = 100
        dimensions = ['sensor', 'location', 'frequency']
        path = './Data/PAM2'

    elif (name == 'UCI'):
        num_classes = 6
        sensors = ['body_acc', 'gyr', 'total_acc']
        locations = ['waist']
        label_names = ['Walking', 'Walking_Upstairs', 'Walking_Downstairs',
                       'Sitting', 'Standing', 'Laying']
        f_hz = 50
        dimensions = ['sensor']
        path = './Data/UCI_HAR'

    elif (name == 'SHL'):
        num_classes = 8
        sensors = ['acc', 'gyr', 'linear_acc']
        locations = ['pocket']
        label_names = ['Still', 'Walk', 'Run',
                       'Bike', 'Car', 'Bus', 'Train', 'Subway']
        f_hz = 100
        dimensions = ['sensor', 'frequency']
        path = './Data/SHL'

    elif (name == 'OPP'):
        num_classes = 4
        sensors = ['acc', 'gyr', 'mag']
        locations = ['Back', 'RUA', 'RLA', 'LUA', 'LLA']
        label_names = ['Stand', 'Walk', 'Sit', 'Lie']
        f_hz = 25
        dimensions = ['sensor', 'location']
        path = './Data/OPP'

    elif (name == 'OPP_g'):
        num_classes = 17
        sensors = ['acc', 'gyr', 'mag']
        locations = ['Back', 'RUA', 'RLA', 'LUA', 'LLA']
        label_names = ['Stand', 'Walk', 'Sit', 'Lie']
        f_hz = 25
        dimensions = ['sensor', 'location']
        path = 'F:/Vikranth/Arm/Datasets/Opp'

    elif (name == 'mydata'):
        num_classes = 4
        sensors = ['acc', 'gyr']
        locations = ['left_wrist', 'right_wrist']
        label_names = [
            'sitting(still)', 'sitting(computer use)', 'walking', 'standing(still)']
        f_hz = 25
        dimensions = []
        path = './Motionsense'

    else:
        print("No such dataset")

    return num_classes, sensors, locations, label_names, f_hz, dimensions, path


def load_dataset(name, path, num_classes):
    if (name == 'PAM2'):
        X_train0 = np.load(os.path.join(path, 'X_train_{}.npy'.format(name)))
        y_train_binary = np.load(os.path.join(
            path, 'y_train_{}.npy'.format(name)))
        X_val0 = np.load(os.path.join(path, 'X_val_{}.npy'.format(name)))
        y_val_binary = np.load(os.path.join(path, 'y_val_{}.npy'.format(name)))
        X_test0 = np.load(os.path.join(path, 'X_test_{}.npy'.format(name)))
        y_test_binary = np.load(os.path.join(
            path, 'y_test_{}.npy'.format(name)))

    elif (name == 'UCI'):
        X_train0 = np.load(os.path.join(path, 'X_train_{}.npy'.format(name)))
        y_train = np.load(os.path.join(path, 'y_train_{}.npy'.format(name)))
        X_test0 = np.load(os.path.join(path, 'X_test_{}.npy'.format(name)))
        y_test = np.load(os.path.join(path, 'y_test_{}.npy'.format(name)))
        X_val0 = X_test0[:500]
        y_val = y_test[:500]
        y_train_binary = keras.utils.to_categorical(y_train, num_classes)
        y_test_binary = keras.utils.to_categorical(y_test, num_classes)
        y_val_binary = keras.utils.to_categorical(y_val, num_classes)

    elif (name == 'SHL'):
        X_train0 = np.load(os.path.join(path, 'X_train_{}.npy'.format(name)))
        y_train_binary = np.load(os.path.join(
            path, 'y_train_{}.npy'.format(name)))
        X_val0 = np.load(os.path.join(path, 'X_val_{}.npy'.format(name)))
        y_val_binary = np.load(os.path.join(path, 'y_val_{}.npy'.format(name)))
        X_test0 = np.load(os.path.join(path, 'X_test_{}.npy'.format(name)))
        y_test_binary = np.load(os.path.join(
            path, 'y_test_{}.npy'.format(name)))

    elif (name == 'OPP'):
        X_train0 = np.load(os.path.join(path, 'X_train_{}.npy'.format(name)))
        y_train = np.load(os.path.join(path, 'y_train_{}.npy'.format(name)))
        X_test0 = np.load(os.path.join(path, 'X_test_{}.npy'.format(name)))
        y_test = np.load(os.path.join(path, 'y_test_{}.npy'.format(name)))
        X_val0 = X_test0
        y_val = y_test
        y_train_binary = keras.utils.to_categorical(y_train, num_classes)
        y_test_binary = keras.utils.to_categorical(y_test, num_classes)
        y_val_binary = keras.utils.to_categorical(y_val, num_classes)

    elif (name == 'OPP_g'):
        X_train0 = np.load(os.path.join(path, 'X_train_OPP.npy'))
        y_train = np.load(os.path.join(path, 'y_train_OPP.npy'))
        X_test0 = np.load(os.path.join(path, 'X_test_OPP.npy'))
        y_test = np.load(os.path.join(path, 'y_test_OPP.npy'))
        X_val0 = X_test0
        y_val = y_test
        y_train_binary = keras.utils.to_categorical(y_train, num_classes)
        y_test_binary = keras.utils.to_categorical(y_test, num_classes)
        y_val_binary = keras.utils.to_categorical(y_val, num_classes)

    elif (name == 'mydata'):
        X_data0 = np.load(os.path.join(path, 'X_{}.npy'.format(name)))
        y_data = np.load(os.path.join(path, 'y_{}.npy'.format(name)))
        np.random.seed(42)
        p = np.random.permutation(y_data.shape[0])
        y_data = y_data[p]
        X_data0 = X_data0[p]
        z = 500
        X_train0 = X_data0[:z]
        y_train = y_data0[:z]
        X_test0 = X_data0[z:]
        y_test = y_data0[z:]
        X_val0 = X_test0
        y_val = y_test
        y_train_binary = keras.utils.to_categorical(y_train, num_classes)
        y_test_binary = keras.utils.to_categorical(y_test, num_classes)
        y_val_binary = keras.utils.to_categorical(y_val, num_classes)

    else:
        print("No such dataset")

    return X_train0, y_train_binary, X_val0, y_val_binary, X_test0, y_test_binary


def get_data(X_data, sens, locs, sensors, locations):
    n_sensors = len(sensors)
    n_locations = len(locations)
    if set(locs).issubset(locations) and set(sens).issubset(sensors):
        index = []
        for loc in locs:
            l_ind = locations.index(loc)
            for sen in sens:
                s_ind = sensors.index(sen)
                ind = l_ind * (n_sensors * 3) + (s_ind * 3)
                index.extend([ind, ind + 1, ind + 2])
        index.sort()
        print(index)
        X_data = X_data[:, index, :, :]

    else:
        print('Sensor not available')

    return X_data


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.shape[0] - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L, a.shape[1]), strides=(S * n, n, a.strides[1]))


def data_reshaping(X_tr, X_va, X_tst):

    # make it into (frame_number, dimension, window_size, channel=1) for convNet
    _, win_len, dim = X_tr.shape
    X_tr = np.swapaxes(X_tr, 1, 2)
    X_va = np.swapaxes(X_va, 1, 2)
    X_tst = np.swapaxes(X_tst, 1, 2)

    X_tr = np.reshape(X_tr, (-1, dim, win_len, 1))
    X_va = np.reshape(X_va, (-1, dim, win_len, 1))
    X_tst = np.reshape(X_tst, (-1, dim, win_len, 1))

    return X_tr, X_va, X_tst


def calculate_metrics(model, X_test, y_test_binary):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test_binary, axis=1)
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return cf_matrix, accuracy, micro_f1, macro_f1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
