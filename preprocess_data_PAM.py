import numpy as np
import pandas as pd
from os import listdir
import os.path
import zipfile
from keras.utils.np_utils import to_categorical
import json

def split_activities(labels, X, exclude_activities, borders=10 * 100):
    """
    Splits up the data per activity and exclude activity=0.
    Also remove borders for each activity.
    Returns lists with subdatasets

    Parameters
    ----------
    labels : numpy array
        Activity labels
    X : numpy array
        Data points
    borders : int
        Nr of timesteps to remove from the borders of an activity
    exclude_activities : list or tuple
        activities to exclude from the

    Returns
    -------
    X_list
    y_list
    """
    tot_len = len(labels)
    startpoints = np.where([1] + [labels[i] != labels[i - 1]
                                  for i in range(1, tot_len)])[0]
    endpoints = np.append(startpoints[1:] - 1, tot_len - 1)
    acts = [labels[s] for s, e in zip(startpoints, endpoints)]
    # Also split up the data, and only keep the non-zero activities
    xysplit = [(X[s + borders:e - borders + 1, :], a)
               for s, e, a in zip(startpoints, endpoints, acts)
               if a not in exclude_activities and e-borders+1>=0 and s+borders<tot_len]
    xysplit = [(Xs, y) for Xs, y in xysplit if len(Xs) > 0]
    Xlist = [Xs for Xs, y in xysplit]
    ylist = [y for X, y in xysplit]
    return Xlist, ylist


def sliding_window(frame_length, step, Xsampleslist, ysampleslist):
    """
    Splits time series in ysampleslist and Xsampleslist
    into segments by applying a sliding overlapping window
    of size equal to frame_length with steps equal to step
    it does this for all the samples and appends all the output together.
    So, the participant distinction is not kept

    Parameters
    ----------
    frame_length : int
        Length of sliding window
    step : int
        Stepsize between windows
    Xsamples : list
        Existing list of window fragments
    ysamples : list
        Existing list of window fragments
    Xsampleslist : list
        Samples to take sliding windows from
    ysampleslist
        Samples to take sliding windows from

    """
    Xsamples = []
    ysamples = []
    for j in range(len(Xsampleslist)):
        X = Xsampleslist[j]
        ybinary = ysampleslist[j]
        for i in range(0, X.shape[0] - frame_length, step):
            xsub = X[i:i + frame_length, :]
            ysub = ybinary
            Xsamples.append(xsub)
            ysamples.append(ysub)
    return Xsamples, ysamples


def transform_y(y, mapclasses, nr_classes):
    """
    Transforms y, a list with one sequence of A timesteps
    and B unique classes into a binary Numpy matrix of
    shape (A, B)

    Parameters
    ----------
    y : list or array
        List of classes
    mapclasses : dict
        dictionary that maps the classes to numbers
    nr_classes : int
        total number of classes
    """
    ymapped = np.array([mapclasses[c] for c in y], dtype='int')
    ybinary = to_categorical(ymapped, nr_classes)
    return ybinary

def get_header():
    axes = ['x', 'y', 'z']
    IMUsensor_columns = ['temperature'] + \
        ['acc_16g_' + i for i in axes] + \
        ['acc_6g_' + i for i in axes] + \
        ['gyroscope_' + i for i in axes] + \
        ['magnometer_' + i for i in axes] + \
        ['orientation_' + str(i) for i in range(4)]
    header = ["timestamp", "activityID", "heartrate"] + ["hand_" + s for s in IMUsensor_columns] \
        + ["chest_" + s for s in IMUsensor_columns] + ["ankle_" + s for s in IMUsensor_columns]
    return header

def addheader(datasets):
    """
    The columns of the pandas data frame are numbers
    this function adds the column labels

    Parameters
    ----------
    datasets : list
        List of pandas dataframes
    """
    header = get_header()
    for i in range(0, len(datasets)):
        datasets[i].columns = header
    return datasets


def numpify_and_store(X, y, X_name, y_name, outdatapath, shuffle=False):
    """
    Converts python lists x 3D and y 1D into numpy arrays
    and stores the numpy array in directory outdatapath
    shuffle is optional and shuffles the samples

    Parameters
    ----------
    X : list
        list with data
    y : list
        list with data
    X_name : str
        name to store the x arrays
    y_name : str
        name to store the y arrays
    outdatapath : str
        path to the directory to store the data
    shuffle : bool
        whether to shuffle the data before storing
    """
    X = np.array(X)
    y = np.array(y)
    # Shuffle the train set
    if shuffle is True:
        np.random.seed(123)
        neworder = np.random.permutation(X.shape[0])
        X = X[neworder, :, :]
        y = y[neworder, :]
    # Save binary file
    xpath = os.path.join(outdatapath, X_name)
    ypath = os.path.join(outdatapath, y_name)
    np.save(xpath, X)
    np.save(ypath, y)
    print('Stored ' + xpath, y_name)


def map_class(datasets_filled, exclude_activities):
    ysetall = [set(np.array(data.activityID)) - set(exclude_activities)
               for data in datasets_filled]
    class_ids = list(set.union(*[set(y) for y in ysetall]))
    class_labels = [ACTIVITIES_MAP[i] for i in class_ids]
    nr_classes = len(class_ids)
    mapclasses = {class_ids[i]: i for i in range(len(class_ids))}
    return class_labels, nr_classes, mapclasses


def split_data(Xlists, ybinarylists, indices):
    """ Function takes subset from list given indices

    Parameters
    ----------
    Xlists: tuple
        tuple (samples) of lists (windows) of numpy-arrays (time, variable)
    ybinarylist :
        list (samples) of numpy-arrays (window, class)
    indices :
        indices of the slice of data (samples) to be taken

    Returns
    -------
    x_setlist : list
        list (windows across samples) of numpy-arrays (time, variable)
    y_setlist: list
        list (windows across samples) of numpy-arrays (class, )
    """
    tty = str(type(indices))
    # or statement in next line is to account for python2 and python3
    # difference
    if tty == "<class 'slice'>" or tty == "<type 'slice'>":
        x_setlist = [X for Xlist in Xlists[indices] for X in Xlist]
        y_setlist = [y for ylist in ybinarylists[indices] for y in ylist]
    else:
        x_setlist = [X for X in Xlists[indices]]
        y_setlist = [y for y in ybinarylists[indices]]
    return x_setlist, y_setlist


ACTIVITIES_MAP = {
    0: 'no_activity',
    1: 'lying',
    2: 'sitting',
    3: 'standing',
    4: 'walking',
    5: 'running',
    6: 'cycling',
    7: 'nordic_walking',
    9: 'watching_tv',
    10: 'computer_work',
    11: 'car_driving',
    12: 'ascending_stairs',
    13: 'descending_stairs',
    16: 'vaccuum_cleaning',
    17: 'ironing',
    18: 'folding_laundry',
    19: 'house_cleaning',
    20: 'playing_soccer',
    24: 'rope_jumping'
}


## MAIN Code

targetdir = 'F:/media/windows-share/PAMAP2/'
outdatapath = 'F:/Vikranth/Arm/Datasets/Pamap_test/'
columns_to_use = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z',
                  'hand_gyroscope_x', 'hand_gyroscope_y', 'hand_gyroscope_z', 
                  'hand_magnometer_x', 'hand_magnometer_y', 'hand_magnometer_z',
                  'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',
                  'ankle_gyroscope_x', 'ankle_gyroscope_y', 'ankle_gyroscope_z',
                  'ankle_magnometer_x', 'ankle_magnometer_y', 'ankle_magnometer_z',
                  'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z',
                  'chest_gyroscope_x', 'chest_gyroscope_y', 'chest_gyroscope_z',
                  'chest_magnometer_x', 'chest_magnometer_y', 'chest_magnometer_z']
exclude_activities = [0]

datadir = os.path.join(targetdir, 'PAMAP2_Dataset', 'Protocol')
filenames = os.listdir(datadir)
filenames.sort()
print('Start pre-processing all ' + str(len(filenames)) + ' files...')
# load the files and put them in a list of pandas dataframes:
datasets = [pd.read_csv(os.path.join(datadir, fn), header=None, sep=' ')
            for fn in filenames]
datasets = addheader(datasets)  # add headers to the datasets
# Interpolate dataset to get same sample rate between channels
datasets_filled = [d.interpolate() for d in datasets]
print('loaded the dataset')

class_labels, nr_classes, mapclasses = map_class(datasets_filled, exclude_activities=[0])

#Create input (x) and output (y) sets
xall = [np.array(data[columns_to_use]) for data in datasets_filled]
yall = [np.array(data.activityID) for data in datasets_filled]

xylists = [split_activities(y, x, exclude_activities) for x, y in zip(xall, yall)]
Xlists, ylists = zip(*xylists)
ybinarylists = [transform_y(y, mapclasses, nr_classes) for y in ylists]

frame_length = int(512)
step = 100

# Split in train, test and val
train_range_1 = slice(0, 4)
train_range_2 = slice(5, len(datasets_filled))
x_testlist, y_testlist = split_data(Xlists, ybinarylists, indices=4)
x_trainlist_1, y_trainlist_1 = split_data(Xlists, ybinarylists,
                                      indices=train_range_1)
x_trainlist_2, y_trainlist_2 = split_data(Xlists, ybinarylists,
                                      indices=train_range_2)
# Take sliding-window frames, target is label of last time step,
# and store as numpy file
x_train_1, y_train_1 = sliding_window(frame_length, step, x_trainlist_1,
                                  y_trainlist_1)
x_train_2, y_train_2 = sliding_window(frame_length, step, x_trainlist_2,
                                  y_trainlist_2)
x_train = x_train_1 + x_train_2
y_train = y_train_1 + y_train_2

x_test, y_test = sliding_window(frame_length, step, x_testlist,
                                  y_testlist)

numpify_and_store(x_train, y_train, X_name='X_train', y_name='y_train',
                outdatapath=outdatapath, shuffle=True)
numpify_and_store(x_test, y_test, X_name='X_test', y_name='y_test',
                outdatapath=outdatapath, shuffle=False)

print('Processed data succesfully stored in ' + outdatapath)

