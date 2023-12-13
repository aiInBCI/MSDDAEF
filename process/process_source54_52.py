from scipy.signal import resample
from os.path import join as pjoin
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from scipy.signal import butter, lfilter
from scipy import signal

# data alignment for openBMI

src = './openBMI'
out = './cd_openBMI'

def butter_bandpass(lowcut, highcut, fs, order=5):   
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b,a,data,axis=2)
    return y


def get_data(sess, subj):
    filename = 'sess{:02d}_subj{:02d}_EEG_MI.mat'.format(sess, subj)
    filepath = pjoin(src, filename)
    raw = loadmat(filepath)
    ele = [8, 33, 9, 10, 34, 11, 35, 13, 36,14, 37, 15, 38, 18, 39, 19, 40, 20, 41, 21]     # index of 20 channels
    X1 = np.moveaxis(raw['EEG_MI_train']['smt'][0][0], 0, -1)  
    X2 = np.moveaxis(raw['EEG_MI_test']['smt'][0][0], 0, -1)
# select 0.5-2.5s period and 20 channels
    X1_ = np.zeros([100, 20, 2000])
    X2_ = np.zeros([100, 20, 2000])
    for i in range(100):
        for j in range(20):
            X1_[i][j] = X1[i][ele[j]-1, 500:2500]
            X2_[i][j] = X2[i][ele[j]-1, 500:2500]
# CAR
    for i in range(100):
        for j in range(2000):
            X1_[i, :, j] = X1_[i, :, j] - np.mean(X1_[i, :, j])
            X2_[i, :, j] = X2_[i, :, j] - np.mean(X2_[i, :, j])

# downsample
    _X1_ = resample(X1_, 1024, axis=2)
    _X2_ = resample(X2_, 1024, axis=2)
    X = np.concatenate((_X1_, _X2_), axis=0)
# bandpass filter
    X = butter_bandpass_filter(X, 8, 30, 512)
# labels
    Y1 = (raw['EEG_MI_train']['y_dec'][0][0][0] - 1)   #100
    Y2 = (raw['EEG_MI_test']['y_dec'][0][0][0] - 1)    #100
    Y = np.concatenate((Y1, Y2), axis=0)               #200
    return X, Y

with h5py.File(pjoin(out, 'KU_mi_smt.h5'), 'w') as f:
    for subj in tqdm(range(1, 55)):
        X1, Y1 = get_data(1, subj)
        X2, Y2 = get_data(2, subj)
        X = np.concatenate((X1, X2), axis=0)
        X = X.astype(np.float32)
        Y = np.concatenate((Y1, Y2), axis=0)
        Y = Y.astype(np.int64)
        f.create_dataset('s' + str(subj) + '/X', data=X)     #400, 20, 1024
        f.create_dataset('s' + str(subj) + '/Y', data=Y)     #400

