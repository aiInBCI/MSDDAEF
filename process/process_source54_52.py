from scipy.signal import resample
from os.path import join as pjoin
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from scipy.signal import butter, lfilter
from scipy import signal

# 源域54人数据集处理，对应目标域52人数据集。
# src = 'data54'
src = r'D:\苗团队\苗老师\杨忠\深度迁移 exp\54数据集'
out = '../波幅调整参数计算/54'

def butter_bandpass(lowcut, highcut, fs, order=5):     #带通滤波
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
# 选取20导联
    ele = [8, 33, 9, 10, 34, 11, 35, 13, 36,14, 37, 15, 38, 18, 39, 19, 40, 20, 41, 21]     #20个导联的索引
    X1 = np.moveaxis(raw['EEG_MI_train']['smt'][0][0], 0, -1)  # 100, 62, 4000 读取第一个三维向量，交换轴
    X2 = np.moveaxis(raw['EEG_MI_test']['smt'][0][0], 0, -1)
# 选取0.5-2.5秒数据
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

# 降采样
    _X1_ = resample(X1_, 1024, axis=2)
    _X2_ = resample(X2_, 1024, axis=2)
    X = np.concatenate((_X1_, _X2_), axis=0)
# 带通滤波
    X = butter_bandpass_filter(X, 8, 30, 512)
# 对应label
    Y1 = (raw['EEG_MI_train']['y_dec'][0][0][0] - 1)   #100
    Y2 = (raw['EEG_MI_test']['y_dec'][0][0][0] - 1)    #100
    Y = np.concatenate((Y1, Y2), axis=0)               #200
    return X, Y
# 读取54人数据处理并保存
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

###Checked 2022.12.25 mmm