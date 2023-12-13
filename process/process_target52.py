import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
from scipy.io import loadmat
import h5py
from scipy.signal import butter, lfilter
from scipy import signal

# data alignment for GIST
src = './GIST/
out = './cd_GIST'

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

def read_data(num):
    # onset of MI
    onset = [1024, 4608	,8192	,11776,	15360,	18944	,22528	,26112	,29696	,33280	,36864	,40448,	44032	,47616	,51200	,54784	,58368	,61952	,65536	,69120	,72704	,76288	,79872	,83456	,87040	,90624	,94208	,97792	,101376	,104960	,108544	,112128	,115712	,119296	,122880	,126464	,130048	,133632	,137216	,140800	,144384	,147968	,151552	,155136	,158720	,162304	,165888	,169472	,173056	,176640	,180224	,183808	,187392	,190976	,194560	,198144	,201728	,205312	,208896	,212480	,216064	,219648	,223232	,226816	,230400	,233984	,237568	,241152	,244736	,248320	,251904	,255488	,259072	,262656	,266240	,269824	,273408	,276992	,280576	,284160	,287744	,291328	,294912	,298496	,302080	,305664	,309248	,312832	,316416	,320000	,323584	,327168	,330752	,334336,	337920	,341504	,345088	,348672	,352256	,355840]
    ele = [9, 10, 11, 46, 45, 44, 14, 13, 12, 48, 49, 50, 51, 17, 18, 19, 32, 56, 55, 54]  # index of 20 channels
    filename = 's{:02d}.mat'.format(num)
    filepath = pjoin(src, filename)
    raw = loadmat(filepath)
# the first 50 left/right hand MI trials for training
    left = np.zeros([50, 64, 1536])
    right = np.zeros([50, 64, 1536])
    for i in range(50):
        left[i] = raw['eeg']['imagery_left'][0][0][0:64, onset[i] - 1 :onset[i] - 1 + 1536]
        right[i] = raw['eeg']['imagery_right'][0][0][0:64, onset[i] - 1 :onset[i]-1 + 1536]
# CAR
    left_ = np.zeros([50, 64, 1536])
    right_ = np.zeros([50, 64, 1536])
    for i in range(50):
        for j in range(1536):
            left_[i, :, j] = left[i, :, j] - np.mean(left[i, :, j])
            right_[i, :, j] = right[i, :, j] - np.mean(right[i, :, j])
# select 20 channels
    _left = np.zeros([50, 20, 1536])
    _right = np.zeros([50, 20, 1536])
    for i in range(50):
        for j in range(20):
            _left[i][j] = left_[i][ele[j]-1, :]
            _right[i][j] = right_[i][ele[j] - 1, :]
# bandpass filter 
    buffer_left = butter_bandpass_filter(_left, 8, 30, 512)
    buffer_right = butter_bandpass_filter(_right, 8, 30, 512)
# Select a period of 0.5-2.5 seconds and slide backward four times in a step of 50 to expand the number of samples
    left__ = np.zeros([50, 4, 20, 1024])
    right__ = np.zeros([50, 4, 20, 1024])
    for i in range(50):
        for j in range(4):
            for k in range(20):
                left__[i][j][k] = buffer_left[i][k][256 + 50 * j:1280 + 50 * j]
                right__[i][j][k] = buffer_right[i][k][256 + 50 * j:1280 + 50 * j]

    left__ = left__.reshape((200, 20, 1024))
    right__ = right__.reshape((200, 20, 1024))

    train = np.concatenate((left__, right__), axis=0)      #400*20*1024
# the last 50 left/right hand MI trials for evaluation
    left_test = np.zeros([50, 64, 1536])
    right_test = np.zeros([50, 64, 1536])
    for i in range(50):
        left_test[i] = raw['eeg']['imagery_left'][0][0][0:64, onset[i+50] - 1:onset[i+50] - 1 + 1536]
        right_test[i] = raw['eeg']['imagery_right'][0][0][0:64, onset[i+50] - 1:onset[i+50] - 1 + 1536]
# CAR
    left__test = np.zeros([50, 64, 1536])
    right__test = np.zeros([50, 64, 1536])
    for i in range(50):
        for j in range(1536):
            left__test[i, :, j] = left_test[i, :, j] - np.mean(left_test[i, :, j])
            right__test[i, :, j] = right_test[i, :, j] - np.mean(right_test[i, :, j])
# select 20 channels
    test_left = np.zeros([50, 20, 1536])
    test_right = np.zeros([50, 20, 1536])
    for i in range(50):
        for j in range(20):
            test_left[i][j] = left__test[i][ele[j] - 1, :]
            test_right[i][j] = right__test[i][ele[j] - 1, :]
# bandpass filter
    buffer_left_test = butter_bandpass_filter(test_left, 8, 30, 512)
    buffer_right_test = butter_bandpass_filter(test_right, 8, 30, 512)

    test__left = np.zeros([50, 20, 1024])
    test__right = np.zeros([50, 20, 1024])
# Select a period of 0.5-2.5 seconds
    for i in range(50):
        for j in range(20):
            test__left[i][j] = buffer_left_test[i][j][256:1280]
            test__right[i][j] = buffer_right_test[i][j][256:1280]

    test = np.concatenate((test__left, test__right), axis=0)    #100*20*1024

    X = np.concatenate((train, test), axis=0)     
# Amplitude adjustment
    X = X/33.674670216208554
# label，‘1’：left hand，‘0’：right hand
    Y_L_T = np.ones(200)
    Y_R_T = np.zeros(200)
    Y_train = np.concatenate((Y_L_T,Y_R_T), axis=0)   #400

    Y_L_test = np.ones(50)
    Y_R_test = np.zeros(50)
    Y_test = np.concatenate((Y_L_test,Y_R_test), axis=0)  #100

    Y = np.concatenate((Y_train,Y_test),axis=0)     

    return X, Y


with h5py.File(pjoin(out, 'ku_mi_smt.h5'), 'w') as f:
    for num in tqdm(range(1, 53)):
        X, Y = read_data(num)
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        f.create_dataset('s' + str(num) + '/X', data=X)   
        f.create_dataset('s' + str(num) + '/Y', data=Y)   

