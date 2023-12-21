import h5py
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
import math
from vdeep4 import deep
"""
52人数据集不做adapt
"""

subjs = [35, 47, 46, 37, 13, 27, 12, 32, 4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]



dfile = h5py.File('../transfer/target/ku_mi_smt.h5', 'r')
outpath = '../pretrain/pretrain_model/'
device = torch.device('cuda')


def evaluate(model, x, y):
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=100, shuffle=True)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            out = model(test_input)
            _, indices = torch.max(out, dim=1)
            correct = torch.sum(indices == test_lab)
            num += correct.cpu().numpy()
    return num * 1.0 / x.shape[0]


def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)



for n in range(52):
    X, Y = get_data(n+1)
    # print(X.shape, Y.shape)
    #
    # print(X.shape)

    T_X_train, T_Y_train = X[:400], Y[:400]
    T_X_test, T_Y_test = X[400:], Y[400:]

    T_X_train = T_X_train.transpose([0, 2, 1])
    T_X_test = T_X_test.transpose([0, 2, 1])

    T_X_train = torch.tensor(np.expand_dims(T_X_train, axis=1), dtype=torch.float32)
    T_X_test = torch.tensor(np.expand_dims(T_X_test, axis=1), dtype=torch.float32)
    T_Y_train = torch.tensor(T_Y_train, dtype=torch.long)
    T_Y_test = torch.tensor(T_Y_test, dtype=torch.long)

    T_X_train, T_Y_train = T_X_train.to(device), T_Y_train.to(device)
    T_X_test, T_Y_test = T_X_test.to(device), T_Y_test.to(device)

    target_set = TensorDataset(T_X_train, T_Y_train)
    target_loader = DataLoader(dataset=target_set, batch_size=40, shuffle=True)

    model = deep()
    checkpoint = torch.load(pjoin(outpath, 'model_cv{}.pt'.format(5)))  # 加载模型参数
    model.load_state_dict(checkpoint)  # 给模型参数赋值
    model.to(device)

    acc = evaluate(model, T_X_test, T_Y_test)  # 生产的预测标签放到投票矩阵的对应位置
    # print('acc: {}'.format(acc))
    print(acc)



