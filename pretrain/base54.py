import h5py
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn.functional as F
from vdeep4 import deep
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
import time

'''
pretraining when openBMI is the target domain 
'''
subjs = [35, 47, 46, 37, 13, 27, 12, 32, 4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]

dfile = h5py.File('../process/cd_GIST/ku_mi_smt.h5', 'r')     # data path of GIST after alignment
outpath = './pretrain_model_54/'  # data path of pretrained models

def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)


def get_multi_data(subjs):   
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def evaluate(model, x, y):   
    # print(x.shape, y.shape)
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)   
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


cv_set = np.array(subjs)
print(cv_set)
print(cv_set.shape)
kf = KFold(n_splits=6)
device = torch.device('cuda')

patience = 20  # When the validation set loss does not significantly decrease in 20 consecutive training epoches, the training is stopped

cv_loss = np.ones([6])  # Minimum validation loss for saving each fold
result = pd.DataFrame(columns=('cv_index', 'test_acc', 'loss'))  # Training process record
train_loss2 = np.zeros([6,100])
test_loss2 = np.zeros([6,100])
train_accuracy2 = np.zeros([6,100])
test_accuracy2 = np.zeros([6,100])

start_time = time.time()

for cv_index, (train_index, test_index) in enumerate(kf.split(cv_set)):  
    tra_acc = np.zeros([100])
    tra_loss = np.zeros([100])
    tst_acc = np.zeros([100])
    tst_loss = np.zeros([100])
    # print('train_index',train_index)
    # print('test_index',test_index)


    train_subjs = cv_set[train_index]
    test_subjs = cv_set[test_index]

    X_train, Y_train = get_multi_data(train_subjs)
    X_test, Y_test = get_multi_data(test_subjs)
    X_train = X_train.transpose([0, 2, 1])
    X_test = X_test.transpose([0, 2, 1])

    print(Y_train.shape, Y_test.shape)
    print(np.sum(Y_train == 0) / Y_train.shape[0], np.sum(Y_test == 0) / Y_test.shape[0])

    X_train = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
    X_test = torch.tensor(np.expand_dims(X_test, axis=1), dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    train_set = TensorDataset(X_train, Y_train)
    test_set = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)  
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    model = deep().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1 * 0.0001, weight_decay=0.5 * 0.001)

    train_losses = []  # A list of training losses for each epoch
    test_losses = []  #  A list of evaluation losses for each epoch
    train_loss_avg = []  # Averaging training losses by batch
    test_loss_avg = []  # veraging evaluation losses by batch
    early_stop = EarlyStopping(patience, delta=0.0001, path=pjoin(outpath, 'model_cv{}.pt'.format(cv_index)),
                               verbose=True)

    for epoch in tqdm(range(100)):
        model.train()
        t = 0
        for i, (train_fea, train_lab) in enumerate(train_loader):
            out = model(train_fea)
            _, pred = torch.max(out, dim=1)
            t += (pred == train_lab).sum().cpu().numpy()
            loss = F.nll_loss(out, train_lab)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:   
                print('\n Train, epoch: {}, i: {}, loss: {}'.format(epoch, i, loss))

        e = 0
        model.eval()
        for data, target in test_loader:
            output = model(data)
            _, pred = torch.max(output, dim=1)
            e += (pred == target).sum().cpu().numpy()
            loss = F.nll_loss(output, target)
            test_losses.append(loss.item())

        train_loss = np.average(train_losses)
        test_loss = np.average(test_losses)
        train_loss_avg.append(train_loss)
        test_loss_avg.append(test_loss)

        tra_loss[epoch] = train_loss
        tst_loss[epoch] = test_loss
        tra_acc[epoch] = t / 21500
        tst_acc[epoch] = e / 4500

        if test_loss < cv_loss[cv_index]:
            cv_loss[cv_index] = test_loss  # Iterate to get the minimum validation loss for each fold

        train_losses = [] # update
        test_losses = []  # update

        test_acc = evaluate(model, X_test, Y_test)
        print('\n Test: acc: {}'.format(test_acc))

        res = {'cv_index': cv_index, 'test_acc': test_acc.item(), 'loss': test_loss}
        result = result.append(res, ignore_index=True) 

        early_stop(test_loss, model)
        if early_stop.early_stop:
            print('Early stopping')
            break  # Determine whether the early stop condition is met and save the model
    train_loss2[cv_index] = tra_loss
    test_loss2[cv_index] = tst_loss
    train_accuracy2[cv_index] = tra_acc
    test_accuracy2[cv_index] = tst_acc

   
end_time = time.time()
print('cost %f second' % (end_time -  start_time))

min_index = np.argmin(cv_loss, axis=0)  # index of six-fold cross validation minimum validation loss 
result.to_csv(pjoin(outpath, 'result_cv{}.csv'.format(min_index)), index=False)
