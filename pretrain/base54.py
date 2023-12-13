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
针对目标域是54人数据集的预训练模型
'''
subjs = [35, 47, 46, 37, 13, 27, 12, 32, 4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]

#dfile = h5py.File('./source/KU_mi_smt.h5', 'r')
dfile = h5py.File('./adjust_52/ku_mi_smt.h5', 'r')     #20个导联的数据，Data aligment之后的数据
outpath = './pretrain_model_54/'

def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)


def get_multi_data(subjs):   #读取54人的数据，整合。  54*400
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def evaluate(model, x, y):    #进行模型评估
    # print(x.shape, y.shape)
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)   #batch size 为64
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

patience = 20  # 当验证集损失在连续20次训练周期中没有降低，停止训练模型

cv_loss = np.ones([6])  # 交叉验证 最小验证损失
result = pd.DataFrame(columns=('cv_index', 'test_acc', 'loss'))  # 数据框架通过append字段填充 即res
train_loss2 = np.zeros([6,100])
test_loss2 = np.zeros([6,100])
train_accuracy2 = np.zeros([6,100])
test_accuracy2 = np.zeros([6,100])

start_time = time.time()

for cv_index, (train_index, test_index) in enumerate(kf.split(cv_set)):  # cv_index交叉验证次数
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

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)    #batch size为32
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    model = deep().to(device)
    # 优化器adamW改为SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=1 * 0.0001, weight_decay=0.5 * 0.001)

    train_losses = []  # 每一个epoch的训练损失列表，新的epoch会重空开始
    test_losses = []  # 每一个epoch的训练损失列表，新的epoch会重空开始
    train_loss_avg = []  # 训练损失求平均 针对batch求平均
    test_loss_avg = []  # 训练损失求平均
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
            if i % 50 == 0:   #每隔50个batch打印一下
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
            cv_loss[cv_index] = test_loss  # 不断迭代的得到每一次交叉验证的最小验证损失

        train_losses = []
        test_losses = []  # 每一个epoch的验证损失列表，新的epoch会重空开始

        test_acc = evaluate(model, X_test, Y_test)
        print('\n Test: acc: {}'.format(test_acc))

        res = {'cv_index': cv_index, 'test_acc': test_acc.item(), 'loss': test_loss}
        result = result.append(res, ignore_index=True)  # pd创造的数据格式

        early_stop(test_loss, model)
        if early_stop.early_stop:
            print('Early stopping')
            break  # 判断是否达到早停条件并保存模型
    train_loss2[cv_index] = tra_loss
    test_loss2[cv_index] = tst_loss
    train_accuracy2[cv_index] = tra_acc
    test_accuracy2[cv_index] = tst_acc

    # plt.subplot(2, 2, 1)
    # plt.plot(tra_acc)
    # plt.subplot(2, 2, 2)
    # plt.plot(tst_acc)
    # plt.subplot(2, 2, 3)
    # plt.plot(tra_loss)
    # plt.subplot(2, 2, 4)
    # plt.plot(tst_loss)
    # plt.show()
#np.save('train_loss2',train_loss2)
#np.save('test_loss2',test_loss2)
#np.save('train_accuracy2',train_accuracy2)
#np.save('test_accuracy2',test_accuracy2)

end_time = time.time()
print('cost %f second' % (end_time -  start_time))

min_index = np.argmin(cv_loss, axis=0)  # 六折交叉验证最小验证损失序号
result.to_csv(pjoin(outpath, 'result_cv{}.csv'.format(min_index)), index=False)

##Checked 2022.12.26 mmm