import h5py
import numpy as np
import torch
import torch.nn.functional as F
from vdeep4 import deep
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
import time


dfile = h5py.File('../transfer/54/ku_mi_smt.h5', 'r')



def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)




def evaluate(model, x, y):
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
            num += correct
    return num * 1.0 / x.shape[0]




device = torch.device('cuda')
time_start = time.time()
acc = np.zeros([54])
for n in range(54):
    X, Y = get_data(n + 1)
    print(X.shape, Y.shape)

    print(X.shape)

    T_X_train, T_Y_train = X[:300], Y[:300]
    T_X_valid, T_Y_valid = X[:300], Y[:300]
    T_X_test, T_Y_test = X[300:], Y[300:]

    T_X_train = T_X_train.transpose([0, 2, 1])
    T_X_valid = T_X_valid.transpose([0, 2, 1])
    T_X_test = T_X_test.transpose([0, 2, 1])

    T_X_train = torch.tensor(np.expand_dims(T_X_train, axis=1), dtype=torch.float32)
    T_X_valid = torch.tensor(np.expand_dims(T_X_valid, axis=1), dtype=torch.float32)
    T_X_test = torch.tensor(np.expand_dims(T_X_test, axis=1), dtype=torch.float32)

    T_Y_train = torch.tensor(T_Y_train, dtype=torch.long)
    T_Y_valid = torch.tensor(T_Y_valid, dtype=torch.long)
    T_Y_test = torch.tensor(T_Y_test, dtype=torch.long)

    T_X_train, T_Y_train = T_X_train.to(device), T_Y_train.to(device)
    T_X_valid, T_Y_valid = T_X_valid.to(device), T_Y_valid.to(device)
    T_X_test, T_Y_test = T_X_test.to(device), T_Y_test.to(device)

    train_set = TensorDataset(T_X_train, T_Y_train)
    train_loader = DataLoader(dataset=train_set, batch_size=40, shuffle=True)
    valid_set = TensorDataset(T_X_valid, T_Y_valid)
    valid_loader = DataLoader(dataset=valid_set, batch_size=40, shuffle=True)

    model = deep().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1*0.0001, weight_decay=0.5*0.001)



    # model.train()
    for epoch in tqdm(range(60)):
        model.train()
        for i, (train_fea, train_lab) in enumerate(train_loader):
            out = model(train_fea)
            _, pred = torch.max(out, dim=1)
            loss = F.nll_loss(out, train_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print('\n Train, epoch: {}, i: {}, loss: {}'.format(epoch, i, loss))

        model.eval()
        for data, valid in valid_loader:
            output = model(data)
            valid_loss = F.nll_loss(output, valid)
            # print('valid_loss:{}'.format(valid_loss))



        test_acc = evaluate(model, T_X_test, T_Y_test)
        print('\n Test: acc: {},sub:{}'.format(test_acc,n+1))
    test_acc = evaluate(model, T_X_test, T_Y_test)
    acc[n] = test_acc
time_end = time.time()
print('time',time_end-time_start)
np.save('specific_acc_54',acc)

