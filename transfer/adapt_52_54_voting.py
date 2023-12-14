import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from model52 import Transfer_Net
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
import time

#  dual stream model and voting

subjs_sor = [35, 47, 46, 37, 13, 27, 12, 32,  4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]

dfile = h5py.File('../process/cd_openBMI/ku_mi_smt.h5', 'r')  # target domain data path
dfile_sor = h5py.File('../process/cd_GIST/KU_mi_smt.h5', 'r')  # source domain data path
outpath = '../pretrain/pretrain_model_54/'  # pretrained model datapath
device = torch.device('cuda')


def evaluate(model, x, y):
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=100)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            out = model.predict(test_input)
            _, indices = torch.max(out, dim=1)
            correct = torch.sum(indices == test_lab)
            num += correct.cpu().numpy()
    return num * 1.0 / x.shape[0], indices.cpu().detach().numpy()  


def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)

def get_data_sor(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile_sor[pjoin(dpath, 'X')]
    Y = dfile_sor[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)

accu = np.zeros([54,53])    # save the single source and multi-source classification accuracy
ts_loss = np.zeros([54,52]) # save the single source loss
# start_time = time.time()
for n in range(0,54):
    X, Y = get_data(n+1)
    print(X.shape, Y.shape)
    print(X.shape)

    T_X_train, T_Y_train = X[:300], Y[:300]   # training and testing sets
    T_X_test, T_Y_test = X[300:], Y[300:]

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

    pred_voting = np.zeros([52, 100])  # Prediction label matrix for 52 source classifiers

    tra_loss = np.zeros([52, 100])
    tar_loss = np.zeros([52, 100])
    sor_loss = np.zeros([52, 100])
    tst_acc = np.zeros([52, 100])

    for ind, sor in enumerate(subjs_sor):

        model = Transfer_Net(outpath=outpath, cv=5)  # load pretrained model
        model.to(device)

        optimizer = torch.optim.SGD([
            {'params': model.base_network.parameters()},
            {'params': model.bottleneck_layer.parameters(), 'lr': 10 * 0.001},
            {'params': model.classifier_layer.parameters(), 'lr': 10 * 0.001},
        ], lr=0.001, momentum=0.2, weight_decay=5 * 0.0001)

        S_X_train, S_Y_train = get_data_sor(sor)
        S_X_train = S_X_train.transpose([0, 2, 1])

        S_X_train = torch.tensor(np.expand_dims(S_X_train, axis=1), dtype=torch.float32)
        S_Y_train = torch.tensor(S_Y_train, dtype=torch.long)

        S_X_train, S_Y_train = S_X_train.to(device), S_Y_train.to(device)

        source_set = TensorDataset(S_X_train, S_Y_train)
        source_loader = DataLoader(dataset=source_set, batch_size=40, shuffle=True)


        for epoch in tqdm(range(120)):
            model.train()
            source_loader_iter, target_loader_iter = iter(source_loader), iter(target_loader)
            for i in range(8):   
                data_source, label_source = source_loader_iter.next()  
                data_target, label_target = target_loader_iter.next()

                optimizer.zero_grad()
                label_source_pred, label_target_pred, transfer_loss = model(data_source, data_target)


                ts_loss[n,ind] = ts_loss[n,ind]+transfer_loss
                source_clf_loss = F.nll_loss(label_source_pred, label_source)
                target_clf_loss = F.nll_loss(label_target_pred, label_target)
                # print('******************************target_clf_loss*******************',target_clf_loss)
                # print('******************************source_clf_loss*******************', source_clf_loss)
                # print('******************************transfer_loss*******************', transfer_loss)
                loss = target_clf_loss + 0.1 * transfer_loss
                loss.backward()
                optimizer.step()


            # test
            test_acc, _ = evaluate(model, T_X_test, T_Y_test)
            print('\n Test: acc: {}'.format(test_acc))

        accu[n,ind], pred_voting[ind, :] = evaluate(model, T_X_test, T_Y_test)  # The produced prediction label is placed in the corresponding position in the voting matrix

        # save new models    
        # path = pjoin('./coralloss_model/sub{}'.format(n), 'model_cv{}.pt'.format(ind))
        # torch.save(model.state_dict(), path)
             
        print(pred_voting[ind, :])


    print(pred_voting)
    pred = np.zeros(100)  # labels after the voting
    for i in range(100):
        tmp = pred_voting[:, i].astype('int64').T  
        tmp_counts = np.bincount(tmp)  
        pred[i] = np.argmax(tmp_counts)  

    print(pred)
    print(Y[300:])
    print('acc: {}'.format(np.sum(Y[300:] == pred) / 100))  
    accu[n,52] = np.sum(Y[300:] == pred) / 100
    print(accu)
    
# end_time = time.time()
# print('time',end_time-start_time)
# save the classification accuracy and transfer loss
np.save('100_acc_coral', accu)
np.save('100_acc_coral_weight', ts_loss)


