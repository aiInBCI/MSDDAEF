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

"""
目标域52人具体迁移过程
多源域投票
"""

# subjs = [35, 47, 46, 37, 13, 27, 12, 32, 4, 40, 19, 41, 18, 42, 34, 7,
#          49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
#          52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]

subjs_sor = [35, 47, 46, 37, 13, 27, 12, 32,  4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]

#dfile = h5py.File('./target/ku_mi_smt.h5', 'r')
dfile = h5py.File('./54/ku_mi_smt.h5', 'r')
#dfile_sor = h5py.File('../pretrain/source/KU_mi_smt.h5', 'r')
dfile_sor = h5py.File('./adjust_52/KU_mi_smt.h5', 'r')
outpath = '../pretrain/pretrain_model_54/'
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
    return num * 1.0 / x.shape[0], indices.cpu().detach().numpy()  # 导出预测标签用于投票


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

accu = np.zeros([54,53])    #前54是单源的准确率 最后一个是投票后的准确率
ts_loss = np.zeros([54,52])
# fold = 51
start_time = time.time()
for n in range(0,54):
    if n == 27:
        time.sleep(3600)
    X, Y = get_data(n+1)
    print(X.shape, Y.shape)

    print(X.shape)

    T_X_train, T_Y_train = X[:300], Y[:300]   #前面的300个是训练，后面的100个是测试。
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

    pred_voting = np.zeros([52, 100])  # 52个分类器的预测标签矩阵

    tra_loss = np.zeros([52, 100])
    tar_loss = np.zeros([52, 100])
    sor_loss = np.zeros([52, 100])
    tst_acc = np.zeros([52, 100])

    for ind, sor in enumerate(subjs_sor):

        model = Transfer_Net(outpath=outpath, cv=5)  # 做实验根据预训练结果手动改cv的值
        model.to(device)
        # optimizer = torch.optim.SGD([
        #     {'params': model.base_network.parameters()},
        #     {'params': model.bottleneck_layer.parameters(), 'lr': 10 * 0.00001},
        #     {'params': model.classifier_layer.parameters(), 'lr': 10 * 0.00001},
        # ], lr=0.00001, momentum=0.2)

        optimizer = torch.optim.SGD([
            {'params': model.base_network.parameters()},
            {'params': model.bottleneck_layer.parameters(), 'lr': 10 * 0.001},
            {'params': model.classifier_layer.parameters(), 'lr': 10 * 0.001},
        ], lr=0.001, momentum=0.2, weight_decay=5 * 0.0001)



        # optimizer = torch.optim.AdamW([
        #     {'params': model.base_network.parameters()},
        #     {'params': model.bottleneck_layer.parameters(), 'lr': 0.00001},
        #     {'params': model.classifier_layer.parameters(), 'lr': 0.00001},
        # ], lr=0.000001 )  #

        S_X_train, S_Y_train = get_data_sor(sor)
        S_X_train = S_X_train.transpose([0, 2, 1])

        S_X_train = torch.tensor(np.expand_dims(S_X_train, axis=1), dtype=torch.float32)
        S_Y_train = torch.tensor(S_Y_train, dtype=torch.long)

        S_X_train, S_Y_train = S_X_train.to(device), S_Y_train.to(device)

        source_set = TensorDataset(S_X_train, S_Y_train)
        source_loader = DataLoader(dataset=source_set, batch_size=40, shuffle=True)

        transfer_l = np.zeros([120,8])

        for epoch in tqdm(range(120)):
            model.train()
            source_loader_iter, target_loader_iter = iter(source_loader), iter(target_loader)
            for i in range(8):   #300个样本  batch size是40  40*7+20=300
                data_source, label_source = source_loader_iter.next()  # 按顺序读取迭代器中的值
                data_target, label_target = target_loader_iter.next()
                # print(data_source.shape, label_source.shape, data_target.shape, label_target.shape)hezhiqian

                optimizer.zero_grad()
                label_source_pred, label_target_pred, transfer_loss = model(data_source, data_target)

                # transfer_l[epoch,i] = transfer_loss

                ts_loss[n,ind] = ts_loss[n,ind]+transfer_loss
                # print(ts_loss)
                source_clf_loss = F.nll_loss(label_source_pred, label_source)
                target_clf_loss = F.nll_loss(label_target_pred, label_target)
                # print('******************************target_clf_loss*******************',target_clf_loss)
                # print('******************************source_clf_loss*******************', source_clf_loss)
                # print('******************************transfer_loss*******************', transfer_loss)
                #loss = target_clf_loss + source_clf_loss + 0.1* transfer_loss
                loss = target_clf_loss + 0.1 * transfer_loss
                loss.backward()
                optimizer.step()


                # # 5.4 修改
                # tra_loss[ind, epoch] = loss.cpu().detach().numpy()
                # sor_loss[ind, epoch] = source_clf_loss.cpu().detach().numpy()
                # tra_loss[ind, epoch] = target_clf_loss.cpu().detach().numpy()

            # test
            test_acc, _ = evaluate(model, T_X_test, T_Y_test)
            # tst_acc[ind, epoch] = test_acc
            print('\n Test: acc: {}'.format(test_acc))

        # plt.plot(np.sum(transfer_l, axis=1))
        # plt.show()

        accu[n,ind], pred_voting[ind, :] = evaluate(model, T_X_test, T_Y_test)  # 生产的预测标签放到投票矩阵的对应位置
        # path = pjoin('./coralloss_model/sub{}'.format(n), 'model_cv{}.pt'.format(ind))
        # torch.save(model.state_dict(), path)
        # path = pjoin('./cosine_model/sub{}'.format(n), 'model_cv{}.pt'.format(ind))
        # torch.save(model.state_dict(), path)
        path = pjoin('./mmd_model/sub{}'.format(n), 'model_cv{}.pt'.format(ind))
        torch.save(model.state_dict(), path)
        print(pred_voting[ind, :])


    print(pred_voting)
    pred = np.zeros(100)  # 最终投票后的标签
    for i in range(100):
        tmp = pred_voting[:, i].astype('int64').T  # 取100个测试样本中每一个样本53个分类器对其标签的预测值
        tmp_counts = np.bincount(tmp)  # 统计不同值的总个数
        pred[i] = np.argmax(tmp_counts)  # 取总数最多的对应序号

    print(pred)
    print(Y[300:])
    print('acc: {}'.format(np.sum(Y[300:] == pred) / 100))  # 对比样本真实标签计算出准确率
    accu[n,52] = np.sum(Y[300:] == pred) / 100
    print(accu)
    np.save('100_acc_mmm_mmd_new', accu)
    np.save('100_acc_mmm_mmd_weight', ts_loss)
end_time = time.time()
print('time',end_time-start_time)
np.save('100_acc_mmm_mmd_new',accu)
np.save('100_acc_mmm_mmd_weight',ts_loss)

# 5.4 改，写入
# writer = pd.ExcelWriter('./res.xlsx')
# pd.DataFrame(tra_loss).to_excel(writer, sheet_name='tra_loss')
# pd.DataFrame(sor_loss).to_excel(writer, sheet_name='sor_loss')
# pd.DataFrame(tra_loss).to_excel(writer, sheet_name='tra_loss')
# pd.DataFrame(tst_acc).to_excel(writer, sheet_name='tst_acc')
# pd.DataFrame(pred_voting).to_excel(writer, sheet_name='pred_voting')
# pd.DataFrame(T_Y_test.cpu().numpy()).to_excel(writer, sheet_name='true_label')
# writer.close()
