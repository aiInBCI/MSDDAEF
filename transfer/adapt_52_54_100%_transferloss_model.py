import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transfer_weight_deep import Transfer_Net
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
outpath2 = './mmd_model_scheme3'
device = torch.device('cuda')

ts_loss = np.load('100_acc_mmm_mmd_weight_scheme3.npy')
# print(ts_loss/8)
def evaluate(model, x, y):
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=200)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            out = model.predict(test_input)
            _, indices = torch.max(out, dim=1)
            correct = torch.sum(indices == test_lab)
            num += correct.cpu().numpy()
    return num * 1.0 / x.shape[0], out.cpu().detach().numpy()  # 导出预测标签用于投票


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

label_s3 = np.zeros([54,200])
start_time = time.time()
acc = np.zeros(54)
for n in range(0,54):
    weight_pred = np.zeros([52,200,2])
    weight = np.zeros([52])

    X, Y = get_data(n + 1)
    print(X.shape, Y.shape)

    print(X.shape)

    T_X_train, T_Y_train = X[:200], Y[:200]  # 前面的300个是训练，后面的100个是测试。
    T_X_test, T_Y_test = X[200:], Y[200:]

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
    for ind in range(52):
        model = Transfer_Net(outpath=outpath, cv=5)
        checkpoint = torch.load(pjoin(outpath2, 'sub{}/model_cv{}.pt'.format(n,ind)))  # 加载模型参数
        model.load_state_dict(checkpoint)  # 给模型参数赋值
        model.to(device)

        test_acc, weight_pred[ind] = evaluate(model, T_X_test, T_Y_test)
        print('test_acc',test_acc)
        # print(weight_pred)
        # print(ts_loss[n,ind]/np.sum(ts_loss[n]))
        # weight_pred[ind] = weight_pred[ind] * ((ts_loss[n,ind]/np.sum(ts_loss[n])))
    weight_sort = np.sort(ts_loss[n,:])
    for w in range(52):  ##############################################
        for e in range(52):
            if ts_loss[n,w] == weight_sort[e]:
                weight[w] = 52-e
                weight_sort[e] = 0
                break
    print('***********weight*****************',weight)
    for p in range(52):
        weight_pred[p] = weight_pred[p] * (weight[p] / np.sum(weight))
    weight_pred = np.mean(weight_pred,axis=0)
    # print('weight_pred',weight_pred.shape)
    weight_pred = torch.tensor(weight_pred)
    weight_pred = weight_pred.to(device)
    _, indices = torch.max(weight_pred, dim=1)
    label_s3[n,:] = indices.cpu().detach().numpy()

    # print(indices)
    correct = torch.sum(indices == T_Y_test)
    test_acc = correct/200
    acc[n] = test_acc
    print('test_acc',test_acc)
print('***********************54acc*******************',np.mean(acc))
np.save('100_acc_mmm__weight_model_scheme3',acc)
np.save('100_acc_mmm__weight_scheme3_label',label_s3)

# 5.4 改，写入
# writer = pd.ExcelWriter('./res.xlsx')
# pd.DataFrame(tra_loss).to_excel(writer, sheet_name='tra_loss')
# pd.DataFrame(sor_loss).to_excel(writer, sheet_name='sor_loss')
# pd.DataFrame(tra_loss).to_excel(writer, sheet_name='tra_loss')
# pd.DataFrame(tst_acc).to_excel(writer, sheet_name='tst_acc')
# pd.DataFrame(pred_voting).to_excel(writer, sheet_name='pred_voting')
# pd.DataFrame(T_Y_test.cpu().numpy()).to_excel(writer, sheet_name='true_label')
# writer.close()
