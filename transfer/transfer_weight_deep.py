import torch.nn as nn
from Coral import CORAL
from cosine_distance import cosine_distance
from euclidean_distance import euclidean_squared_distance

import mmd
from vdeep4 import deep
import torch
from os.path import join as pjoin
from torch.nn import functional as F

# 52人数据集的迁移模型
# 添加瓶颈层后的新模型
class basemodel(nn.Module):  # vdeep4去掉最后一层
    def __init__(self, outpath, cv):
        super(basemodel, self).__init__()
        base_model = deep()
        # print(pjoin(outpath, 'model_f{}_cv{}.pt'.format(fold, cv)))
        checkpoint = torch.load(pjoin(outpath, 'model_cv{}.pt'.format(cv)))  # 加载模型参数
        base_model.load_state_dict(checkpoint)  # 给模型参数赋值

        self.conv1 = base_model.conv1
        self.conv1_1 = base_model.conv1_1
        self.bn1 = base_model.bn1
        self.max1 = base_model.max1

        self.conv2 = base_model.conv2
        self.bn2 = base_model.bn2
        self.max2 = base_model.max2

        self.conv3 = base_model.conv3
        self.bn3 = base_model.bn3
        self.max3 = base_model.max3

        self.conv4 = base_model.conv4
        self.bn4 = base_model.bn4
        self.max4 = base_model.max4

        self.fc = base_model.fc  # 不取vdeep4最后一层，即classfier层

    def forward(self, x):
        out = self.conv1(x)  # (64 , 25, 991, 62)
        out = self.conv1_1(out) # (64 , 25, 991, 1)
        out = F.elu(self.bn1(out))
        # dropout的p设置为0，相当于冻结，不会随机失活
        out = F.dropout(self.max1(out), p=0, training=self.training)

        out = self.conv2(out)
        out = F.elu(self.bn2(out))
        out = F.dropout(self.max2(out), p=0, training=self.training)

        out = self.conv3(out)
        out = F.elu(self.bn3(out))
        out = F.dropout(self.max3(out), p=0, training=self.training)

        out = self.conv4(out)
        out = F.elu(self.bn4(out))
        out = self.max4(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)  # 64 256
        return out


class Transfer_Net(nn.Module):
    def __init__(self, outpath='', cv=0, transfer_loss='edu', use_bottleneck=True):
        super(Transfer_Net, self).__init__()

        self.base_network = basemodel(outpath, cv)
        self.use_bottleneck = use_bottleneck  # 下面的if判断
        self.transfer_loss = transfer_loss   # 距离

        # 瓶颈层：全连接，BN，ELU激活函数，Dropout
        bottleneck_list = [nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ELU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        # 分类层：全连接，ELU激活函数，Dropout， 全连接，LogSoftmax
        classifier_layer_list = [nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.5), nn.Linear(256, 2), nn.Softmax(dim=1)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target)
        source_clf = self.classifier_layer(source)
        target_clf = self.classifier_layer(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
        return source_clf, target_clf, transfer_loss

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral
        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss
        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        elif adapt_loss == 'cosine_distance':
            loss = cosine_distance(X, Y)
        elif adapt_loss == 'edu':
            loss = euclidean_squared_distance(X, Y)
        else:
            loss = 0
        return loss