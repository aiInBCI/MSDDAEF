import torch
import torch.nn as nn
from torch.nn import functional as F
# 目标域为52人数据集的预训练模型
# 模型参考deep4，修改分类层，二维卷积替换为2层全连接
class deep(nn.Module):
    def __init__(self):
        super(deep, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(10, 1), stride=(1, 1))
        self.conv1_1 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 20), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.max1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(10, 1), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.max2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(10, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.max3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(10, 1), stride=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.max4 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc = nn.Sequential(
            nn.Linear(in_features=1600, out_features=512)
        )

        # self.conv_classifier = nn.Conv2d(in_channels=200, out_channels=2, kernel_size=(7, 1), stride=(1, 1))

        self.conv_classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=2),
        )

    def forward(self, x):  # (64, 1, 1000, 62)
        out = self.conv1(x)  # (64 , 25, 991, 62)
        out = self.conv1_1(out)  # (64 , 25, 991, 1)
        out = F.elu(self.bn1(out))
        out = F.dropout(self.max1(out), p=0.5, training=self.training)

        out = self.conv2(out)
        out = F.elu(self.bn2(out))
        out = F.dropout(self.max2(out), p=0.5, training=self.training)

        out = self.conv3(out)
        out = F.elu(self.bn3(out))
        out = F.dropout(self.max3(out), p=0.5, training=self.training)

        out = self.conv4(out)
        out = F.elu(self.bn4(out))
        out = self.max4(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.conv_classifier(out)
        out = nn.LogSoftmax(dim=1)(out)  # 对批量样本中的每个样本取概率

        return out


if __name__ == '__main__':
    model = deep()
    # 查看网络结构
    print(model)
    for param in model.named_parameters():
        print(param[0])