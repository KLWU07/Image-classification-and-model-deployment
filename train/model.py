import torch.nn as nn
import torch

class AlexNet(nn.Module):
    #类ALEXNET继承nn.module这个父类
    def __init__(self, num_classes=1000, init_weights=False):
        #通过初始化函数，定义网络在正向传播过程中需要使用的层结构
        #num_classes是指输出的图片种类个数，init_weights=False意味着不定义模型中的初始权重
        super(AlexNet, self).__init__()
        #nn.Sequential模块，可以将一系列的层结构进行打包，组合成一个新的结构，
        # 将专门用于提取图像特征的结构的名称取为features
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # input[3, 224, 224]  output[48, 55, 55]
            #第一个卷积层，彩色图片深度为3，卷积核个数位48，卷积核大小11，步长4，padding2
            nn.ReLU(inplace=True),
            #使用Relu激活函数时要将设置inplace=True
            #使用relu激活函数，f(x)=max(0,x)，
            #相比sigmod函数与tanh函数有以下几个优点
            # 1)克服梯度消失的问题
            # 2）加快训练速度
            # 注：正因为克服了梯度消失问题，训练才会快
            # 缺点：
            # 1）输入负数，则完全不激活，ReLU函数死掉。
            # 2）ReLU函数输出要么是0，要么是正数，也就是ReLU函数不是以0为中心的函数

            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),#步长默认为1，当步长为1时不用设置# output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        #将三个全连接层打包成一个新的模块，分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),#随机将一半的节点失活，默认为0.5
            nn.Linear(128 * 6 * 6, 2048),#将特征矩阵展平，128*6*6最后输出的长*宽*高，2048为全连接层节点个数
            nn.ReLU(inplace=True),#Relu激活函数
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),#全连接层2的输入为全连接层1的输出2048，全连接层2的节点个数2048
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),#全连接层3的输入为全连接层2的输出2048
            #全连接层最后的输出就是图片类别的个数
        )
        if init_weights:
            self._initialize_weights()
        #是否初始化权重，如果初始化函数中的init_weights=Ture,就会进入到初始化权重的函数


    def forward(self, x):
        #forward正向传播过程，x为输入的变量
        x = self.features(x)
        #将训练样本输入features
        x = torch.flatten(x, start_dim=1)
        #将输入的变量进行展平从深度高度宽度三个维度进行展开，索引从1开始，展成一个一维向量
        x = self.classifier(x)
        #将展平后的数据输入到分类器中（三个全连接层组成的）
        return x#最后的输出为图片类别

    #初始化权重的函数
    def _initialize_weights(self):
        for m in self.modules():#遍历self.modules这样一个模块，该模块继承自它的父类nn.module，该模块会迭代每一个层次
            if isinstance(m, nn.Conv2d):#如果该层次是一个卷积层，就会使用kaiming_normal_这样一个方法初始化权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)#如果偏值不为空的话，就用0对权重进行初始化
            elif isinstance(m, nn.Linear):#如果该层次是一个全连接层，就用normal进行初始化
                nn.init.normal_(m.weight, 0, 0.01)#正态分布对权重进行赋值，均值为0，方差为0.01
                nn.init.constant_(m.bias, 0)#设置全连接层的偏值为0
