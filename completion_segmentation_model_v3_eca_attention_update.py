
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:57:24 2020

@author: kerui
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import math
import cv2

print_model = False # 是否打印网络结构

# 初始化参数
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# 卷积-> 批标准化-> relu
def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

# 转置卷积-> 批标准化-> relu
def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers
                    
'''
前融合：在第一层ResBlock前融合
    

'''     
# 点云+RGB作为输入，只有车道线分割这一分支      
class DepthCompletionFrontNet(nn.Module):
    def __init__(self, args, k_size=3):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            args.layers)
        super(DepthCompletionFrontNet, self).__init__()
        self.modality = args.input

        # 点云:原始 + KNN补全 + 高度
        if 'd' in self.modality:
            #channels = 64 * 3 // len(self.modality)
            channels = 32
            self.conv1_d = conv_bn_relu(3,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            
        # rgb
        if 'rgb' in self.modality:
            #channels = 64 * 3 // len(self.modality)
            channels = 32
            self.conv1_img = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
            
        # gray
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        # 加载resnet预训练模型
        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
            
        # encoding layers
        # sk_attention


        #self.maxpool = pretrained_model._modules['maxpool']
        # resnet预训练模型的第一个块
        self.conv2 = pretrained_model._modules['layer1']
        # resnet预训练模型的第二个块
        self.conv3 = pretrained_model._modules['layer2']
        # resnet预训练模型的第三个块
        #self.conv4 = pretrained_model._modules['layer3']
        self.conv4 = conv_bn_relu(in_channels=128,
                                         out_channels=256,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
        # resnet预训练模型的第四个块
        #self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels

        # 两个分支共用的两层解码层
        kernel_size = 3
        stride = 2
        self.conv_connnet = conv_bn_relu(in_channels=256,
                                         out_channels=128,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        
        # decoding layers for lane segmentation
        self.convt2_ = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1_ = convt_bn_relu(in_channels=64 + 64,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf_ = conv_bn_relu(in_channels=64 + 64,
                                   out_channels=2, # 二分类
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)
        self.softmax_lane = self.softmax = nn.LogSoftmax(dim=1)


        # sk_attention-0512
        # 这个就是把池化得到的一维向量经过全连接层降维再经过一个全连接层回复原大小增加非线性特征
        # planes = 64
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_fc1 = nn.Conv2d(planes, planes//2, 1, bias=False)
        # # self.bn_fc1 = nn.BatchNorm2d(planes//2)
        # self.conv_fc2 = nn.Conv2d(planes//2, planes, 1, bias=False)
        # self.D1 = planes//2


        # eca-net : 先定义结构再调用，k需要人工挑选
        # self.avg_pool_eca = nn.AdaptiveAvgPool2d(1)
        # self.conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.sigmoid_eca = nn.Sigmoid()


    def eca_layer(self,x,gamma = 2,b=1):
        # eca-net
        # 原理：通过GPA（全局平均池化）转为1*1*C的向量，再通过1维conv进行权重更新

        N,C,H,W = x.size()
        t = int(abs((math.log(C,2)+b)/gamma))
        k_size = t if t%2 else t+1
        # k_size = 3
        avg_pool_eca = nn.AdaptiveAvgPool2d(1)
        # conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=int(k_size // 2), bias=False)
        sigmoid_eca = nn.Sigmoid()
        y = avg_pool_eca(x)
        # print(x)
        # print(y)
        y = y.cpu()
        y = conv1d_eca(y.squeeze(-1).transpose(-1, -2))
        # print("dasdasada")
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = sigmoid_eca(y)
        y = y.cuda()

        # 将 y 变成和 x 一样的 shape
        return x * y.expand_as(x)


    def forward(self, x):
        if print_model:
            print("\n-------------------------encoder-------------------------\n")
        
        # first layer
        if 'd' in self.modality:
            if print_model:
                print("\n    input shape of reflectance: {}".format(x['d'].shape))
            conv1_d = self.conv1_d(x['pc'])
            if print_model:
                print("\n    first layer 3x3 conv_bn_relu for reflectance --> output shape: {}".format(conv1_d.shape))
        if 'rgb' in self.modality:
            if print_model:
                print("\n    input shape of rgb: {}".format(x['rgb'].shape))
            conv1_img = self.conv1_img(x['rgb'])
            if print_model:
                print("\n    first layer 3x3 conv_bn_relu for rgb --> output shape: {}".format(conv1_img.shape))
        elif 'g' in self.modality:
            if print_model:
                print("\n    input shape of gray image: {}".format(x['g'].shape))
            conv1_img = self.conv1_img(x['g'])
            if print_model:
                print("\n    first layer 3x3 conv_bn_relu for Gray Image --> output shape: {}".format(conv1_img.shape))

        if self.modality == 'rgbd' or self.modality == 'gd':

            # sk_attention-0512
            # conv1 = torch.cat((conv1_d, conv1_img), 1)
            # d1 = torch.cat((conv1_d, conv1_img), 1)
            #
            # d = self.avg_pool(d1)
            # d = self.conv_fc1(d)
            # # d = self.bn_fc1(d)
            # d = F.relu(d)
            # d = self.conv_fc2(d)
            # d = torch.unsqueeze(d, 1).view(-1, 2, self.D1, 1, 1)
            # d = F.softmax(d, 1)
            # # 0 1 如果交换会不会有效果
            # x1_cloud = conv1_d * d[:, 0, :, :, :].squeeze(1)
            # x1 = conv1_img * d[:, 1, :, :, :].squeeze(1)


            # eca-attention：对应已定义的eca结构，需人工挑选k
            # print(conv1_d.shape,conv1_img.shape)

            # 逐层融合:间隔
            c_count = conv1_d.shape[1]
            x_cat = conv1_d[:,0,:,:]
            for i in range(c_count):
                if i==0:
                    # now i=0
                    x_img = conv1_img[:,i,:,:]
                    x_cat = torch.cat((x_cat,x_img),1)
                else:
                    x_d = conv1_d[:,i,:,:]
                    x_img = conv1_img[:,i,:,:]
                    x_cat = torch.cat((x_cat,x_d),1)
                    x_cat = torch.cat((x_cat,x_img),1)
            x = x_cat
            x = x.view(1,64,-1,1216)

            # x = torch.cat((conv1_d, conv1_img), 1)
            # print(x.shape)
            # torch.Size([1, 64, 352, 1216])

            # x: input features with shape [b, c, h, w]
            # b,c,h,w = x.size()
            #
            # # feature descriptor on the global spatial information
            # y = self.avg_pool_eca(x)
            #
            # # Two different branches of ECA module
            # y = self.conv1d_eca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            #
            # # Multi-scale information fusion
            # y = self.sigmoid_eca(y)
            #
            # conv1 = x * y.expand_as(x)
            conv1 = self.eca_layer(x)
            if print_model:
                print("\n    concat the feature of first layer  --> output shape: {}".format(conv1.shape))

        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        # encoder
        conv2 = self.conv2(conv1)
        if print_model:
            print("\n    ResNet Block{} output shape: {}".format(1, conv2.shape))
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        if print_model:
            print("\n    ResNet Block{} output shape: {}".format(2, conv3.shape))
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        if print_model:
            print("\n    ResNet Block{} output shape: {}".format(3, conv4.shape))

        if print_model:
            print("\n-------------------------decoder for reflectance completion-------------------------\n")
            
        # 两个分支共用的两层解码层
        convt4 = self.conv_connnet(conv4)
        y_common = torch.cat((convt4, conv4), 1)
        if print_model:
            print("\n    skip connection from ResNet Block{}".format(3))

        convt3 = self.convt3(y_common)
        if print_model:
            print("\n    3x3 TransposeConv_bn_relu {} --> output shape: {}".format(3, convt3.shape))
        y_common = torch.cat((convt3, conv3), 1)
        if print_model:
            print("\n    skip connection from ResNet Block{}".format(2))

        # decoder for lane segmentation
        convt2_ = self.convt2_(y_common)
        if print_model:
            print("\n    3x3 TransposeConv_bn_relu {} --> output shape: {}".format(4, convt2_.shape))
        y_ = torch.cat((convt2_, conv2), 1)
        if print_model: 
            print("\n    skip connection from ResNet Block{}".format(1))
        
        convt1_ = self.convt1_(y_)
        if print_model:
            print("\n    3x3 TransposeConv_bn_relu {} --> output shape: {}".format(5, convt1_.shape))
        y_ = torch.cat((convt1_, conv1), 1)
        if print_model:
            print("\n    skip connection from the concat feature of first layer")
        
        y_ = self.convtf_(y_)
        if print_model:
            print("\n    the end layer 1x1 conv_bn_relu --> output shape: {}".format(y_.shape))
        
        lane = self.softmax_lane(y_)
        if print_model:
            print("\n    softmax for road segmentation --> output shape: {}".format(lane.shape))

        return lane
