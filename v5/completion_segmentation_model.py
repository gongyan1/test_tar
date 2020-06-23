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
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            args.layers)
        super(DepthCompletionFrontNet, self).__init__()
        self.modality = args.input

        # 点云:原始 + KNN补全 + 高度
        if 'd' in self.modality:
            #channels = 64 * 3 // len(self.modality)
            channels = 64
            self.conv1_d = conv_bn_relu(3,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            
        # rgb
        if 'rgb' in self.modality:
            #channels = 64 * 3 // len(self.modality)
            channels = 64
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
        
        #self.maxpool = pretrained_model._modules['maxpool']
        
        # 分支1
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
        
        # 分支2
        # resnet预训练模型的第一个块
        self.conv2_ = pretrained_model._modules['layer1']
        # resnet预训练模型的第二个块
        self.conv3_ = pretrained_model._modules['layer2']
        # resnet预训练模型的第三个块
        #self.conv4_ = pretrained_model._modules['layer3']
        self.conv4_ = conv_bn_relu(in_channels=128,
                                         out_channels=256,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
        
        # resnet预训练模型的第四个块
        #self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory
            
        num_channels = 256
        '''
        self.conv5 = conv_bn_relu(num_channels,
                                  256,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        
        self.conv5_ = conv_bn_relu(num_channels,
                                  256,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        '''

        # 前两层解码层
        kernel_size = 3
        stride = 2
        '''
        self.convt4 = convt_bn_relu(in_channels=256,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        '''
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
        '''
        self.convt4_ = convt_bn_relu(in_channels=256,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        '''
        self.conv_connnet_ = conv_bn_relu(in_channels=256,
                                         out_channels=128,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.convt3_ = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        
        # 融合层
        self.conv1x1 = conv_bn_relu(in_channels=64 + 64,
                                    out_channels=64,
                                    kernel_size=1)
        
        # decoding layers for lane segmentation
        self.convt2_ = convt_bn_relu(in_channels=(128 + 128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1_ = convt_bn_relu(in_channels=64 + 64 + 64,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf_ = conv_bn_relu(in_channels=64 + 64 + 64,
                                   out_channels=2, # 二分类
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)
        self.softmax_lane = self.softmax = nn.LogSoftmax(dim=1)
        
    def make_resnet_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv1_cat = torch.cat((conv1_d, conv1_img), 1)
        # encoder
        # 分支1
        conv2 = self.conv2(conv1_d)
        if print_model:
            print("\n    ResNet Block{} output shape: {}".format(1, conv2.shape))
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        if print_model:
            print("\n    ResNet Block{} output shape: {}".format(2, conv3.shape))
            
        conv4 = self.conv4(conv3)
        
        #conv5 = self.conv5(conv4)
            
        # 分支2
        conv2_ = self.conv2_(conv1_img)
        if print_model:
            print("\n    ResNet Block{} output shape: {}".format(1, conv2.shape))
        conv2_cat = torch.cat((conv2_, conv2), 1)
        
        conv3_ = self.conv3_(conv2_)  # batchsize * ? * 176 * 608
        #conv3_cat = torch.cat((conv3_, conv3), 1)
        
        conv4_ = self.conv4_(conv3_)
        #conv4_cat = torch.cat((conv4_, conv4), 1)
        
        #conv5_ = self.conv5_(conv4_)
        if print_model:
            print("\n    ResNet Block{} output shape: {}".format(2, conv3.shape))

        if print_model:
            print("\n-------------------------decoder for reflectance completion-------------------------\n")
            
        # 前两层解码层
        #convt4 = self.convt4(conv5)
        #convt4_ = self.convt4_(conv5_)
        #if print_model:
        #    print("\n    3x3 TransposeConv_bn_relu {} --> output shape: {}".format(2, convt4.shape))
        convt4 = self.conv_connnet(conv4)
        convt4_ = self.conv_connnet_(conv4_)
        cat = torch.cat((convt4, conv4), 1)
        cat_ = torch.cat((convt4_, conv4_), 1)
        if print_model:
            print("\n    skip connection from ResNet Block{}".format(3))

        convt3 = self.convt3(cat)
        convt3_ = self.convt3(cat_)
        if print_model:
            print("\n    3x3 TransposeConv_bn_relu {} --> output shape: {}".format(3, convt3.shape))
        cat = torch.cat((convt3, convt3_), 1)
        
        conv1x1 = self.conv1x1(cat)
        if print_model:
            print("\n    skip connection from ResNet Block{}".format(2))

        # decoder for lane segmentation
        cat = torch.cat((conv3, conv3_, conv1x1), 1)
        convt2_ = self.convt2_(cat)
        if print_model:
            print("\n    3x3 TransposeConv_bn_relu {} --> output shape: {}".format(4, convt2_.shape))
        cat = torch.cat((convt2_, conv2, conv2_), 1)
        if print_model: 
            print("\n    skip connection from ResNet Block{}".format(1))
        
        convt1_ = self.convt1_(cat)
        if print_model:
            print("\n    3x3 TransposeConv_bn_relu {} --> output shape: {}".format(5, convt1_.shape))
        cat = torch.cat((convt1_, conv1_cat), 1)
        if print_model:
            print("\n    skip connection from the concat feature of first layer")
        
        y_ = self.convtf_(cat)
        if print_model:
            print("\n    the end layer 1x1 conv_bn_relu --> output shape: {}".format(y_.shape))
        
        lane = self.softmax_lane(y_)
        if print_model:
            print("\n    softmax for road segmentation --> output shape: {}".format(lane.shape))

        return lane
