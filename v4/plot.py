# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:57:05 2020

@author: kerui
"""
import matplotlib.pyplot as plt

def plot(mode, epoch_num, road_acc_list = None, lane_acc_list = None, road_loss_list = None, lane_loss_list = None):
    assert mode in ['road_lane_train', 'road_lane_val', 'lane_train', 'lane_val'], \
    "unsupported mode: {}".format(mode)
    
    if mode == 'road_lane_train':
        # loss
        x1 = range(epoch_num)
        x2 = range(epoch_num)
        y1 = road_loss_list
        y2 = lane_loss_list
        plt.subplot(2,2,1)
        plt.plot(x1, y1, 'o-')
        plt.title('Train road loss vs epoches')
        plt.xlabel('Epoches')
        plt.ylabel('Train road loss')
        
        plt.subplot(2,2,2)
        plt.plot(x2, y2, '.-')
        plt.title('Train lane loss vs epoches')
        plt.ylabel('Train lane loss')
        plt.xlabel('Epoches')
        
        # acc
        x1 = range(epoch_num)
        x2 = range(epoch_num)
        y1 = road_acc_list
        y2 = lane_acc_list
        plt.subplot(2,2,3)
        plt.plot(x1, y1, 'o-')
        plt.title('Train road acc vs epoches')
        plt.xlabel('Epoches')
        plt.ylabel('Train road acc')
        
        plt.subplot(2,2,4)
        plt.plot(x2, y2, '.-')
        plt.title('Train lane acc vs epoches')
        plt.ylabel('Train lane acc')
        plt.xlabel('Epoches')
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)
        plt.savefig('train.png')
        
    elif mode == 'road_lane_val':
        # acc
        x1 = range(epoch_num)
        x2 = range(epoch_num)
        y1 = road_acc_list
        y2 = lane_acc_list
        plt.subplot(1,2,1)
        plt.plot(x1, y1, 'o-')
        plt.title('Val road acc vs epoches')
        plt.xlabel('Epoches')
        plt.ylabel('Val road acc')
        
        plt.subplot(1,2,2)
        plt.plot(x2, y2, '.-')
        plt.title('Val lane acc vs epoches')
        plt.ylabel('Val lane acc')
        plt.xlabel('Epoches')
        plt.subplots_adjust(wspace=0.5)
        plt.savefig('val_acc.png')
        
    elif mode == 'lane_train':
        # loss
        x1 = range(epoch_num)
        y1 = lane_loss_list
        plt.subplot(1,2,1)
        plt.plot(x1, y1, 'o-')
        plt.title('Train lane loss vs epoches')
        plt.xlabel('Epoches')
        plt.ylabel('Train lane loss')
        
        # acc
        x1 = range(epoch_num)
        y1 = lane_acc_list
        plt.subplot(1,2,2)
        plt.plot(x1, y1, 'o-')
        plt.title('Train lane acc vs epoches')
        plt.xlabel('Epoches')
        plt.ylabel('Train lane acc')
        plt.subplots_adjust(wspace=0.5)
        plt.savefig('log/train.png')
        
    elif mode == 'lane_val':
        # acc
        x1 = range(epoch_num)
        y1 = lane_acc_list
        plt.close() # 清除已有的图
        plt.plot(x1, y1, 'o-')
        plt.title('Val lane acc vs epoches')
        plt.xlabel('Epoches')
        plt.ylabel('Val lane acc')
        plt.savefig('log/val_acc.png')
        
