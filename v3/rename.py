# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:46:14 2020

@author: kerui
"""
import glob

dir_name1 = r'E:/Study/THpractice/code/data/testing/test_velodyne_depth/'
dir_name2 = r'E:\Study\THpractice\code\results\mode=sparse+photo.w1=0.1.w2=0.1.input=gd.resnet34.criterion=l2.lr=1e-05.bs=16.wd=0.pretrained=False.jitter=0.1.time=2020-03-03@17-26\test_completion_output_test'
path1 = r'E:/Study/THpractice/code/data/testing/test_velodyne_depth/*.png'
#path2 = r'E:\Study\THpractice\code\results\mode=sparse+photo.w1=0.1.w2=0.1.input=gd.resnet34.criterion=l2.lr=1e-05.bs=16.wd=0.pretrained=False.jitter=0.1.time=2020-03-03@17-26\test_completion_output_train/*.png'
path2 = r'E:\Study\THpractice\code\results\mode=sparse+photo.w1=0.1.w2=0.1.input=gd.resnet34.criterion=l2.lr=1e-05.bs=16.wd=0.pretrained=False.jitter=0.1.time=2020-03-03@16-23\test_completion_output_test\*.png'
path1 = glob.glob(path1)
path2 = glob.glob(path2)
#for x in zip(path1, path2):
#    print(x)
#    
#for x in zip(sorted(path1), path2)[10]:
#    print(x)
#    
#for x in zip(sorted(path1)[:10], path2[:10]):
#    print(x)
    
import os
os.path.split('1111/333.png')
for x1, x2 in zip(sorted(path1), path2):
    _, filename = os.path.split(x1)
    dirname, _ = os.path.split(x2)
    os.rename(x2, os.path.join(dirname, filename))
    
