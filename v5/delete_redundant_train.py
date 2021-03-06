# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:37:44 2020

@author: kerui
"""

'''
删除训练集中多余的图片
'''
import os

# RGB图像路径
depth_root = r'E:\Study\THpractice\code\data\training\train_velodyne_reflectance'
RGB = os.listdir(r'E:\Study\THpractice\code\data\training\train_image_2_lane')
# 深度图路径
depth = os.listdir(r'E:\Study\THpractice\code\data\training\train_velodyne_reflectance')

# 两个文件夹中共同含有的文件
common_file = [file for file in depth if file not in RGB]



# 删除depth中多余的文件

for file in common_file:
    common_file_path = os.path.join(depth_root, file)
    if os.path.exists(common_file_path):
        os.remove(common_file_path)
        print('delete file: %s' % common_file_path)
    else:
        print('no such file: %s' % common_file_path)