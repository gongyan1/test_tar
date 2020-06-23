# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:12:38 2020

@author: kerui
"""

import struct
import os
if __name__ == '__main__':
    filepath = r'S:\迅雷下载\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data\0000000000.bin'
    binfile = open(filepath, 'rb')
    size = os.path.getsize(filepath) #获得文件大小
    for i in range(800):
        data = binfile.read(1) # 每次输出一个字节
        num = struct.unpack("B", data)
        print(num)
        
    print(size)
    binfile.close()
    
