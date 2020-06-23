import numpy as np
import cv2 as cv
import glob
import os
path = r'G:\data\train_results\um_road_000000.png'
dir_path = r'G:\data\train_results\*.png'
save_dir_path = r'G:\data\train_label_results'
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

files = np.array(sorted(glob.glob(dir_path)))

print(files)
for image in files[:56]:
    image_name = image.split('\\')[-1]
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    #cv.imshow('原图', image)
    THRESH = 100
    ret, binary = cv.threshold(image, THRESH, 1, cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
    #cv.imshow('binary', binary)
    cv.imwrite(save_dir_path + '/' + image_name, binary)
    #cv.waitKey(0)

