# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:43:00 2020

@author: kerui
"""

import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from dataloaders.pose_estimator import get_pose_pnp
import skimage
import collections

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']


def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[
        0,
        2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[
        1,
        2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K


def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'
            
    if split == "train":
        transform = train_transform
        
        # 1
        glob_pc = os.path.join(
            args.data_folder,
            #'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            'audi_dataset/knn_pc/*.png'
        )
        
        # 2、用于道路分割的label
        glob_road_label = os.path.join(
            args.data_folder,
            #'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            'audi_dataset/road_label/*.png'
        )
        
        # 3、用于车道线分割的label
        glob_lane_label = os.path.join(
            args.data_folder,
            #'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            'audi_dataset/lane_label/*.png'
        )
        
        # 4、
        glob_rgb = os.path.join(
                args.data_folder,
                'audi_dataset/train_image_2_lane/*.png'
            )

        def get_rgb_paths(p):
            rgb_path = '../data/data_rgb/train/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
            
            ps = p.split('\\')
#            pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
#                            ps[-2:-1] + ['data'] + ps[-1:])
            
            pnew = os.path.join(rgb_path, ps[-1])
            
            #print('-------------------1-------------------\n')
            return pnew
    elif split == "val":
        if args.val == "select":
            transform = no_transform
            
            # 1
            glob_pc = os.path.join(
                args.data_folder,
                #'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
                'audi_dataset/test/knn_pc/*.png'
            )
            
            # 2、用于道路分割的label
            glob_road_label = os.path.join(
                args.data_folder,
                #'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
                'audi_dataset/test/road_label/*.png'
            )
            
            # 3、用于车道线分割的label
            glob_lane_label = os.path.join(
                args.data_folder,
                #'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
                'audi_dataset/test/lane_label/*.png'
            )
            
            # 4、
            glob_rgb = os.path.join(
                    args.data_folder,
                    'audi_dataset/test/train_image_2_lane/*.png'
                )       
    
    # 测试, 输入为点云强度和rgb
    elif split == 'test_road_lane_segmentation':
        transform = no_transform
        glob_pc = os.path.join(
            args.data_folder,
            'reflectance/*.png'
        )
        glob_road_label = None
        glob_lane_label = None
        glob_rgb = os.path.join(
            args.data_folder,
            "train_image_2_lane/*.png")

    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_lane_label is not None:
        # train or val-full or val-select
        # 点云：原始+强度+高度
        paths_pc = sorted(glob.glob(glob_pc)) 
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_road_label = sorted(glob.glob(glob_road_label))
        paths_lane_label = sorted(glob.glob(glob_lane_label))
    else:  
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_road_label = [None]*len(paths_rgb)
        paths_lane_label = [None]*len(paths_rgb)
        paths_pc = sorted(glob.glob(glob_pc))
            

    if len(paths_pc) == 0 and len(paths_rgb) == 0 and len(paths_lane_label) == 0 and len(paths_road_label) == 0:
        raise (RuntimeError("Found 0 images under {}".format(paths_lane_label)))
    if len(paths_pc) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_pc) or len(paths_rgb) != len(paths_lane_label):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "pc": paths_pc, "lane_label": paths_lane_label, "road_label": paths_road_label}
    return paths, transform

#oheight, owidth = 352, 1216

def rgb_read(filename, args):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    
    #img_file = img_file.resize((400, 416))
    #img_file = img_file.resize((args.image_height, args.image_width))
    img_file = img_file.resize((args.image_width, args.image_height))
    # 将读入的数据统一成该原始代码所使用的尺寸 oheight, owidth = 352, 1216
    #img_file = img_file.resize((owidth, oheight))
    
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename, args):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    
    #img_file = img_file.resize((400, 416))
    #img_file = img_file.resize((args.image_height, args.image_width))
    img_file = img_file.resize((args.image_width, args.image_height))
    # 将读入的数据统一成该原始代码所使用的尺寸 oheight, owidth = 352, 1216
    #img_file = img_file.resize((owidth, oheight))
    
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


def label_read(filename, args):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    
    #img_file = img_file.resize((400, 416))
    #img_file = img_file.resize((args.image_height, args.image_width))
    img_file = img_file.resize((args.image_width, args.image_height))
    # 将读入的数据统一成该原始代码所使用的尺寸 oheight, owidth = 352, 1216
    #img_file = img_file.resize((owidth, oheight))
    
    label_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(label_png) < 2, \
        "np.max(label_png)={}, path={}".format(np.max(label_png),filename)

    label = label_png.astype(np.float)
    # depth[depth_png == 0] = -1.
    #label = np.expand_dims(label, -1)
    return label


oheight, owidth = 352, 1216


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth

def train_transform(rgb, sparse, target, segmentation_label, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        # 将读入的数据统一改成原始代码所使用的尺寸 oheight, owidth = 352, 1216
        #transforms.BottomCrop((oheight, owidth)),
        #transforms.BottomCrop((args.image_height, args.image_width)),
        #Resize((args.image_height, args.image_width)),
        
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    segmentation_label = transform_geometric(segmentation_label)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target, segmentation_label


def val_transform(rgb, sparse, target, segmentation_label, args):
    transform = transforms.Compose([
        #transforms.BottomCrop((oheight, owidth)),
        #transforms.BottomCrop((args.image_height, args.image_width)),
        # 将读入的数据统一改成原始代码所使用的尺寸 oheight, owidth = 352, 1216
        #transforms.BottomCrop((oheight, owidth)),
        #Resize((args.image_height, args.image_width)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
        
    if segmentation_label is not None:
        segmentation_label = transform(segmentation_label)
    return rgb, sparse, target, segmentation_label


def no_transform(rgb, sparse, target, segmentation_label, args):
    return rgb, sparse, target, segmentation_label


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img

class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        # rgb图像
        rgb = rgb_read(self.paths['rgb'][index], self.args) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        
        # 点云：原始+强度+高度
        point_cloud = rgb_read(self.paths['pc'][index], self.args) if \
            (self.paths['pc'][index] is not None and self.args.use_d) else None
            
        # 车道标签
        road_label = label_read(self.paths['road_label'][index], self.args) if \
            self.paths['road_label'][index] is not None else None
            
        # 车道线标签
        lane_label = label_read(self.paths['lane_label'][index], self.args) if \
            self.paths['lane_label'][index] is not None else None
        
        return rgb, point_cloud, road_label, lane_label

    def __getitem__(self, index):
        rgb, point_cloud, road_label, lane_label = self.__getraw__(index)
        rgb, point_cloud, road_label, lane_label = self.transform(rgb, point_cloud, road_label,lane_label
                                                       ,self.args)

        rgb, gray = handle_gray(rgb, self.args)
        #gray = None
        candidates = {"rgb":rgb, "pc":point_cloud, "road_label":road_label, "lane_label":lane_label ,\
            "g":gray}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['road_label'])
