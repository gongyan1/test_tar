# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:48:32 2020

@author: kerui
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from dataloaders.completion_segmentation_loader_new_ import load_calib, oheight, owidth, input_options, KittiDepth
from completion_segmentation_model import DepthCompletionFrontNet
from metrics import AverageMeter, Result
import criteria
import completion_segmentation_helper
from inverse_warp import Intrinsics, homography_from

import numpy as np
import plot

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w',
                    '--workers',
                    default=2,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=11,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('--image_height',
                    default=352,
                    type=int,
                    help='height of image for train (default: 80)')

parser.add_argument('--image_width',
                    default=1216,
                    type=int,
                    help='width of image for train (default: 80)')

parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-4,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='../data',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='gd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('-l',
                    '--layers',
                    type=int,
                    default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained',
                    action="store_true",
                    help='use ImageNet pre-trained weights')
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument(
    '--rank-metric',
    type=str,
    default='rmse',
    choices=[m for m in dir(Result()) if not m.startswith('_')],
    help='metrics for which best result is sbatch_datacted')
parser.add_argument(
    '-m',
    '--train-mode',
    type=str,
    default="dense",
    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
    help='dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('--cpu', action="store_true", help='run on cpu')

args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('..', 'results/v5')
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# define loss functions

'''类别权重
        1:99 -> 0.01 0.99
        1:9 -> 0.1 0.9
        1:8 -> 1/9 8/9
        1:4 -> 0.2 0.8
'''
class_weight = torch.tensor([0.1, 0.9])
# 语义分割loss
road_criterion = nn.NLLLoss2d()
lane_criterion = nn.NLLLoss2d(weight=class_weight.cuda())

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
    if cuda:
        kitti_intrinsics = kitti_intrinsics.cuda()

# 计算准确率
def acc(prediction, label, num_class=2):
    bs, c, h, w = prediction.size()
    values, indices = prediction.max(1)
    
    acc_total = 0
    acc_lane = 0

    label_ = label.numpy()
    BS = bs
    for i in range(bs):
        prediction = indices[i].view(h,w).numpy()
        label = label_[i]
        # 混淆矩阵
        mask = (label>=0) & (label<num_class)
        result_ = num_class * label[mask].astype('int') + prediction[mask]
        count = np.bincount(result_, minlength=num_class**2)
        acc_total += (count[0]+count[3])/count.sum()
        if count[2:].sum()>100:	#only images with obvious lanes will be counted	
            acc_lane += count[3]/count[2:].sum()
        else:
            BS -= 1

    acc_total /= bs
    if BS:
        acc_lane /= BS    #only images with obvious lanes will be counted
    else:
        acc_lane = torch.tensor([0.5])
    
    return acc_total, acc_lane
    

def iterate(mode, args, loader, model, optimizer, logger, best_acc, epoch):
    start_val = time.clock()
    nonsense = 0
    acc_sum = 0
    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = completion_segmentation_helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0
    lane_acc_lst = []
    lane_loss_lst = []
    total_acc_lst = []
    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }
        
        # 道路分割的label
        road_label = batch_data[
            'road_label'] if mode != 'test_road_lane_segmentation' else None
                
        # 车道线分割的label
        lane_label = batch_data[
            'lane_label'] if mode != 'test_road_lane_segmentation' else None
                
        data_time = time.time() - start

        start = time.time()
        
        if mode == 'val':
            with torch.no_grad(): # 设置torch.no_grad()，在val时不计算梯度，可以节省显存
                pred = model(batch_data)
        else:
            pred = model(batch_data)
            
        lane_pred = pred
        start_ = time.clock() # 不计入时间
        if mode == 'train':
            # 语义分割loss
            #road_loss = road_criterion(road_pred, road_label.long())
            if epoch==0:
                class_weight = torch.tensor([0.5, 0.5])
            else:
                lane_pred_w = lane_pred.data.cpu()
                bs, c, h, w = lane_pred_w.size()
                value_w, index_w = lane_pred_w.max(1)
                LPW = 0
                for i in range(bs):
                    lpw = index_w[i].view(h,w).numpy()
                    LPW += (np.count_nonzero(lpw)/lpw.size)
                LPW /= bs
                class_weight = torch.tensor([LPW,1-LPW])
                #print('class_weight: ',class_weight)
            lane_criterion = nn.NLLLoss2d(weight=class_weight.cuda())
            lane_loss = lane_criterion(lane_pred, lane_label.long())
            lane_loss_lst.append(lane_loss.item())

            # 损失
            #loss = road_loss + lane_loss
            loss = lane_loss
            
            #print('lane loss {}'.format(lane_loss.data.cpu()))
            
            # 准确率
            #road_acc = acc(road_pred.data.cpu(), road_label.cpu())
            total_acc, lane_acc = acc(lane_pred.data.cpu(), lane_label.cpu())
            lane_acc_lst.append(lane_acc.item())
            total_acc_lst.append(total_acc.item())
            
            #print('total acc {}'.format(total_acc), 'lane acc {}'.format(lane_acc))
            #print('\n-------------------------epoch '+str(epoch)+'-----------------------------\n')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif mode == 'val':
            # 准确率
            #road_acc = acc(road_pred.data.cpu(), road_label.cpu())
            total_acc, lane_acc = acc(lane_pred.data.cpu(), lane_label.cpu())
            lane_acc_lst.append(lane_acc.item())
            total_acc_lst.append(total_acc.item())
            #print('total acc {}'.format(total_acc), 'lane acc {}'.format(lane_acc))
            #print('\n------------------------epoch '+str(epoch)+'------------------------------\n')
            
            #accuracy = (road_acc+lane_acc)/2
            accuracy = lane_acc
            
            acc_sum += accuracy

        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            # 保存预测结果为图片
            logger.conditional_save_pred(mode, i, pred, epoch)
        nonsense += (time.clock()-start_)

    print('total cost time: ', time.clock()-start_val-nonsense)
    if mode=='train':
        lane_loss_mean = np.array(lane_loss_lst).mean()
        lane_acc_mean = np.array(lane_acc_lst).mean()
        total_acc_mean = np.array(total_acc_lst).mean()
        print('lane loss {}'.format(lane_loss_mean), 'lane acc {}'.format(lane_acc_mean), 'total acc {}'.format(total_acc_mean))
    elif mode=='val':
        lane_acc_mean = np.array(lane_acc_lst).mean()
        total_acc_mean = np.array(total_acc_lst).mean()   
        print('lane acc {}'.format(lane_acc_mean), 'total acc {}'.format(total_acc_mean))  
    print('\n-------------------------epoch '+str(epoch)+'-----------------------------\n')
    acc_avg = acc_sum/len(loader)
    
    is_best = (acc_avg>best_acc)
            
    # 每一个epoch保存一次信息
#    avg = logger.conditional_save_info(mode, average_meter, epoch)
#    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
#    if is_best and not (mode == "train"):
#        # 验证时，保存最好的预测结果为图片
#        logger.save_img_comparison_as_best(mode, epoch)
#    logger.conditional_summarize(mode, avg, is_best)
    
    if mode == 'train':
        return acc_avg, is_best, lane_loss_mean, lane_acc_mean, total_acc_mean

    elif mode == 'val':
        return acc_avg, is_best, lane_acc_mean, total_acc_mean


def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            args = checkpoint['args']
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    elif args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = DepthCompletionFrontNet(args).to(device)
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(model_named_params,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    print("completed.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = completion_segmentation_helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate("val", args, val_loader, model, None, logger,
                                  checkpoint['epoch'])
        return

    # main loop
    print("=> starting main loop ...")
    best_acc = 0
    # 记录loss, acc
    train_lane_loss_list = []
    train_lane_acc_list = []
    train_total_acc_list = []
    
    val_lane_acc_list = []
    val_total_acc_list = []
    
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting EVALUATING epoch {} ..".format(epoch))
        
        start = time.clock()
        result, is_best, val_lane_acc, val_total_acc = iterate("val", args, val_loader, model, None, logger, best_acc, epoch)  # evaluate on validation set
        print('Validation Time Cost: ', (time.clock()-start)/2300)
        val_lane_acc_list.append(val_lane_acc)
        val_total_acc_list.append(val_total_acc)
        
        completion_segmentation_helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)

    #plot.plot('lane_train', args.epochs, lane_acc_list = train_lane_acc_list, lane_loss_list=train_lane_loss_list)
    #plot.plot('lane_val', args.epochs, lane_acc_list = val_lane_acc_list)


        with open('log/val_lane_acc_v5_'+time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))+'.txt', 'w+') as f:
            val_lane_acc_list_w = [str(line) for line in val_lane_acc_list]
            f.writelines('\n'.join(val_lane_acc_list_w))
        with open('log/val_total_acc_v5_'+time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))+'.txt', 'w+') as f:
            val_total_acc_list_w = [str(line) for line in val_total_acc_list]
            f.writelines('\n'.join(val_total_acc_list_w))


if __name__ == '__main__':
    main()
