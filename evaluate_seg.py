import cv2
import os
import numpy as np

# add mIOU

# evaluate the accuracy between pred and gt, using F1-score
def f1(pred, gt):
    pred = cv2.imread(pred,cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt,cv2.IMREAD_GRAYSCALE)
    # resize
    h,w = gt.shape
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    # normalization
    np.putmask(gt, gt>0, 1.0)
    np.putmask(pred, pred>=0.5, 1.0)
    np.putmask(pred, pred<0.5, 0)
    # indices
    total = h*w
    result = (gt*2 + pred).reshape(-1,)
    tn,fp,fn,tp = np.bincount(result,minlength=4)
    # outliers
    if tp+fp==0:
        if fn==0:
            precise = 1
        else:
            precise = 0.5
    else:
        precise = np.divide(tp,tp+fp)

    if tp+fn==0:
        if fp==0:
            recall = 1
        else:
            recall = 0.5
    else:
        recall = np.divide(tp,tp+fn)

    if tp+fn<100:
        balance_acc = 0.5
    else:
        balance_acc = (np.divide(tp,tp+fn) + np.divide(tn,tn+fp))/2

    if precise+recall==0:
        F1 = 0
    else:
        beta2 = 2**2
        F1 = np.divide((1+beta2)*precise*recall,beta2*precise+recall)
    if fp+fn+tp==0:
        IOU = 0
    else:
        IOU = tp/(tp+fp+fn)

    return precise, recall, F1, balance_acc, IOU, (tp+tn)/total


def f1_batch(pred_path, gt_path):
    path_lst = (pred_path, gt_path)
    PATH = []
    for path in path_lst: #rank files
        files = os.listdir(path)
        files.sort()
        PATH.append(files)

    precise = []
    recall = []
    F1 = []
    Bacc = []
    mIOU = []
    total = []
    for pred,gt in zip(PATH[0],PATH[1]):
        p, r, f, b, iou, t = f1(os.path.join(pred_path,pred),os.path.join(gt_path,gt))
        precise.append(p)
        recall.append(r)
        F1.append(f)
        Bacc.append(b)
        mIOU.append(iou)
        total.append(t)

    return np.mean(precise), np.mean(recall), np.mean(F1), np.mean(Bacc), np.mean(mIOU), np.mean(total)


if __name__ == "__main__":
    #pred_path = ['./lanenet-tensorflow/results_audi/','./']
    pred_path=[]
#    for version in ('v1','v2','v3','v4','v5','v6v3','v6v4'):
    for version in ('v2',''):
    #    pred_path.append('./results/'+version+'/hardtest_383/val_lane_segmentation/')
        pred_path.append('./results/'+version+'/audi200_re/val_lane_segmentation/')
        #pred_path.append('./results/'+version+'/thelastone_pc2/val_lane_segmentation/')
        #pred_path.append('./results/'+version+'/audi200/val_lane_segmentation/')
    gt_path = './data/audi_dataset/test/lane_label/'
    for model in pred_path:
        p,r,f,b,iou,t = f1_batch(model, gt_path)
        #print(model,'\nprecise: %.2f\nrecall: %.2f\nf2-score: %.2f\nb-acc: %.2f' % (p*100, r*100, f*100, b*100))
        print(model,'\n%.2f\n%.2f\n%.2f\n%.2f\n%.2f\n%.2f' % (p*100, r*100, f*100, b*100, iou*100, t*100))


