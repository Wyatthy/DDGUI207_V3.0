# -*- coding: utf-8 -*- #

import os
import cv2
import numpy as np
import torch
import scipy.io as scio
import argparse

from utils.CAM import t2n, HookValues, GradCAM, GradCAMpp, XGradCAM, \
    EigenGradCAM, LayerCAM
from dataset.HRRP_mat import read_project

def vis_cam(args, mat_data, top1 = False, gt_known = True):
    # model
    model = torch.load(f'{args.project_path}/{args.model_name}')
    print(model)
    model.cuda()

    try:
        target_layer = eval(f'model.{args.visualize_layer}')
    except Exception as e:
        print(model)
        raise RuntimeError('layer does not exist', e)
    hookValues = HookValues(target_layer)
    signal = torch.from_numpy(signal).cuda().float()

    # forward
    logits = model(signal)
    logits = torch.sigmoid(logits)

    # backward
    if top1:
        pred_scores = logits.max(dim = 1)[0]
    elif gt_known:
        # GT-Known指标
        batch_size, _ = logits.shape
        _range = torch.arange(batch_size)
        pred_scores = logits[_range, labels]
    else: 
        print("Error in indicator designation!!!")
        exit()
    # pred_labels = logits.argmax(dim = 1)
    model.zero_grad()                          
    pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)

    # Calculate CAM
    activations = hookValues.activations    # ([1, 15, 124, 1])
    gradients = hookValues.gradients        # ([1, 15, 124, 1])
    signal_array = t2n(signal.permute(0, 2, 3, 1)) # bz, nc, h, w -> bz, h, w, nc
    
    camCalculator = eval(method)(signal_array, [name])
    scaledCAMs = camCalculator(t2n(activations), t2n(gradients))    # bz, h, w (1, 512, 1)
    camsOverlay = camCalculator._overlay_cam_on_image(layerName="model."+vis_layer)

    return camsOverlay, scaledCAMs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='基于HRRP数据的Pytorch模型CAM决策可视化'
    )
    parser.add_argument(
        '--project_path',
        default="../../../work_dirs/基于-14db仿真HRRP的DropBlock模型",
        type=str,
        help='工程文件路径, 包含数据集的工程文件夹路径'
    )
    parser.add_argument(
        '--model_name', 
        default="CNN_HRRP128.pth",
        type=str,
        help='工程路径下模型名指定'
    )
    parser.add_argument(
        '--dataset',
        default='test',
        type=str,
        help='数据集指定, 可选数据集: train, val, test'
    )
    parser.add_argument(
        '--mat_name',
        default="DT.mat",
        type=str,
        help='指定要可视化的.mat文件名'
    )
    parser.add_argument(
        '--mat_idx',
        default=[1, 10],
        type=list,
        help='指定.mat文件的索引,指定起始和终止位置,支持单个或多个索引'
    )
    parser.add_argument(
        '--batch_size',
        default=10,
        type=int,
        help='批大小'
    )
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='数据加载时的线程数'
    )
    parser.add_argument(
        '--cam_method',
        default="GradCAMpp",
        type=str,
        help='CAM决策可视化算法指定'
    )
    parser.add_argument(
        '--visualize_layer',
        default="Block_3[0]",
        type=str,
        help='可视化的隐层名'
    )
    args = parser.parse_args()

    # 读取数据
    ori_data = read_project(args.project_path, stages=[args.dataset], repeat=0)

    # 计算CAM
    camsOverlay, scaledCAMs = vis_cam(args, ori_data)

    # 保存图像
    saveImgPath = args.project_path + "/CAM_Output/" + args.dataset +"/"+ \
                args.mat_name
    if not os.path.exists(saveImgPath):
        os.makedirs(saveImgPath)
    for i, (camOverlay, scaledCAM) in enumerate(camsOverlay, scaledCAMs):
        # 保存CAM图像
        cv2.imwrite(saveImgPath +"/"+ str(args.mat_idx[0]+i) + \
                    +'_'+ args.cam_method + ".png", camOverlay)
        # 保存CAM矩阵数据
        scio.savemat(saveImgPath +"/"+ str(args.mat_idx[0]+i) + \
                    +'_'+ args.cam_method + ".mat", {'scaledCAM': scaledCAM})
        

    print("finished")


