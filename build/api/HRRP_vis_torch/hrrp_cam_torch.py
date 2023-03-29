# -*- coding: utf-8 -*- #

import os
import cv2
import numpy as np
import torch
import scipy.io as scio
import argparse

from dataset.HRRP_mat import data_normalization
from utils.CAM import t2n, HookValues, GradCAM, GradCAMpp, XGradCAM, \
    EigenGradCAM, LayerCAM

def vis_cam(model, signals, labels, label_names, cam_method, vis_layer, \
             top1 = False, gt_known = True):
    try:
        target_layer = eval(f'model.{vis_layer}')
    except Exception as e:
        print(model)
        raise RuntimeError('layer does not exist', e)
    hookValues = HookValues(target_layer)
    signals = torch.from_numpy(signals).cuda().float()

    logits = model(signals)
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
    assert activations.ndim == 4 and gradients.ndim == 4, \
        "维度错误, 全连接层不能进行CAM可视化"
    signal_array = t2n(signals.permute(0, 2, 3, 1)) # bz, nc, h, w -> bz, h, w, nc
    
    camCalculator = eval(cam_method)(signal_array, label_names)
    scaledCAMs = camCalculator(t2n(activations), t2n(gradients))    # bz, h, w (1, 512, 1)
    camsOverlay = camCalculator._overlay_cam_on_image(layerName="model."+vis_layer)

    return camsOverlay, scaledCAMs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='基于HRRP数据的Pytorch模型CAM决策可视化'
    )
    parser.add_argument(
        '--project_path',
        default="../../../work_dirs/【优化】基于HRRP数据的DropBlock模型（-14dB）",
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
        '--mat_path',
        default="../../../work_dirs/【优化】基于HRRP数据的DropBlock模型（-14dB）/train/DT/DT.mat",
        type=str,
        help='指定要可视化的.mat文件名'
    )
    parser.add_argument(
        '--mat_idx',
        nargs='+',
        type=int,
        default=[5,6],
        help='指定.mat文件的索引,指定起始和终止位置,支持单个或多个索引'
    )
    parser.add_argument(
        '--visualize_layer',
        default="Block_1[1]",
        type=str,
        help='可视化的隐层名'
    )
    parser.add_argument(
        '--cam_method',
        default="GradCAM",
        type=str,
        help='CAM决策可视化算法指定, \
            可选: GradCAM, GradCAMpp, XGradCAM, EigenGradCAM, LayerCAM'
    )
    parser.add_argument(
        '--RCS',
        default=0,
        type=int,
        help='是否使用RCS数据集, 0表示不使用RCS数据集, 1表示使用RCS数据集'
    )
    parser.add_argument(
        '--IMAGE_WINDOWS_LENGTH',
        default=0,
        type=int,
        help='历程图数据集的窗口长度, 0表示不使用历程图数据集, 默认为32'
    )
    parser.add_argument(
        '--IMAGE_WINDOWS_STEP',
        default=0,
        type=int,
        help='历程图数据集的窗口步长, 0表示不使用历程图数据集, 默认为10'
    )
    args = parser.parse_args()

    # 获取dataset的类别名, 并进行排序，保证与模型对应
    dataset_path, class_name, mat_name = args.mat_path.rsplit('/', 2)
    folder_names = [folder for folder in os.listdir(dataset_path) \
                    if os.path.isdir(dataset_path+'/'+folder)]
    folder_names.sort()                         # 按文件夹名进行排序
    for i in range(0, len(folder_names)):
        if folder_names[i].casefold() == 'dt':  # 将指定类别放到首位
            folder_names.insert(0, folder_names.pop(i))
    # 读取数据
    ori_data = scio.loadmat(args.mat_path)
    signals = data_normalization(ori_data[list(ori_data.keys())[-1]].T)
    signals = signals[args.mat_idx[0]-1:args.mat_idx[1]]
    signals = signals[:, None, :, None]
    # 分配标签
    labels = np.full((signals.shape[0],), folder_names.index(class_name))
    label_names = [class_name for i in range(signals.shape[0])]

    # 载入模型
    model = torch.load(f'{args.project_path}/{args.model_name}')
    # print(model)
    model.cuda()

    # 计算CAM
    camsOverlay, scaledCAMs = vis_cam(model, signals, labels, label_names, \
                                      args.cam_method, args.visualize_layer)

    # 保存图像
    saveImgPath = args.project_path + "/CAM_Output/" + \
                dataset_path.rsplit('/', 1)[-1] +"/"+ class_name +"/"+ mat_name
    if not os.path.exists(saveImgPath):
        os.makedirs(saveImgPath)
    for i, (camOverlay, scaledCAM) in enumerate(zip(camsOverlay, scaledCAMs)):
        # 保存CAM图像
        cv2.imencode('.png', camOverlay)[1].tofile(saveImgPath +"/"+ \
                        str(args.mat_idx[0]+i) +'_'+ args.cam_method + ".png")
        # 保存CAM矩阵数据
        scio.savemat(saveImgPath +"/"+ str(args.mat_idx[0]+i) \
                    +'_'+ args.cam_method + ".mat", {'scaledCAM': scaledCAM})
        

    print("finished")


