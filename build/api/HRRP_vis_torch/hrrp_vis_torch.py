# -*- coding: utf-8 -*- #

import os
import scipy.io as scio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import gc
import shutil

import argparse

from dataset.HRRP_mat import data_normalization
from utils.CAM import t2n, HookValues
from utils.Guided_BP import *


def vis_fea(model, signals, labels, vis_layer):
    try:
        target_layer = eval(f'model.{vis_layer}')
    except Exception as e:
        print(model)
        raise RuntimeError('layer does not exist', e)
    hookValues = HookValues(target_layer)
    signals = torch.from_numpy(signals).cuda().float()

    # forward
    logits = model(signals)
    logits = torch.sigmoid(logits)

    # bakward
    batch_size, _ = logits.shape
    _range = torch.arange(batch_size)
    pred_scores = logits[_range, labels]
    pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)

    # Capture activations and gradients
    activations = t2n(hookValues.activations)       # ([1, 15, 124, 1])
    gradients = t2n(hookValues.gradients)           # ([1, 15, 124, 1])

    for i in range(len(activations)):
        # 规整到一个合适的范围
        if np.max(np.abs(activations[i])) != 0:    
            activations[i] *= 1/np.max(np.abs(activations[i]))
        if np.max(np.abs(gradients[i])) != 0:
            gradients[i] *= 1/np.max(np.abs(gradients[i]))

    return activations, gradients


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='基于HRRP数据的Pytorch模型特征可视化'
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
        '--feature_type',
        default="gradient",
        type=str,
        help='Visual feature or gradient'
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

    # 捕获特征
    activations, gradients = vis_fea(model, signals, labels, args.visualize_layer)    # bz, nc, sig_len, 1
    
    # 保存激活图/梯度图
    if args.feature_type == "feature":
        visFeatures = activations
    elif args.feature_type == "gradient":
        visFeatures = gradients
    else:
        raise RuntimeError('args.feature_type must be "feature" or "gradient"')

    # 对batch中的每个样本进行保存
    for sampleIdx, visFeature in enumerate(visFeatures):
        # 检查保存路径
        saveImgPath = args.project_path + "/Features_Output/" + \
            dataset_path.rsplit('/', 1)[-1] +"/"+ class_name +"/"+ mat_name + \
            "/"+ str(sampleIdx+args.mat_idx[0]) + "/"+ args.feature_type + \
            "/"+ args.visualize_layer
        if os.path.exists(saveImgPath):
            shutil.rmtree(saveImgPath)
        os.makedirs(saveImgPath)

        if visFeature.ndim == 1:
            plt.figure(figsize=(18, 4), dpi=400)
            plt.grid(axis="y")
            plt.bar(range(len(visFeature)), visFeature)
            plt.title("Signal Type: "+label_names[sampleIdx]+"    Model Layer: "+"model."+args.visualize_layer)
            plt.xlabel("Number of units")
            plt.ylabel("Activation value")
            plt.savefig(saveImgPath + "/FC_vis.png")

            plt.clf()
            plt.close()
            gc.collect()
            # 保存特征矩阵数据
            scio.savemat(saveImgPath +"/FC_vis.mat", {'feature': visFeature})
        else:
            # 对每个通道进行保存
            for chIdx, featureMap in enumerate(visFeature):   # for every channels
                plt.figure(figsize=(18, 4), dpi=400)
                plt.title("Signal Type: "+label_names[sampleIdx]+"    Model Layer: "+"model."+args.visualize_layer)
                plt.xlabel('N')
                plt.ylabel("Value")
                plt.plot(featureMap[:, 0], linewidth=2, label = 'Hidden layer features')
                plt.legend(loc="upper right")
                plt.savefig(saveImgPath + "/" + str(chIdx+1) + ".png")

                plt.clf()
                plt.close()
                gc.collect()
                # 保存特征矩阵数据
                scio.savemat(saveImgPath +"/"+ str(chIdx+1) + ".mat", {'feature': featureMap[:, 0]})

    print("finished")


