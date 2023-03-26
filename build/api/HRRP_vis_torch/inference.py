# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/inference.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 测试模型分类准确率，并绘制混淆矩阵
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/configs.py
                    <1> PATH_ROOT/dataset/RML2016.py
                    <3> PATH_ROOT/utils/strategy.py;plot.py
                    <4> PATH_ROOT/dataset/ACARS.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> inference():
                        -- 使用训练好的模型对测试集进行推理，测试分类准确率，
                        绘制混淆矩阵
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
        |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <0> | JunJie Ren |   v1.0    | 2020/06/14 |           creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    <1> | JunJie Ren |   v1.1    | 2020/07/09 |    新增ACARS测试程序选项
--------------------------------------------------------------------------
'''

import time
import argparse

import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.HRRP_mat import HRRPDataset, read_project
from networks.hrrpCNN import CNN_HRRP512, CNN_HRRP128
from utils.strategy import accuracy
from utils.plot import plot_confusion_matrix, confusion_matrix


def inference():
    # Dataset
    ori_data = read_project(args.project_path, stages=[args.dataset], repeat=0)
    transform = transforms.Compose([])
    # Test data
    test_dataset = HRRPDataset(ori_data[args.dataset], transform=transform) 
    dataloader_train = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )
    # model
    # model = eval(args.model)(num_classes=len(ori_data[args.dataset]["label_name"]))
    model = torch.load(f'{args.project_path}/{args.model_name}')
    print(model)
    model.cuda()

    # log
    log = open(f'{args.project_path}/Test_log.txt', 'a+')
    log.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',\
                                    time.localtime(time.time()))+'-'*30+'\n')
    for name, value in args.__dict__.items():
        log.write(f"{name}: {value} \n") 
    log.write("\n")

    sum = 0
    val_top1_sum = 0
    labels = []
    preds = []
    model.eval()
    for ims, label in dataloader_train:
        labels += label.numpy().tolist()

        input = Variable(ims).cuda().float()
        target = Variable(label).cuda()
        output = model(input)

        _, pred = output.topk(1, 1, True, True)
        preds += pred.t().cpu().numpy().tolist()[0]

        top1_val = accuracy(output.data, target.data, topk=(1,))
        
        sum += 1
        val_top1_sum += top1_val[0]
    avg_top1 = val_top1_sum / sum

    log.write('acc: {}\n'.format(avg_top1.data))
    cm = confusion_matrix(labels, preds)
    log.write('confusion matrix:\n                    ')
    for pre_name in ori_data[args.dataset]["label_name"]:
        log.write("%15s"%pre_name)
    log.write('\n')
    for i in range(len(cm)):
        log.write("%20s" % ori_data[args.dataset]["label_name"][i])
        for j in range(len(cm[0])):
            log.write("%15s"%str(cm[i][j]))
        log.write('\n')
        
    log.write("\n\n")
    log.close()
              
    print('acc: {}'.format(avg_top1.data))
    plot_confusion_matrix(
        labels, preds, 
        ori_data[args.dataset]["label_name"],
        save_path=f'{args.project_path}/Test_ConfusionMatrix.png'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='基于HRRP数据的Pytorch模型的训练参数'
    )
    parser.add_argument(
        '--project_path',
        default="../../../work_dirs/基于-14db仿真HRRP的DropBlock模型",
        type=str,
        help='数据集指定, 包含数据集的工程文件夹路径'
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
    args = parser.parse_args()

    inference()
