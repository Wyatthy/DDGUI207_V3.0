# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/train.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/06/14
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别(可解释)系列代码 <--        
                    -- 训练主程序,移植之前信号识别tensorflow代码至PyTorch,
                    并进行项目工程化处理
                    -- TODO train()部分代码需要模块化,特别是指标记录、数据集
                    方面
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/configs.py
                    <1> PATH_ROOT/dataset/RML2016.py
                    <2> PATH_ROOT/networks/MsmcNet.py
                    <3> PATH_ROOT/utils/strategy.py;plot.py
                    <4> PATH_ROOT/dataset/ACARS.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> train():
                        -- 训练主程序,包含了学习率调整、log记录、收敛曲线绘制
                        ,每训练n(1)轮验证一次,保留验证集上性能最好的模型
                    <1> eval():
                        -- 验证当前训练模型在测试集中的性能
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2020/06/14 | 使用PyTorch复现之前keras代码
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2020/07/09 |    新增ACARS训练程序选项
--------------------------------------------------------------------------
'''

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.HRRP_mat import HRRPDataset, read_project
from networks.hrrpCNN import CNN_HRRP512, CNN_HRRP128
from utils.strategy import step_lr, accuracy
from utils.plot import draw_curve

from utils.CAM import HookValues, GradCAM, GradCAMpp, XGradCAM, \
   EigenGradCAM, LayerCAM, t2n
from utils.CAM_opti import CAMbasedDropout

def train(args):
   ''' HRRP信号分类训练主程序 '''
   # Dataset
   ori_data = read_project(args.project_path, stages=['train', 'val'], repeat=0)
   transform = transforms.Compose([])
   # Train data
   train_dataset = HRRPDataset(ori_data['train'], transform=transform) 
   dataloader_train = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      shuffle=True,
      drop_last=False
   )
   # Valid data
   dataset_valid = HRRPDataset(ori_data['val'], transform=transform) 
   dataloader_valid = DataLoader(
      dataset_valid,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      shuffle=True,
      drop_last=False
   )
   # model
   model = eval(args.model)(num_classes=len(ori_data["train"]["label_name"]))
   print(model)
   model.cuda()
   if args.resume_path != "None":
      model = torch.load(args.resume_path)

   # log
   log = open(f'{args.project_path}/Train_log.txt', 'a+')
   log.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',\
                                  time.localtime(time.time()))+'-'*30+'\n')
   for name, value in args.__dict__.items():
      log.write(f"{name}: {value} \n") 
   log.write("\n\n")

   # loss
   criterion = nn.CrossEntropyLoss().cuda()  # 交叉熵损失

   ################## CAM CODE HERE ###################
   # 初始化CAMbasedDropout
   if args.camDropoutLayers is not None:
      layerNameDict = {
         'layer1': model.Block_1,
         'layer2': model.Block_2,
         'layer3': model.Block_3,
         'layer4': model.Block_4,
      }
      hookValuesMap = dict()
      for Hooklayer in args.camDropoutLayers:
         hookValuesMap[Hooklayer] = HookValues(layerNameDict[Hooklayer])
      camDropout = CAMbasedDropout(dropRate=args.drop_rate, maxFeatureStride=32)
   ################## CAM CODE END ###################
   # train 
   sum = 0
   train_loss_sum = 0
   train_top1_sum = 0
   max_val_acc = 0
   train_draw_acc = []
   val_draw_acc = []
   lr = args.lr
   for epoch in range(args.num_epochs):
      ep_start = time.time()

      # adjust lr 
      lr = step_lr(epoch, lr)

      # optimizer
      optimizer = torch.optim.Adam(
         filter(lambda p: p.requires_grad, model.parameters()), 
         lr=lr, betas=(0.9, 0.999), weight_decay=0.0002
      )

      model.train()
      top1_sum = 0
      for i, (signals, labels) in enumerate(dataloader_train):
            input = Variable(signals).cuda().float()
            target = Variable(labels).cuda()

            ################## CAM CODE HERE ###################
            if args.camDropoutLayers is not None:
               logits = model(input)            # inference
               logits = torch.sigmoid(logits)
               _range = torch.arange(signals.shape[0])
               pred_scores = logits[_range, labels]
               model.zero_grad()                          
               pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)

               dropedActivations = dict()
               for hookLayer in args.camDropoutLayers:
                  activations = hookValuesMap[hookLayer].activations
                  gradients = hookValuesMap[hookLayer].gradients
                  # 计算CAM
                  camCalculator = eval(args.cam_method)(
                     t2n(signals.permute(0, 2, 3, 1)), 
                     np.asarray(ori_data["train"]["label_name"])[labels]
                  )
                  camCalculator(t2n(activations), t2n(gradients))
                  CAMs = camCalculator.CAMs
                  # Vis Debug
                  # camsOverlay = camCalculator._overlay_cam_on_image()
                  # import cv2
                  # cv2.imwrite("2.png", camsOverlay[0])

                  # 计算Mask,并对特征进行掩码
                  stride = signals.shape[2]//activations.shape[2]
                  camDropout._reset_()
                  activations_droped = camDropout(activations, CAMs, stride)
                  dropedActivations[hookLayer] = activations_droped
                  # 重新前向传播，计算损失
               output = model(input, **dropedActivations)            
               loss = criterion(output, target) # 计算交叉熵损失
               optimizer.zero_grad()
               loss.backward()                  # 反传
            ################## CAM CODE END ###################
            else:
               output = model(input)            # inference
               loss = criterion(output, target) # 计算交叉熵损失
               optimizer.zero_grad()
               loss.backward()                  # 反传
            optimizer.step()
            top1 = accuracy(output.data, target.data, topk=(1,)) 
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += top1[0]
            sum += 1
            top1_sum += top1[0]

            if (i+1) % args.iter_smooth == 0:
               print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                     %(epoch+1, args.num_epochs, i+1, len(train_dataset)//args.batch_size, 
                     lr, train_loss_sum/sum, train_top1_sum/sum))
               log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f\n'
                           %(epoch+1, args.num_epochs, i+1, len(train_dataset)//args.batch_size, 
                           lr, train_loss_sum/sum, train_top1_sum/sum))
               sum = 0
               train_loss_sum = 0
               train_top1_sum = 0

      train_draw_acc.append(t2n(top1_sum/len(dataloader_train)))
      
      epoch_time = (time.time() - ep_start) / 60.
      if epoch % args.valid_freq == 0 and epoch < args.num_epochs:
            # eval
            val_time_start = time.time()
            val_loss, val_top1 = _eval(model, dataloader_valid, criterion)
            val_draw_acc.append(t2n(val_top1))
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f'
                  %(epoch+1, args.num_epochs, val_loss, val_top1, val_time*60, max_val_acc))
            print('epoch time: {}s'.format(epoch_time*60))
            if val_top1[0].data > max_val_acc:
               max_val_acc = val_top1[0].data
               print('Taking snapshot...')
               torch.save(model, f'{args.project_path}/{args.model}.pth')

            log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f\n'
                     %(epoch+1, args.num_epochs, val_loss, val_top1, val_time*60, max_val_acc))
   draw_curve(train_draw_acc, val_draw_acc, save_path=f'{args.project_path}/train_acc.png')
   log.write('-'*40+"End of Train"+'-'*40+'\n')
   log.close()


# validation
def _eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda().float()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1


if __name__ == '__main__':
   parser = argparse.ArgumentParser(
      description='基于HRRP数据的Pytorch模型的训练参数'
   )
   parser.add_argument(
      '--model', 
      default="CNN_HRRP128",
      type=str,
      help='模型指定, 可选模型: CNN_HRRP128, CNN_HRRP512'
   )
   parser.add_argument(
      '--resume_path',
      default="None",
      type=str,
      help='模型恢复, 恢复模型的路径, 默认为None表示重新训练'
   )
   parser.add_argument(
      '--project_path',
      default="../../../work_dirs/基于-14db仿真HRRP的DropBlock模型",
      type=str,
      help='数据集指定, 包含数据集的工程文件夹路径'
   )
   parser.add_argument(
      '--batch_size',
      default=10,
      type=int,
      help='训练时的批大小'
   )
   parser.add_argument(
      '--num_workers',
      default=4,
      type=int,
      help='数据加载时的线程数'
   )
   parser.add_argument(
      '--num_epochs',
      default=100,
      type=int,
      help='训练的轮数'
   )
   parser.add_argument(
      '--lr',
      default=0.01,
      type=float,
      help='初始学习率'
   )
   parser.add_argument(
      '--valid_freq',
      default=1,
      type=int,
      help='每隔多少个epoch进行一次验证'
   )
   parser.add_argument(
      '--iter_smooth',
      default=1,
      type=int,
      help='每隔多少个batch打印一次训练信息'
   )

   # CAMbasedDropout参数
   parser.add_argument(
      '--camDropoutLayers',
      default=None,
      # default=['layer3'],
      type=list,
      help="使用CAMbasedDropout的层, ['layer1','layer2','layer3','layer4']"
   )
   parser.add_argument(
      '--drop_rate',
      default=0.1,
      type=float,
      help='对特征图drop的概率'
   )
   parser.add_argument(
      '--cam_method',
      default='GradCAM',
      type=str,
      help='CAM的方法, 可选: GradCAM, GradCAMpp, XGradCAM, \
            EigenGradCAM, LayerCAM'
   )
   args = parser.parse_args()
   
   train(args)