# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        dataset/HRRP_mat.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2022/08/18
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> DD信号识别（可解释）系列代码 <--        
                    -- 数据集HRRP处理载入程序
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> PATH_ROOT/confgs.py
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> TODO
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> HRRPDataset(Dataset): 
                        -- 定义HRRPDataset类,继承Dataset方法,并重写
                        __getitem__()和__len__()方法
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/08/18 |   完成HRRP数据载入功能
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v1.1    | 2022/11/16 | 数据从.txt结构改为.mat结构
--------------------------------------------------------------------------
'''

import os
import numpy as np
import scipy.io as sio
# from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class HRRPDataset(Dataset):
    ''' 定义HRRPDataset类,继承Dataset方法,并重写__getitem__()和__len__()方法 '''
    def __init__(self, data_dict, transform=None):
        ''' 初始化函数,得到数据 '''
        self.signals = data_dict["data"]
        self.labels = data_dict["label"]
        self.label_names = data_dict["label_name"]
        self.transform = transform

    def __getitem__(self, index):
        ''' index是根据batchsize划分数据后得到的索引,最后将data和对应的labels进行一起返回 '''
        data = self.signals[index][None, :, None]
        label = self.labels[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        ''' 该函数返回数据大小长度,目的是DataLoader方便划分 '''
        return len(self.labels)


def read_mat(folder_path, repeat=64):
    ''' 从mat文件中读取数据 '''
    folder_names = [folder for folder in os.listdir(folder_path) \
                    if os.path.isdir(folder_path+'/'+folder)]
    folder_names.sort()                         # 按文件夹名进行排序

    for i in range(0, len(folder_names)):
        if folder_names[i].casefold() == 'dt':  # 将指定类别放到首位
            folder_names.insert(0, folder_names.pop(i))

    # 读取单个文件夹下的内容
    all_data = []
    all_label = []
    for class_name in folder_names:
        mat_files = [file for file in os.listdir(folder_path+'/'+class_name) \
                      if file.endswith(".mat")]
        concate_data = []
        for file in mat_files:
            ori_data = sio.loadmat(folder_path +'/'+ class_name +'/'+ file)
            concate_data.append(ori_data[list(ori_data.keys())[-1]].T)
        # 归一化处理
        concate_data = data_normalization(np.concatenate(concate_data, axis=0))
        # 重复堆叠数据
        if repeat > 1:
            concate_data = concate_data[:,:,None].repeat(repeat, 2)
        # 分配标签
        label = np.zeros((concate_data.shape[0], len(folder_names)))
        label[:, folder_names.index(class_name)] = 1

        all_data.append(concate_data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return {"data":all_data, "label":np.argmax(all_label, axis=1), \
            "label_name":folder_names}


def read_project(folder_path, stages=['train', 'val', 'test'], repeat=64):
    ''' 从工程文件路径中制作训练集、验证集、测试集 '''
    folder_names = [folder.casefold() for folder in os.listdir(folder_path) \
                    if os.path.isdir(folder_path+'/'+folder)]
    assert folder_names.sort() == stages.sort(), "工程目录下不包含指定文件夹"

    data = {stage: read_mat(folder_path +'/'+ stage +'/', repeat) for stage in stages}
    if 'train' in stages and 'val' in stages:
        assert data["train"]["label_name"] == data["val"]["label_name"], \
            "训练集和验证集的类别不一致"
    return data


def data_normalization(data):
    ''' 数据归一化处理 '''
    for i in range(0, len(data)):
        data[i] -= np.min(data[i])
        data[i] /= np.max(data[i])
    return data


if __name__ == "__main__":
    ''' 测试HRRP.py,测试dataLoader是否正常读取、处理数据 '''
    ori_data = read_project('./work_dirs/基于-14db仿真HRRP的DropBlock模型', stages=['train'], repeat=0)

    transform = transforms.Compose([ 
                                    # waiting add
                                    ])
    dataset = HRRPDataset(ori_data["train"], transform=transform)
    # 通过DataLoader读取数据
    HRRPLoader = DataLoader( 
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )
    for data, i in HRRPLoader:
        print("Size:", data.shape, i.shape)
    # for data, i in tqdm(HRRPLoader):
    #     print("Size:", data.shape, i.shape)