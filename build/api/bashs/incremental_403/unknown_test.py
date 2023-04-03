import os
import csv
import torch
import numpy as np
import scipy.io as sio
from myNetwork import Network
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

from model import AlexNet_128 as IncrementalModel_128

plt.rcParams['font.sans-serif'] = ['SimHei']


# 从工程文件路径中制作训练集、验证集、测试集
def read_project(read_path, oldclass_name):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    for i in range(0, len(folder_name)):
        if folder_name[i].casefold() == 'train':
            train_path = read_path + '/train/'
            train_classname = read_train_classname(train_path, oldclass_name)
        if folder_name[i].casefold() == 'unknown_test':
            unknown_path = read_path + '/unknown_test/'
            unknown_data = read_unknown_test_mat(unknown_path)

    return unknown_data, train_classname


# 读取未知类别
def read_unknown_test_mat(unkown_test_mat_path):
    folder_path = unkown_test_mat_path
    file_name = os.listdir(folder_path)
    unkown_data = {}
    for i in range(0, len(file_name)):
        one_unkown_mat_path = folder_path + '/' + file_name[i]
        one_unkown_mat_data = sio.loadmat(one_unkown_mat_path)
        one_unkown_mat_data = one_unkown_mat_data[list(one_unkown_mat_data.keys())[-1]].T

        one_unkown_mat_data_norm = data_normalization(one_unkown_mat_data)  # 归一化处理

        matrix_base = os.path.basename(one_unkown_mat_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        unkown_data[matrix_name] = one_unkown_mat_data_norm[:, np.newaxis, :, np.newaxis]
    return unkown_data


# 从.mat文件读取数据并预处理
def read_train_classname(read_path, oldclass_name):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path + '/' + file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序

    # 将旧类放在前面
    newclass_name = []
    for i in range(0, len(folder_name)):
        if folder_name[i] not in oldclass_name:
            newclass_name.append(folder_name[i])
    old_new_name = oldclass_name
    old_new_name.extend(newclass_name)
    folder_name = old_new_name

    return folder_name


# 数据归一化
def data_normalization(data):
    DATA = []
    for i in range(0, len(data)):
        data_max = max(data[i])
        data_min = min(data[i])
        data_norm = []
        for j in range(0, len(data[i])):
            data_one = (data[i][j] - data_min) / (data_max - data_min)
            data_norm.append(data_one)
        DATA.append(data_norm)
    DATA = np.array(DATA)
    return DATA


def unknown_test_trained_model(project_path, oldclass_name_select):
    unknown_data, class_name = read_project(project_path, oldclass_name_select)
    model = Network(len(class_name), IncrementalModel_128())
    res = torch.load(project_path + "/incrementModel.pt")
    model.load_state_dict(res['model'])
    model.eval()

    for i in range(0, len(unknown_data.keys())):
        unknown_one_data = unknown_data[list(unknown_data.keys())[i]]
        unknown_one_data = np.float32(unknown_one_data)
        unknown_one_data = torch.tensor(unknown_one_data)
        with torch.no_grad():
            logits = model(unknown_one_data)
            probabilities = F.softmax(logits, dim=1)
        unknown_pred = probabilities
        unknown_pred = unknown_pred.detach().numpy()
        unknown_class = np.argmax(unknown_pred, axis=1)
        headname = ['序号', '类别', '预测概率']
        csvfile = open(project_path + '/' + list(unknown_data.keys())[i] + '.csv', mode='w', newline='')
        writecsv = csv.DictWriter(csvfile, fieldnames=headname)
        writecsv.writeheader()
        for j in range(0, len(unknown_pred)):
            writecsv.writerow({'序号': j + 1, '类别': class_name[unknown_class[j]], '预测概率': max(unknown_pred[j])})


if __name__ == '__main__':
    project_path = 'D:/Engineering/HRRP_project/HRRP数据/增量测试/'
    # oldclass_name_select = ['类别2', '类别3', '类别4', '类别6']  # 旧类名称
    oldclass_name_select = ['Cone', 'Cone_cylinder', 'DT', 'Small_ball']  # 旧类名称

    unknown_test_trained_model(project_path, oldclass_name_select)
