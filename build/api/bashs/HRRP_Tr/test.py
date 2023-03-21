import os
import csv
import argparse
import matplotlib
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import shutil
# import keras as K
# from tensorflow import keras as K
import re
from functools import reduce
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from matplotlib import rcParams
# from datetime import datetime
from sklearn.metrics import classification_report
import sys
plt.rcParams['font.sans-serif'] = ['SimHei']

import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data_dir', help='the directory of the project dir', default="db/datasets/local_dir/基于HRRP数据的Resnet50网络")
args = parser.parse_args()

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


# 从工程文件路径中制作训练集、验证集、测试集
def read_project(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    for i in range(0, len(folder_name)):
        if folder_name[i].casefold() == 'test':
            test_path = read_path + '/test/'
            test_data, test_label, test_classname = read_mat(test_path)
    
    return test_data, test_label, test_classname


# 从.mat文件读取数据并预处理
def read_mat(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    classNum=len(folder_name)
    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的所有.mat文件名称
        for j in range(0, len(class_mat_name)):
            one_mat_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[j]
            one_mat_data = sio.loadmat(one_mat_path)
            one_mat_data = one_mat_data[list(one_mat_data.keys())[-1]].T
            if j == 0 :
                all_mat_data = one_mat_data
            else:
                all_mat_data = np.concatenate((all_mat_data, one_mat_data))
        
        class_data_picture = []
        class_data_normalization = data_normalization(all_mat_data)  # 归一化处理
        for j in range(0, len(class_data_normalization)):
            class_data_one = class_data_normalization[j]
            empty = np.zeros((len(class_data_one), 64))
            # empty = np.zeros((64, len(class_data_one)))
            for k in range(0, len(class_data_one)):
                empty[k, :] = class_data_one[k]
            class_data_picture.append(empty)
        class_data_picture = np.array(class_data_picture)   # 列表转换为数组

        # 设置标签
        label = np.zeros((len(class_data_normalization), len(folder_name)))
        label[:, i] = 1

        if i == 0:
            all_class_data =class_data_picture
            all_label = label
        else:
            all_class_data = np.concatenate((all_class_data, class_data_picture))
            all_label = np.concatenate((all_label, label))
    return all_class_data, all_label, folder_name


# 特征矩阵存储
def storage_characteristic_matrix(result, test_Y, output_size):
    characteristic_matrix = np.zeros((output_size, output_size))
    for i in range(0, len(result)):
        characteristic_matrix[test_Y[i], result[i]] += 1
    single_class_sum = np.zeros(output_size)
    pre_right_sum = np.zeros(output_size)
    for i in range(0, output_size):
        single_class_sum[i] = sum(characteristic_matrix[i])
        pre_right_sum[i] = characteristic_matrix[i, i]
    accuracy_every_class = pre_right_sum/single_class_sum
    return characteristic_matrix, accuracy_every_class


# 绘制混淆矩阵
def show_confusion_matrix(classes, confusion_matrix, work_dir):
    plt.figure()
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []  # 百分比(行遍历)
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)   # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12, rotation=20)
    plt.yticks(tick_marks, classes, fontsize=12)
    config = {"font.family": 'Times New Roman'} 
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues) 
    plt.colorbar()

    thresh = confusion_matrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='red',
                     weight=5)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='red')
        else:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/test_confusion_matrix.jpg', dpi=1000)
    # plt.show()


# 测试
def test_trained_model(x_test, y_test, folder_name, model_name, class_num, work_dir):
    save_model = tf.keras.models.load_model(work_dir + '/' + model_name + '.hdf5')
    Y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(save_model.predict(x_test), axis=1)
    labels = folder_name
    characteristic_matrix, accuracy_every_class = storage_characteristic_matrix(y_pred, Y_test, class_num)
    show_confusion_matrix(labels, characteristic_matrix, work_dir)
    classification_report_txt = open(work_dir+'/test_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(Y_test, y_pred, digits=4))
    classification_report_txt.close()
    print(classification_report(Y_test, y_pred, digits=4))


if __name__ == '__main__':
    project_path = args.data_dir  # 工程目录
    model_naming = project_path.split('/')[-1]

    x_test, y_test, folder_name = read_project(project_path)
    class_num = len(folder_name)
    test_trained_model(x_test, y_test, folder_name, model_naming, class_num, project_path)