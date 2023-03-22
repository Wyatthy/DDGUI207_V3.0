import os
import csv
import numpy as np
import scipy.io as sio
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler

import argparse


parser = argparse.ArgumentParser(description='UNKNOWN_TEST')
parser.add_argument('--data_dir', help='the directory of the project dir', default="db/datasets/local_dir/基于HRRP数据的Baseline_CNN网络")
args = parser.parse_args()

# 归一化
def trans_norm(data):
    data_trans = list(map(list, zip(*data)))
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(np.array(data_trans))
    trans_data = np.array(list(map(list, zip(*data_norm))))

    return trans_data


# 从工程文件路径中制作训练集、验证集、测试集
def read_project(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path + '/' + file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    for i in range(0, len(folder_name)):
        if folder_name[i].casefold() == 'train':  # 从训练集获取类别名称
            train_path = read_path + '/train/'
            train_classname = read_train_classname(train_path)
        if folder_name[i].casefold() == 'unknown_test':
            unknown_path = read_path + '/unknown_test/'
            unkown_data = read_unkown_test_mat(unknown_path)

    return unkown_data, train_classname


# 获取类别名称
def read_train_classname(read_path):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    # 将指定类别放到首位
    for i in range(0, len(folder_name)):
        if folder_name[i] == 'DT':
            folder_name.insert(0, folder_name.pop(i))
    return folder_name


# 读取未知类别
def read_unkown_test_mat(unkown_test_mat_path):
    folder_path = unkown_test_mat_path
    file_name = os.listdir(folder_path)
    unkown_data = {}
    for i in range(0, len(file_name)):
        one_unkown_mat_path = folder_path + '/' + file_name[i]
        one_unkown_mat_data = sio.loadmat(one_unkown_mat_path)
        one_unkown_mat_data = one_unkown_mat_data[list(one_unkown_mat_data.keys())[-1]].T

        one_unkown_mat_data_norm = trans_norm(one_unkown_mat_data)  # 归一化处理

        matrix_base = os.path.basename(one_unkown_mat_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        unkown_data[matrix_name] = one_unkown_mat_data_norm
    return unkown_data


# 测试
def unknown_trained_model(unkown_data, folder_name, model_name, model_naming, work_dir):
    save_model = keras.models.load_model(work_dir + '/' + model_naming + '.hdf5')
    for i in range(0, len(unkown_data.keys())):
        x_test = unkown_data[list(unkown_data.keys())[i]]
        if model_name == 'CNN':
            x_test = np.expand_dims(x_test, axis=-1)
        unkown_pred = save_model.predict(x_test)
        unknow_class = np.argmax(unkown_pred, axis=1)
        headname = ['序号', '类别', '预测概率']
        csvfile = open(project_path+'/'+list(unkown_data.keys())[i]+'.csv', mode='w', newline='')
        writecsv = csv.DictWriter(csvfile, fieldnames=headname)
        writecsv.writeheader()
        for j in range(0, len(unkown_pred)):
            writecsv.writerow({'序号': j+1, '类别': folder_name[unknow_class[j]], '预测概率': max(unkown_pred[j])})


if __name__ == '__main__':
    project_path = args.data_dir  # 工程目录
    model_naming = project_path.split('/')[-1]
    lowProjectPath = model_naming.lower()
    if 'dnn' in lowProjectPath:
        model_name = 'DNN'
    elif 'cnn' in lowProjectPath:
        model_name = 'CNN'
    unkown_data, folder_name = read_project(project_path)
    unknown_trained_model(unkown_data, folder_name, model_name, model_naming, project_path)
