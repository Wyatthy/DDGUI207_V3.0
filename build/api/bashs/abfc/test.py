import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']

import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data_dir', help='the directory of the project dir', default="db/datasets/local_dir/基于特征数据的ABFC网络")
args = parser.parse_args()

# HRRP数据归一化
def data_norm_hrrp(data):
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


# 特征数据归一化
def data_normalization(data):
    minmax = MinMaxScaler()
    X = minmax.fit_transform(data)
    return X


# 从工程文件路径中制作训练集、验证集、测试集
def read_project(read_path):
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
        if folder_name[i].casefold() == 'test':
            test_path = read_path + '/test/'
            test_data, test_label, test_classname = read_mat(test_path)
    
    return test_data, test_label, test_classname


# 从.mat文件读取数据并预处理
def read_mat(read_path):
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

    # 读取单个文件夹下的内容
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的所有.mat文件名称
        for j in range(0, len(class_mat_name)):
            one_mat_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[j]
            one_mat_data = sio.loadmat(one_mat_path)
            one_mat_data = one_mat_data[list(one_mat_data.keys())[-1]].T
            if j == 0:
                all_mat_data = one_mat_data
            else:
                all_mat_data = np.concatenate((all_mat_data, one_mat_data))

        # 设置标签
        label = np.zeros((len(all_mat_data), len(folder_name)))
        label[:, i] = 1

        if i == 0:
            all_class_data = all_mat_data
            all_label = label
        else:
            all_class_data = np.concatenate((all_class_data, all_mat_data))
            all_label = np.concatenate((all_label, label))
    return all_class_data, all_label, folder_name


# 绘制混淆矩阵
def show_confusion_matrix(classes, confusion_matrix, path_name):
    # confusion_matrix 为分类的特征矩阵
    plt.figure()
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12, rotation=20)
    plt.yticks(tick_marks, classes, fontsize=12)
    config = {"font.family": 'Times New Roman'}
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='red',
                     weight=5)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='red')
        else:
            plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(path_name, dpi=1000)


# 测试
def test_trained_model(x_test, y_test, folder_name, work_dir):
    attention_params_path = work_dir + '/attention_params.txt'
    attention_weights_rank_path = work_dir + '/attention_weight_rank.txt'
    attention_params = open(attention_params_path, 'r', encoding='utf-8').read().splitlines()
    attention_weights_rank = open(attention_weights_rank_path, 'r', encoding='utf-8').read().splitlines()
    feature_rank = []
    for i in range(0, len(attention_weights_rank)):
        feature_rank.append(int(attention_weights_rank[i]))
    if data_type == 'HRRP':
        x_test_norm = data_norm_hrrp(x_test)
    if data_type == 'feature':
        x_test_norm = data_normalization(x_test)
    x_test_rank = x_test_norm[:, feature_rank]
    model_num = int(attention_params[1]) + (int(attention_params[0])-1) * int(attention_params[2])
    x_test_select = x_test_rank[:, :model_num]
    x_test_select = np.expand_dims(x_test_select, axis=-1)
    save_model = tf.keras.models.load_model(work_dir + '/ABFC_feature_' + str(model_num) + '.hdf5')
    Y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(save_model.predict(x_test_select), axis=1)
    characteristic_matrix = confusion_matrix(Y_test, y_pred)
    show_confusion_matrix(folder_name, characteristic_matrix, work_dir+'/test_confusion_matrix.jpg')
    classification_report_txt = open(work_dir+'/test_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(Y_test, y_pred, digits=4))
    classification_report_txt.close()


if __name__ == '__main__':
    project_path = args.data_dir  # 工程目录
    data_type = 'HRRP'  # 输入数据的类型，'HRRP'或'feature'

    x_test, y_test, folder_name = read_project(project_path)
    test_trained_model(x_test, y_test, folder_name, project_path)
