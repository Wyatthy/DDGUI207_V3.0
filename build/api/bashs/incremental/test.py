import os
import torch
import numpy as np
import scipy.io as sio
from myNetwork import Network
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from model import AlexNet_128 as IncrementalModel_128

plt.rcParams['font.sans-serif'] = ['SimHei']


# 从工程文件路径中制作训练集、验证集
def read_project(read_path, oldclass_name):
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
        if folder_name[i].casefold() == 'test':
            test_path = read_path + '/test/'
            test_data, test_label, test_classname = read_mat(test_path, oldclass_name)

    return test_data, test_label, test_classname


# 从.mat文件读取数据并预处理
def read_mat(read_path, oldclass_name):
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

        class_data_norm = data_normalization(all_mat_data)  # 归一化处理

        # 设置标签
        label = np.zeros(len(class_data_norm))
        label[:] = i

        if i == 0:
            all_class_data = class_data_norm
            all_label = label
        else:
            all_class_data = np.concatenate((all_class_data, class_data_norm))
            all_label = np.concatenate((all_label, label))
    all_class_data = all_class_data[:, np.newaxis, :, np.newaxis]

    return all_class_data, all_label, folder_name


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
    # config = {"font.family": 'Times New Roman'}
    # rcParams.update(config)
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


def test_trained_model(project_path, oldclass_name_select):
    x_test, y_test, class_name = read_project(project_path, oldclass_name_select)
    model = Network(len(class_name), IncrementalModel_128())
    res = torch.load(project_path + "/incrementModel.pt")
    model.load_state_dict(res['model'])
    model.eval()
    x_test = np.float32(x_test)
    x_test = torch.tensor(x_test)
    y_pred = model(x_test)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    characteristic_matrix = confusion_matrix(y_test, y_pred)
    show_confusion_matrix(class_name, characteristic_matrix, project_path)
    classification_report_txt = open(project_path + '/test_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(y_test, y_pred, digits=4))
    classification_report_txt.close()


if __name__ == '__main__':
    project_path = 'D:/Engineering/HRRP_project/HRRP数据/增量测试/'
    # oldclass_name_select = ['类别2', '类别3', '类别4', '类别6']  # 旧类名称
    oldclass_name_select = ['Cone', 'Cone_cylinder', 'DT', 'Small_ball']  # 旧类名称

    test_trained_model(project_path, oldclass_name_select)
