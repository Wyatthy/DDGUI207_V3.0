import os
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tensorflow.keras as keras
from data_process import norm_one
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']

import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--choicedProjectPATH', help='the path of the project dir', default="db/datasets/local_dir/基于特征数据的ABFC网络")
parser.add_argument('--choicedMatPATH', help='the path of the mat dir', default="db/datasets/local_dir/基于特征数据的ABFC网络")
parser.add_argument('--inferMode', help='sample or dataset mode', default="dataset")
parser.add_argument('--choicedSampleIndex', type=int, help='the index of the sample', default=1)
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
            if j == 0 :
                all_mat_data = one_mat_data
            else:
                all_mat_data = np.concatenate((all_mat_data, one_mat_data))
        
        all_mat_data_norm = data_normalization(all_mat_data)  # 归一化处理

        # 设置标签
        label = np.zeros((len(all_mat_data_norm), len(folder_name)))
        label[:, i] = 1

        if i == 0:
            all_class_data = all_mat_data_norm
            all_label = label
        else:
            all_class_data = np.concatenate((all_class_data, all_mat_data_norm))
            all_label = np.concatenate((all_label, label))
    return all_class_data, all_label, folder_name

# 读取单样本数据
def read_one_mat(project_path, one_mat_path, one_test_num):
    # 读取路径下所有文件夹的名称并保存
    folder_path = project_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path + '/' + file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    for i in range(0, len(folder_name)):
        if folder_name[i].casefold() == 'train':
            train_path = project_path + '/train/'
            train_classname = read_train_classname(train_path)

    one_test_mat_data = sio.loadmat(one_mat_path)
    one_test_mat_data = one_test_mat_data[list(one_test_mat_data.keys())[-1]].T
    one_test_data_norm = data_normalization(one_test_mat_data)

    return one_test_data_norm[one_test_num], train_classname


# 从.mat文件读取数据并预处理
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

# 绘制混淆矩阵
def show_confusion_matrix(work_dir, classes, confusion_matrix):
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
    plt.savefig(work_dir+'./confusion_matrix_datasetinfer.jpg', dpi=1000)
    #plt.show()

# 数据集测试
def test_trained_model(x_test, y_test, folder_name, work_dir):
    x_test = np.expand_dims(x_test, axis=-1)
    net_fea_save_model = tf.keras.models.load_model(work_dir + '/net_fea.hdf5')
    fit_save_model = tf.keras.models.load_model(work_dir + '/fit_model.hdf5')
    classification_model = tf.keras.models.load_model(work_dir + '/fea_ada_trans.hdf5')

    functor = keras.models.Model(inputs=net_fea_save_model.input, outputs=net_fea_save_model.layers[-2].output)
    test_net_fea = functor(x_test)
    test_fit_fea = fit_save_model.predict(x_test)
    test_new = np.zeros((len(x_test), len(test_net_fea[0])+len(test_fit_fea[0])))
    test_new[:, :len(test_fit_fea[0])] = norm_one(test_net_fea, test_fit_fea)
    test_new[:, len(test_fit_fea[0]):len(test_net_fea[0])+len(test_fit_fea[0])] = test_net_fea
    
    new_test = np.expand_dims(test_new, axis=-1)
    Y_test = np.argmax(y_test, axis=1)
    Y_pred = np.argmax(classification_model.predict(new_test), axis=1)
    characteristic_matrix = confusion_matrix(Y_test, Y_pred)
    show_confusion_matrix(work_dir, folder_name, characteristic_matrix)
    classification_report_txt = open(work_dir+'/test_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(Y_test, Y_pred, digits=4))
    classification_report_txt.close()
    print('val_acc${:.3f}$'.format(accuracy_score(Y_test, Y_pred)*100))



# 单样本测试
def one_test_trained_model(one_test_data, folder_name, work_dir):
    one_test_data = one_test_data[np.newaxis, :, np.newaxis]
    net_fea_save_model = tf.keras.models.load_model(work_dir + '/net_fea.hdf5')
    fit_save_model = tf.keras.models.load_model(work_dir + '/fit_model.hdf5')
    classification_model = tf.keras.models.load_model(work_dir + '/fea_ada_trans.hdf5')

    functor = keras.models.Model(inputs=net_fea_save_model.input, outputs=net_fea_save_model.layers[-2].output)
    test_net_fea = functor(one_test_data)
    test_fit_fea = fit_save_model.predict(one_test_data)
    test_new = np.zeros((len(one_test_data), len(test_net_fea[0]) + len(test_fit_fea[0])))
    test_new[:, :len(test_fit_fea[0])] = norm_one(test_net_fea, test_fit_fea)
    test_new[:, len(test_fit_fea[0]):len(test_net_fea[0]) + len(test_fit_fea[0])] = test_net_fea
    new_test = np.expand_dims(test_new, axis=-1)
    T1 = time.perf_counter()
    y_pred = classification_model.predict(new_test)
    T2 = time.perf_counter()
    for i in y_pred[0]:
        print('$' + str(i), end='')
    print('$'+str(y_pred[0].argmax()), end='')
    print("$%.5f" % (T2 - T1), end="")
    print("$")

if __name__ == '__main__':
    project_path = args.choicedProjectPATH
    one_mat_path = args.choicedMatPATH
    test_type = args.inferMode
    one_test_num = args.choicedSampleIndex
    if test_type == 'dataset':
        x_test, y_test, folder_name = read_project(project_path)
        class_num = len(folder_name)
        test_trained_model(x_test, y_test, folder_name, project_path)
        print("finished")
    if test_type == 'sample':
        one_test_data, folder_name = read_one_mat(project_path, one_mat_path, one_test_num)
        one_test_trained_model(one_test_data, folder_name, project_path)
        print("finished")