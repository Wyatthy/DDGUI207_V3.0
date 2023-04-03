import os
import shutil
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.sans-serif'] = ['SimHei']


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
        if folder_name[i].casefold() == 'train':
            train_path = read_path + '/train/'
            train_data, train_label, train_classname = read_mat(train_path)
        if folder_name[i].casefold() == 'val':
            val_path = read_path + '/val/'
            val_data, val_label, val_classname = read_mat(val_path)
    if len(train_classname) != len(val_classname):
        print('训练集类别数与验证集类别数不一致！！！')
    
    return train_data, train_label, val_data, val_label, train_classname


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


# 读取未知类别
def read_unkown_test_mat(unkown_test_mat_path):
    folder_path = unkown_test_mat_path
    file_name = os.listdir(folder_path)
    unkown_data = {}
    for i in range(0, len(file_name)):
        one_unkown_mat_path = folder_path + '/' + file_name[i]
        one_unkown_mat_data = sio.loadmat(one_unkown_mat_path)
        one_unkown_mat_data = one_unkown_mat_data[list(one_unkown_mat_data.keys())[-1]].T

        matrix_base = os.path.basename(one_unkown_mat_path)
        matrix_name = os.path.splitext(matrix_base)[0]  # 获取去除扩展名的.mat文件名称
        unkown_data[matrix_name] = one_unkown_mat_data
    return unkown_data


# 特征数据归一化
def data_normalization(data):
    minmax = MinMaxScaler()
    X = minmax.fit_transform(data)
    return X


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
    accuracy = sum(pre_right_sum)/sum(single_class_sum)
    return characteristic_matrix, accuracy_every_class, accuracy


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


# 绘制features_Accuracy
def show_feature_selection(ac_score_list, feature_start, feature_end, feature_interval, path):
    plt.figure()
    x_interval = [i for i in range(feature_start, feature_end+1, feature_interval)]
    plt.plot(x_interval, ac_score_list)
    plt.scatter(x_interval, ac_score_list)
    plt.grid()
    plt.xticks(x_interval)
    plt.xlabel("Number of features used")
    plt.ylabel("Accuracy")
    plt.savefig(path + '/features_Accuracy.jpg', dpi=1000)


# 绘制特征权重展示图
def show_feature_weights(feature_weights, path, feature_select_num):
    plt.figure()
    pic_x = np.arange(len(feature_weights)) + 1
    for i in range(0, len(feature_weights)):
        if i in feature_select_num and i != max(feature_select_num):
            plt.bar(pic_x[i], feature_weights[i], color='r')
        elif i == max(feature_select_num):
            plt.bar(pic_x[i], feature_weights[i], color='r', label='选择特征')
        else:
            plt.bar(pic_x[i], feature_weights[i], color='b')
    plt.title("特征权重展示")
    plt.xlabel("特征")
    plt.ylabel("权重")
    plt.legend(loc='upper right', prop={'size': 6})
    plt.savefig(path + '/features_weights.jpg', dpi=1000)


# 训练过程中准确率曲线
def train_acc(epoch, acc, work_dir, fea_num):
    save_path = work_dir + '/train_acc_save/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    x = np.arange(epoch+1)[1:]
    plt.figure()
    plt.plot(x, acc)
    plt.scatter(x, acc)
    plt.grid()
    plt.title('Training accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path+'/training_accuracy_' + str(fea_num) + '.jpg', dpi=1000)


# 验证准确率曲线
def val_acc(v_acc, work_dir, fea_num):
    save_path = work_dir + '/val_acc_save/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    x = np.arange(len(v_acc)+1)[1:]
    plt.figure()
    plt.plot(x, v_acc)
    plt.scatter(x, v_acc)
    plt.grid()
    plt.title('Verification accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path+'/verification_accuracy_' + str(fea_num) + '.jpg', dpi=1000)


def mycopyfile(srcfile, dstpath, new_name):  # 复制函数
    fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)  # 创建路径
    shutil.copy(srcfile, dstpath + '/' + new_name)  # 复制文件
