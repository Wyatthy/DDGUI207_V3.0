'提取训练集中的特征'

import os,argparse
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks

# parser = argparse.ArgumentParser(description='Expert')
# parser.add_argument('--data_dir', help='the directory of the training data',default="N:/207/GUI207_V3.0/db/projects/基于HRRP数据的ATEC网络")
# args = parser.parse_args()

# 归一化
def max_min_norm(data):
    DATA = []
    for i in range(0, len(data)):
        data_norm = (data[i]-min(data))/(max(data)-min(data))
        DATA.append(data_norm)
    return np.array(DATA)


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


# 从工程文件路径中制作专家知识数据
def read_project_knowledge(read_path):
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
        if folder_name[i].casefold() == 'train':
            train_path = read_path + '/train/'
            read_mat_knowledge(train_path, read_path, 'train')
        if folder_name[i].casefold() == 'val':
            val_path = read_path + '/val/'
            read_mat_knowledge(val_path, read_path, 'val')


# 从.mat文件读取数据提取专家知识并保存
def read_mat_knowledge(read_path, project_path, dataset_name):
    # 读取路径下所有文件夹的名称并保存
    folder_path = read_path  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path + '/' + file_name[i]):
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
            one_mat_data_norm = data_normalization(one_mat_data)

            one_mat_knowledge = run_mechanism(one_mat_data_norm)
            save_path = project_path+'/' + str(dataset_name)+'_feature/'+folder_name[i]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            sio.savemat(save_path+'/'+class_mat_name[j], mdict={'data': np.float64(one_mat_knowledge.T)})


# 主瓣宽度法
def main_lobe(data):
    f = []  # 从大到小排序幅值
    f_index = []  # 幅值索引
    data_norm = []

    for i in range(0, len(data)):
        i_data = max_min_norm(data[i])  # 归一化处理
        data_norm.append(i_data)
        f.append(np.sort(i_data)[::-1])  # 由大到小排序
        f_index.append(np.argsort(i_data)[::-1])  # 排序索引

    f = np.array(f)
    f_index = np.array(f_index)

    # 最大幅值主瓣宽度附近所有点的幅值置0
    for i in range(0, len(f_index)):
        sc1_amplitude = f[i][0] * 0.5  # 最大幅值
        zb = []  # 主瓣宽度相同点
        l = 0.01
        while (True):
            for j in range(0, len(f[i])):
                if abs(f[i][j] - sc1_amplitude) <= l:
                    zb.append(f_index[i][j])
            if len(zb) != 0:
                break
            else:
                l += 0.01
        fw = []  # 求所有主瓣宽度相同点与散射中心的距离
        for j in range(0, len(zb)):
            dis = abs(zb[j] - f_index[i][0])
            fw.append(dis)
        kd = min(fw)
        for j in range(0, len(f_index[i])):
            if f_index[i][j] <= f_index[i][0] + kd and f_index[i][j] >= f_index[i][0] - kd:
                data_norm[i][f_index[i][j]] = 0

    # 寻找第2个散射中心
    f2 = []
    f2_index = []
    for i in range(0, len(data_norm)):
        i_f = data_norm[i]
        f2.append(np.sort(i_f)[::-1])
        f2_index.append(np.argsort(i_f)[::-1])  # 排序索引

    # 雷达视线上投影长度
    distance = []
    place = np.zeros((len(f_index), 2))
    for i in range(0, len(f_index)):
        distance.append(abs(f_index[i][0] - f2_index[i][0]))

    return distance


# 门限法
def threshold_value_method(data, threshold_value):
    data_norm = []
    for i in range(0, len(data)):
        i_data = max_min_norm(data[i])
        data_norm.append(i_data)
    data_norm = np.array(data_norm)

    distance = []
    for i in range(0, len(data_norm)):
        thre_line = threshold_value
        while(True):
            peaks, _ = find_peaks(data_norm[i], height=thre_line)
            if len(peaks) <= 1:
                thre_line -= 0.01
            else:
                break
        distance.append(abs(peaks[-1]-peaks[0]))

    return distance


# 均值 方差
def mean_var(data):
    mean = []
    variance = []

    for i in range(0, len(data)):
        i_mean = np.mean(data[i])
        i_variance = np.var(data[i])
        mean.append(i_mean)
        variance.append(i_variance)

    mean = np.array(mean)
    variance = np.array(variance)

    return mean, variance


def run_mechanism(data):
    mechanism_knowledge = np.zeros((len(data), 4))
    mechanism_knowledge[:, 0] = main_lobe(data)
    mechanism_knowledge[:, 1] = threshold_value_method(data, 0.6)
    mechanism_knowledge[:, 2], mechanism_knowledge[:, 3] = mean_var(data)

    return mechanism_knowledge


# if __name__ == '__main__':
#     project_path = r"N:\207\DDGUI207_V3.0\work_dirs\HRRP_ATEC"
#     # project_path = args.data_dir  # 工程路径
#     read_project_knowledge(project_path)
