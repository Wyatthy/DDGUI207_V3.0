import scipy.io as scio
import numpy as np
import os
# 读取每个文件夹中的所有.mat文件，将其分割成训练集和测试集


datasetFolder = 'H:/WIN_11_DESKTOP/onWorking/207GUI/GUI207_V2.0/api/HRRP_vis/data/HRRP_simulate_128xN_c6_-15dB'
# 遍历文件夹
for folder in os.listdir(datasetFolder):
    # 读取.mat文件
    matPath = datasetFolder + '/' + folder + '/' + folder + '.mat'
    matData = scio.loadmat(matPath)
    # 获取数据
    data = matData[folder]
    # 分割数据
    trainData = data[:, :int(data.shape[1]*0.5)]
    testData = data[:, int(data.shape[1]*0.5):]
    # 保存数据
    saveTrainFolder = datasetFolder + "_train/" + folder
    if not os.path.exists(saveTrainFolder):
        os.makedirs(saveTrainFolder)
    saveTestFolder = datasetFolder + "_test/" + folder
    if not os.path.exists(saveTestFolder):
        os.makedirs(saveTestFolder)
    scio.savemat(saveTrainFolder + '/' + folder + '.mat', {folder: trainData})
    scio.savemat(saveTestFolder  + '/' + folder + '.mat', {folder: testData})