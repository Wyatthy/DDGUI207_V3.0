import os
import gc
import shutil
import argparse

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

import numpy as np
import scipy.io as scio
import tensorflow as tf
from tensorflow.keras.models import Model
from contextlib import redirect_stdout   
from tensorflow.keras.utils import plot_model  


def saveModelInfo(model, modelPath):
    rootPath = os.path.dirname(modelPath)
    modelName = os.path.basename(modelPath).split('.')[0]

    # 保存模型所有基本信息
    with open(rootPath + '/'+ modelName + "_modelInfo.txt", 'w') as f:
        with redirect_stdout(f):
            model.summary(line_length=200, positions=[0.30,0.60,0.7,1.0])
        
    # 保存模型所有层的名称至xml文件
    from xml.dom.minidom import Document
    xmlDoc = Document()
    child_1 = xmlDoc.createElement(modelName)
    xmlDoc.appendChild(child_1)
    child_2 = xmlDoc.createElement(modelName)
    child_1.appendChild(child_2)
    for layer in model.layers:  
        layer = layer.name.replace("/", "_")
        nodeList = layer.split("_")
        for i in range(len(nodeList)):
            modeName = nodeList[i].strip()
            if modeName.isdigit():
                modeName = "_" + modeName
            if i == 0:
                # 如果以modeName为名的节点已经存在，就不再创建，直接挂
                if len(child_2.getElementsByTagName(modeName)) == 0:
                    node1 = xmlDoc.createElement(modeName)
                    child_2.appendChild(node1)
                else:
                    node1 = child_2.getElementsByTagName(modeName)[0]
            elif i == 1:
                if len(node1.getElementsByTagName(modeName)) == 0:
                    node2 = xmlDoc.createElement(modeName)
                    node1.appendChild(node2)
                else:
                    node2 = node1.getElementsByTagName(modeName)[0]
            elif i == 2:
                if len(node2.getElementsByTagName(modeName)) == 0:
                    node3 = xmlDoc.createElement(modeName)
                    node2.appendChild(node3)
                else:
                    node3 = node2.getElementsByTagName(modeName)[0]
            elif i == 3:
                if len(node3.getElementsByTagName(modeName)) == 0:
                    node4 = xmlDoc.createElement(modeName)
                    node3.appendChild(node4)
                else:
                    node4 = node3.getElementsByTagName(modeName)[0]
    f = open(rootPath + '/'+ modelName + "_struct.xml", "w")
    xmlDoc.writexml(f, addindent='\t', newl='\n', encoding="utf-8")
    f.close()

    # 保存模型结构图
    if not os.path.exists(rootPath + '/'+ modelName + "_structImage"):
        os.makedirs(rootPath + '/'+ modelName + "_structImage")
    plot_model(model, to_file = rootPath + '/'+ modelName + "_structImage/framework.png", show_shapes=True, show_layer_names=True)


def data_normalization(data):
    """
        Func:
            数据归一化
        Args:
            data: 待归一化的数据
        Return:
            data: 归一化后的数据
    """
    for i in range(0, len(data)):
        data[i] -= np.min(data[i])
        data[i] /= np.max(data[i])
    return data


def capActAndGrad(signals, labels, checkpoint_path, \
                    targetLayerName='conv5_block16_2_conv', \
                    top1 = False, saveInfo = False):

    # Load model
    model = tf.keras.models.load_model(checkpoint_path)
    prediction = model.predict(signals)
    prediction_idx = np.argmax(prediction, axis=1)

    # Target hidden layer
    if ("CNN" in checkpoint_path) or ("DNN" in checkpoint_path) or ("ATEC" in checkpoint_path):   # baseline模型是在一级层下
        modelInner = model
    else:       # 其他模型是经过了sequential,在二级层下
        modelInner = model.get_layer(model.layers[0].name)
    if saveInfo:
        saveModelInfo(modelInner, checkpoint_path)
        exit()

    target_layer = modelInner.get_layer(targetLayerName)
    gradient_model = Model([modelInner.inputs], [target_layer.output, modelInner.output])

    # Compute Gradient of Top Predicted Class
    with tf.GradientTape() as tape:
        activations, prediction = gradient_model(signals)
        if top1:
            scores = tf.gather_nd(prediction, tf.stack([tf.range(prediction.shape[0]), prediction_idx], axis=1))
        else:
            scores = tf.gather_nd(prediction, tf.stack([tf.range(prediction.shape[0]), labels], axis=1))

        # Gradient() computes the gradient using operations recorded in context of this tape
        gradients = tape.gradient(scores, activations)

    # Change the position of channel axes, for visualization
    if activations.ndim == 2:
        activations = activations[:, :, None, None].numpy()
        gradients = gradients[:, :, None, None].numpy()
    elif activations.ndim == 3:
        activations = tf.transpose(activations, perm=[0, 2, 1])[:, :, :, None].numpy()
        gradients = tf.transpose(gradients, perm=[0, 2, 1])[:, :, :, None].numpy()
    elif activations.ndim == 4:
        activations = tf.transpose(activations, perm=[0, 3, 1, 2]).numpy()
        gradients = tf.transpose(gradients, perm=[0, 3, 1, 2]).numpy()

    for i in range(len(activations)):
        # 规整到一个合适的范围
        if np.max(np.abs(activations[i])) != 0:    
            activations[i] *= 1/np.max(np.abs(activations[i]))
        if np.max(np.abs(gradients[i])) != 0:
            gradients[i] *= 1/np.max(np.abs(gradients[i]))

    return activations, gradients, prediction_idx


def shuffle(data, label):
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DD数据的TensroFlow模型特征可视化'
    )
    parser.add_argument(
        '--project_path',
        default="../../../work_dirs/基于-14db仿真HRRP的DropBlock模型",
        type=str,
        help='工程文件路径, 包含数据集的工程文件夹路径'
    )
    parser.add_argument(
        '--model_name', 
        default="CNN_HRRP128.pth",
        type=str,
        help='工程路径下模型名指定'
    )
    parser.add_argument(
        '--mat_path',
        default="../../../work_dirs/基于-14db仿真HRRP的DropBlock模型/train/DT/DT.mat",
        type=str,
        help='指定要可视化的.mat文件名'
    )
    parser.add_argument(
        '--mat_idx',
        nargs='+',
        type=int,
        default=[5,6],
        help='指定.mat文件的索引,指定起始和终止位置,支持单个或多个索引'
    )
    parser.add_argument(
        '--visualize_layer',
        default="Block_1[1]",
        type=str,
        help='可视化的隐层名'
    )    
    parser.add_argument(
        '--feature_type',
        default="gradient",
        type=str,
        help='Visual feature or gradient'
    )
    parser.add_argument(
        '--save_model_info',
        default=False,
        type=str,
        help='是否保存模型信息至xml文件'
    )
    args = parser.parse_args()

    ############################# 读取数据 #############################
    # 获取dataset的类别名, 并进行排序，保证与模型对应
    dataset_path, class_name, mat_name = args.mat_path.rsplit('/', 2)
    folder_names = [folder for folder in os.listdir(dataset_path) \
                    if os.path.isdir(dataset_path+'/'+folder)]
    folder_names.sort()                         # 按文件夹名进行排序
    for i in range(0, len(folder_names)):
        if folder_names[i].casefold() == 'dt':  # 将指定类别放到首位
            folder_names.insert(0, folder_names.pop(i))
    # 读取数据
    ori_data = scio.loadmat(args.mat_path)
    signals = data_normalization(ori_data[list(ori_data.keys())[-1]].T)
    signals = signals[args.mat_idx[0]-1:args.mat_idx[1]]
    # 堆叠数据, 或根据输入调整信号维度
    if "CNN" in args.model_name or "DNN" in args.model_name or "ATEC" in args.model_name:
        repeatData = 0 
    else:
        repeatData = 64
    if repeatData > 1:
        signals = signals[:,:,None].repeat(repeatData, 2)
    signals = signals[..., None]    # 补上通道数
    # 分配标签
    labels = np.full((signals.shape[0],), folder_names.index(class_name))
    label_names = [class_name for i in range(signals.shape[0])]

    ############################# 可视化 #############################
    # 捕获激活和梯度
    activations, gradients, prediction_idx = capActAndGrad(     # bz, c, h, w 
        signals, labels, 
        checkpoint_path=f'{args.project_path}/{args.model_name}',
        targetLayerName=args.visualize_layer,
        top1=False,                 # True: top1, False: Ground Truth
        saveInfo=args.save_model_info
    )

    ############################# 保存 #############################
    # 保存激活图/梯度图
    if args.feature_type == "feature":
        visFeatures = np.mean(activations[0], axis=2)
    elif args.feature_type == "gradient":
        visFeatures = np.mean(gradients[0], axis=2)
    else:
        raise RuntimeError('args.feature_type must be "feature" or "gradient"')


    # 对batch中的每个样本进行保存
    for sampleIdx, visFeature in enumerate(visFeatures):
        # 检查保存路径
        saveImgPath = args.project_path + "/Features_Output/" + \
            dataset_path.rsplit('/', 1)[-1] +"/"+ class_name +"/"+ mat_name + \
            "/"+ str(sampleIdx+args.mat_idx[0]) + "/"+ args.feature_type + \
            "/"+ args.visualize_layer
        if os.path.exists(saveImgPath):
            shutil.rmtree(saveImgPath)
        os.makedirs(saveImgPath)

        if visFeature.ndim == 1:
            plt.figure(figsize=(18, 4), dpi=400)
            plt.grid(axis="y")
            plt.bar(range(len(visFeature)), visFeature)
            plt.title("Signal Type: "+label_names[sampleIdx]+"    Model Layer: "+"model."+args.visualize_layer)
            plt.xlabel("Number of units")
            plt.ylabel("Activation value")
            plt.savefig(saveImgPath + "/FC_vis.png")

            plt.clf()
            plt.close()
            gc.collect()
            # 保存特征矩阵数据
            scio.savemat(saveImgPath +"/FC_vis.mat", {'feature': visFeature})
        else:
            # 对每个通道进行保存
            for chIdx, featureMap in enumerate(visFeature):   # for every channels
                plt.figure(figsize=(18, 4), dpi=400)
                plt.title("Signal Type: "+label_names[sampleIdx]+"    Model Layer: "+"model."+args.visualize_layer)
                plt.xlabel('N')
                plt.ylabel("Value")
                plt.plot(featureMap[:, 0], linewidth=2, label = 'Hidden layer features')
                plt.legend(loc="upper right")
                plt.savefig(saveImgPath + "/" + str(chIdx+1) + ".png")

                plt.clf()
                plt.close()
                gc.collect()
                # 保存特征矩阵数据
                scio.savemat(saveImgPath +"/"+ str(chIdx+1) + ".mat", {'feature': featureMap[:, 0]})

    print("finished")
