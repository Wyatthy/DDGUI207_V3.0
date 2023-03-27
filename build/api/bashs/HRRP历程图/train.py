# encoding: utf-8
import os
import re
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import classification_report

import argparse

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
plt.rcParams['font.sans-serif'] = ['SimHei']


parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data',default="db/datasets/local_dir/基于历程图数据的Resnet50网络")
parser.add_argument('--batch_size', type=int, help='the number of batch size',default=32)
parser.add_argument('--max_epochs', type=int, help='the number of epochs',default=1)
parser.add_argument('--class_number', type=int, help="class_number", default=6)
parser.add_argument('--windows_length', type=int, help="windows_length", default=32)
parser.add_argument('--windows_step', type=int, help="windows_step", default=10)
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
def read_project(read_path, windows_length, windows_step):
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
            train_data, train_label, train_classname = read_mat(train_path, windows_length, windows_step)
        if folder_name[i].casefold() == 'val':
            val_path = read_path + '/val/'
            val_data, val_label, val_classname = read_mat(val_path, windows_length, windows_step)
    if len(train_classname) != len(val_classname):
        print('训练集类别数与验证集类别数不一致！！！')
    
    return train_data, train_label, val_data, val_label, train_classname


# 从.mat文件读取数据并预处理
def read_mat(read_path, windows_length, windows_step):
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
            one_mat_data_norm = data_normalization(one_mat_data)
            one_mat_data_win = HRRP_windows_cut(one_mat_data_norm, windows_length, windows_step)  # 滑窗截取HRRP数据，制作HRRP历程图
            if j == 0:
                all_mat_data_win = one_mat_data_win
            else:
                all_mat_data_win = np.concatenate((all_mat_data_win, one_mat_data_win))
        hrrp_pic = all_mat_data_win

        # 设置标签
        label = np.zeros((len(hrrp_pic), len(folder_name)))
        label[:, i] = 1

        if i == 0:
            all_class_data = hrrp_pic
            all_label = label
        else:
            all_class_data = np.concatenate((all_class_data, hrrp_pic))
            all_label = np.concatenate((all_label, label))
    return all_class_data, all_label, folder_name


# 滑窗截取HRRP数据，制作HRRP历程图
def HRRP_windows_cut(HRRP_data, windows_length, windows_step):
    data_num = len(HRRP_data)
    windows_num = int((data_num-windows_length)/windows_step) + 1
    HRRP_picture = []
    win_start = 0  # 滑窗截取开始位置
    for i in range(0, windows_num):
        hrrp_pic = HRRP_data[win_start:(windows_length+win_start)]
        win_start += windows_step
        HRRP_picture.append(hrrp_pic)
    return np.array(HRRP_picture)


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
    plt.savefig(work_dir+'/verification_confusion_matrix.jpg', dpi=1000)


# 训练过程中准确率曲线
def train_acc(epoch, acc, work_dir):
    x = np.arange(epoch+1)[1:]
    plt.figure()
    plt.plot(x, acc)
    plt.scatter(x, acc)
    plt.grid()
    plt.title('Training accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/training_accuracy.jpg', dpi=1000)


# 验证准确率曲线
def val_acc(v_acc, work_dir):
    x = np.arange(len(v_acc)+1)[1:]
    plt.figure()
    plt.plot(x, v_acc)
    plt.scatter(x, v_acc)
    plt.grid()
    plt.title('Verification accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/verification_accuracy.jpg', dpi=1000)


def run_main(x_train, y_train, x_val, y_val, class_num, folder_name, work_dir, model_name):
    len_train = len(x_train)
    len_val = len(x_val)
    train_shuffle = np.arange(len_train)
    val_shuffle = np.arange(len_val)
    np.random.shuffle(train_shuffle)
    np.random.shuffle(val_shuffle)

    x_train = x_train[train_shuffle, :]
    y_train = y_train[train_shuffle, :]

    x_val = x_val[val_shuffle, :]
    y_val = y_val[val_shuffle, :]

    model = tf.keras.models.Sequential()
    if (model_name == "DenseNet121"):
        model.add(tf.keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=(x_train.shape[1], x_train.shape[2], 1), pooling=None, classes=class_num))
    elif (model_name == "EfficientNetB0"):
        model.add(tf.keras.applications.efficientnet.EfficientNetB0(include_top=True, weights=None, input_tensor=None, input_shape=(x_train.shape[1], x_train.shape[2], 1), pooling=None, classes=class_num))
    elif (model_name == "ResNet50V2"):
        model.add(tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None, input_tensor=None, input_shape=(x_train.shape[1], x_train.shape[2], 1), pooling=None, classes=class_num))
    elif (model_name == "ResNet101"):
        model.add(tf.keras.applications.resnet.ResNet101(include_top=True, weights=None, input_tensor=None, input_shape=(x_train.shape[1], x_train.shape[2], 1), pooling=None, classes=class_num))
    elif (model_name == "MobileNet"):
        model.add(tf.keras.applications.mobilenet.MobileNet(include_top=True, weights=None, input_tensor=None, input_shape=(x_train.shape[1], x_train.shape[2], 1), pooling=None, classes=class_num))
    
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='lr', patience=10, verbose=1, factor=0.99, min_lr=0.00001)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(work_dir+'/'+model_naming+'.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learning_rate_reduction]
    h = model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, shuffle=True,
              validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=2, validation_freq=1)
    h_parameter = h.history
    train_acc(max_epochs, h_parameter['accuracy'], work_dir)
    val_acc(h_parameter['val_accuracy'], work_dir)
    save_model = tf.keras.models.load_model(work_dir+'/'+model_naming+'.hdf5')
    Y_val = np.argmax(y_val, axis=1)
    y_pred = np.argmax(save_model.predict(x_val), axis=1)

    args.valAcc=round(max(h_parameter['val_accuracy'])*100,2)

    labels = folder_name
    characteristic_matrix, accuracy_every_class = storage_characteristic_matrix(y_pred, Y_val, class_num)
    show_confusion_matrix(labels, characteristic_matrix, work_dir)
    classification_report_txt = open(work_dir+'/verification_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(Y_val, y_pred, digits=4))
    classification_report_txt.close()
    print(classification_report(Y_val, y_pred, digits=4))


def convert_h5to_pb(h5Path, pbPath):
    model = tf.keras.models.load_model(h5Path, compile=False)
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=pbPath[:pbPath.rfind(r"/")],
                      name=pbPath[pbPath.rfind(r"/")+1:],
                      as_text=False)
    ipsN, opsN = str(frozen_func.inputs[0]), str(frozen_func.outputs[0])
    inputNodeName = ipsN[ipsN.find("\"")+1:ipsN.find(":")]
    outputNodeName = opsN[opsN.find("\"")+1:opsN.find(":")]
    inputShapeK = ipsN[ipsN.find("=(")+2:ipsN.find("),")]
    inputShapeF = re.findall(r"\d+\.?\d*", inputShapeK)
    inputShape = reduce(lambda x, y: x + 'x' + y, inputShapeF)

    return inputNodeName, outputNodeName, inputShape


def convert_hdf5_to_trt(model_type, work_dir, model_naming, abfcmode_Idx, workspace='3072', optBatch='20', maxBatch='100'):
    if model_type == 'HRRP':
        hdfPath = work_dir+"/"+model_naming+".hdf5"
        trtPath = work_dir+"/"+model_naming+".trt"
    elif model_type == 'ABFC':
        hdfPath = work_dir+"/"+model_naming+"_feature_"+abfcmode_Idx+".hdf5"
        trtPath = work_dir+"/"+model_naming+"_feature_"+abfcmode_Idx+".trt"
    elif model_type == 'FewShot':
        hdfPath = work_dir+"/"+model_naming+".hdf5"
    elif model_type == 'ATEC':
        hdfPath = work_dir+"/fea_ada_trans.hdf5"
        trtPath = work_dir+"/"+model_naming+".trt"
    pbPath = work_dir+"/temp.pb"
    oxPath = work_dir+"/temp.onnx"

    try:
        inputNodeName, outputNodeName, inputShape = convert_h5to_pb(hdfPath, pbPath)
        # pb converto onnx
        '''python -m tf2onnx.convert --input temp.pb --inputs Input:0 --outputs Identity:0 --output temp.onnx --opset 11'''
        os.system("python -m tf2onnx.convert --input "+pbPath+" --inputs "+inputNodeName+":0 --outputs "+outputNodeName+":0 --output "+oxPath+" --opset 11")
        # onnx converto trt
        '''trtexec --explicitBatch --workspace=3072 --minShapes=Input:0:1x128x64x1 --optShapes=Input:0:20x128x64x1 --maxShapes=Input:0:100x128x64x1 --onnx=temp.onnx --saveEngine=temp.trt --fp16'''
        os.system("trtexec --onnx="+oxPath+" --saveEngine="+trtPath+" --workspace="+workspace+" --minShapes=Input:0:1x"+inputShape+\
        " --optShapes=Input:0:"+optBatch+"x"+inputShape+" --maxShapes=Input:0:"+maxBatch+"x"+str(inputShape)+" --fp16")
    except Exception as e:
        print(e)

def generator_model_documents(args):
    from xml.dom.minidom import Document
    doc = Document()  #创建DOM文档对象
    root = doc.createElement('ModelInfo') #创建根元素
    doc.appendChild(root)
    
    model_type = doc.createElement('IMAGE')
    #model_type.setAttribute('typeID','1')
    root.appendChild(model_type)

    model_item = doc.createElement(model_naming)
    #model_item.setAttribute('nameID','1')
    model_type.appendChild(model_item)

    model_infos = {
        'Model_DataType':"IMAGE",
        'Model_Name':model_naming,
        'Model_Algorithm':'TRAD_'+str(model_name),
        'Model_AlgorithmType':'传统深度学习模型',   
        'Model_AccuracyOnTrain':'-',
        'Model_AccuracyOnVal':str(args.valAcc),
        'Model_Framework':'Keras',
        'Model_TrainDataset':args.data_dir.split("/")[-1],
        'Model_TrainEpoch':str(args.max_epochs),
        'Model_TrainLR':'0.001',
        'Model_NumClassCategories':str(args.class_number), 
        'Model_Path':os.path.abspath(os.path.join(project_path,model_naming+'.trt')),
        'Model_TrainBatchSize':str(args.batch_size),
        'Model_WindowsLength':str(args.windows_length), 
        'Model_WindowsStep':str(args.windows_step), 
        'Model_Note':'-',
        'Model_Type':"TRAD"
    } 

    for key in model_infos.keys():
        info_item = doc.createElement(key)
        info_text = doc.createTextNode(model_infos[key]) #元素内容写入
        info_item.appendChild(info_text)
        model_item.appendChild(info_item)

    with open(os.path.join(project_path,model_naming+'.xml'),'w',encoding='utf-8') as f:
        doc.writexml(f,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')

# 保存参数
def save_params():
    params_txt = open(project_path+'/params_save.txt', 'w',encoding='utf-8')
    params_txt.write('project_path: ' + str(project_path) + '\n')
    params_txt.write('model_name: ' + str(model_name) + '\n')
    params_txt.write('model_type: ' + str(model_type) + '\n')
    params_txt.write('max_epochs: ' + str(max_epochs) + '\n')
    params_txt.write('batch_size: ' + str(batch_size) + '\n')
    params_txt.write('HRRP_picture_length: ' + str(picture_length) + '\n')
    params_txt.write('HRRP_picture_step: ' + str(picture_step) + '\n')
    params_txt.write('model_naming: ' + str(model_naming) + '\n')
    params_txt.close()


if __name__ == '__main__':
    # 超参数配置，包括训练集构成、测试集构成、模型文件、输出结果、网络类型等
    # 训练集数据
    project_path = args.data_dir  # 工程目录
    # 分割路径，获取文件名
    model_naming = project_path.split('/')[-1]
    # 网络名字
    lowProjectPath = model_naming.lower()
    if 'densenet' in lowProjectPath:
        model_name = 'DenseNet121'
    elif 'resnet50' in lowProjectPath:
        model_name = 'ResNet50V2'
    elif 'resnet101' in lowProjectPath:
        model_name = 'ResNet101'
    elif 'mobilenet' in lowProjectPath:
        model_name = 'MobileNet'
    elif 'efficientnet' in lowProjectPath:
        model_name = 'EfficientNetB0'
    else:
        assert False, '请在工程目录中包含Densenet、ResNet50、Resnet101、Mobilenet、Efficientnet中的一个'

    model_type = 'HRRP'  # 网络类型
    max_epochs = args.max_epochs  # 训练轮数
    batch_size = args.batch_size  # 批处理数量
    picture_length = args.windows_length  # RCS滑窗长度
    picture_step = args.windows_step  # RCS滑窗步长

    save_params()
    x_train, y_train, x_val, y_val, folder_name = read_project(project_path, picture_length, picture_step)
    print(x_train.shape)
    class_num = len(folder_name)

    run_main(x_train, y_train, x_val, y_val, class_num, folder_name, project_path, model_name)
    h5Path = project_path + '/' + model_naming + '.hdf5'
    pbPath = project_path + '/' + model_naming + '.pb'
    generator_model_documents(args)
    convert_hdf5_to_trt(model_type, project_path, model_naming, '1')
    print("Train Ended:")
