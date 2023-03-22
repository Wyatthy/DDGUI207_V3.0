import os
import re
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from functools import reduce
from net_fea import net_fea_extract
from mapping_net import fea_mapping
from data_process import trans_norm
from expert_knowledge import run_mechanism
from sklearn.metrics import classification_report, confusion_matrix
from data_process import norm_one, show_confusion_matrix
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import shutil
import argparse

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data',default="db/datasets/local_dir/基于HRRP数据的ATEC网络")
parser.add_argument('--batch_size', type=int, help='the number of batch size',default=32)
parser.add_argument('--max_epochs', type=int, help='the number of epochs',default=1)
parser.add_argument('--class_number', type=int, help="class_number", default="6")
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
        if folder_name[i].casefold() == 'train':
            train_path = read_path + '/train/'
            train_data, train_label, train_class_data_num, train_classname, train_folder_file_name, train_file_class_num = read_mat(train_path)
        if folder_name[i].casefold() == 'val':
            val_path = read_path + '/val/'
            val_data, val_label, val_class_data_num, val_classname, val_folder_file_name, val_file_class_num = read_mat(val_path)
    if len(train_classname) != len(val_classname):
        print('训练集类别数与验证集类别数不一致！！！')
    
    return train_data, train_label, val_data, val_label, train_classname, train_folder_file_name, train_file_class_num


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

    folder_file_name = {}
    file_class_num = {}

    # 读取单个文件夹下的内容
    class_data_num = []
    for i in range(0, len(folder_name)):
        class_mat_name = os.listdir(folder_path + '/' + folder_name[i])  # 获取类别文件夹下的所有.mat文件名称
        folder_file_name[folder_name[i]] = class_mat_name  # 存储文件夹名称及其包含的.mat文件名称
        folder_num = []  # 存储.mat文件包含样本的数目
        for j in range(0, len(class_mat_name)):
            one_mat_path = folder_path + '/' + folder_name[i] + '/' + class_mat_name[j]
            one_mat_data = sio.loadmat(one_mat_path)
            one_mat_data = one_mat_data[list(one_mat_data.keys())[-1]].T
            folder_num.append(len(one_mat_data))
            if j == 0:
                all_mat_data = one_mat_data
            else:
                all_mat_data = np.concatenate((all_mat_data, one_mat_data))

        file_class_num[folder_name[i]] = folder_num
        all_mat_data_norm = data_normalization(all_mat_data)  # 归一化处理

        # 设置标签
        label = np.zeros((len(all_mat_data_norm), len(folder_name)))
        label[:, i] = 1

        class_data_num.append(len(all_mat_data_norm))

        if i == 0:
            all_class_data = all_mat_data_norm
            all_label = label
        else:
            all_class_data = np.concatenate((all_class_data, all_mat_data_norm))
            all_label = np.concatenate((all_label, label))

    return all_class_data, all_label, class_data_num, folder_name, folder_file_name, file_class_num


# 存储映射数据
def data_save(data, folder_file_name, file_class_num, work_dir, feature_folder_name):
    mat_folder_name = list(folder_file_name.keys())
    data_start = 0  # 记录数据位置
    for i in range(0, len(mat_folder_name)):
        for j in range(0, len(folder_file_name[mat_folder_name[i]])):
            one_data = data[data_start:data_start+file_class_num[mat_folder_name[i]][j]]
            data_start += file_class_num[mat_folder_name[i]][j]
            save_mat_path = work_dir + '/feature_save/' + feature_folder_name + '/' + mat_folder_name[i] + '/'
            if not os.path.exists(save_mat_path):
                os.makedirs(save_mat_path)
            sio.savemat(save_mat_path+folder_file_name[mat_folder_name[i]][j], mdict={'data': one_data.T})


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
    plt.savefig(work_dir+'./training_accuracy.jpg', dpi=1000)


# 验证准确率曲线
def val_acc(v_acc, work_dir):
    x = np.arange(len(v_acc)+1)[1:]
    plt.figure()
    plt.plot(x, v_acc)
    plt.scatter(x, v_acc)
    plt.grid()
    plt.title('Verification accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Times', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'./verification_accuracy.jpg', dpi=1000)


# 学习模块
def rcn_model(train_x, train_y, val_x, val_y, epoch, batch_size, work_dir):
    rcn_model = keras.Sequential([
        keras.layers.Conv1D(16, kernel_size=1, padding='valid', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])
    rcn_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    save_model_path = work_dir +'/fea_ada_trans.hdf5'
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='lr', factor=0.99, patience=3,
                                                             verbose=0, min_lr=0.0001)
    checkpoint = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learn_rate_reduction]
    h = rcn_model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, shuffle=True, validation_data=(val_x, val_y),
                  callbacks=callbacks_list, verbose=1, validation_freq=1)
    h_parameter = h.history
    train_acc(epoch, h_parameter['accuracy'], work_dir)
    val_acc(h_parameter['val_accuracy'], work_dir)
    val_model = keras.models.load_model(save_model_path)
    Y_val = np.argmax(val_y, axis=1)
    Y_pred = np.argmax(val_model.predict(val_x), axis=1)
    args.valAcc=round(max(h_parameter['val_accuracy'])*100,2)
    return Y_val, Y_pred


# 特征交融
def run_mapping(train_x, train_y, val_x, val_y, epoch, batch_size, work_dir):
    train_Y, val_Y = run_mechanism(train_x), run_mechanism(val_x)
    data_save(train_Y, folder_file_name, file_class_num, work_dir, 'traditional_feature')
    train_x, val_x = trans_norm(train_x), trans_norm(val_x)
    train_x = np.expand_dims(train_x, axis=-1)
    val_x = np.expand_dims(val_x, axis=-1)

    # 神经网络特征提取
    path_one = work_dir +'/net_fea.hdf5'
    train_fea, val_fea = net_fea_extract(path_one, train_x, train_y, val_x, val_y, epoch, batch_size)

    # 特征适应变换
    path_two = work_dir +'/fit_model.hdf5'
    train_mapping, val_mapping = fea_mapping(path_two, train_x, train_Y, val_x, val_Y, epoch, batch_size)
    data_save(train_mapping, folder_file_name, file_class_num, work_dir, 'mapping_feature')

    train_new = np.zeros((len(train_x), len(train_fea[0])+len(train_mapping[0])))
    val_new = np.zeros((len(val_x), len(val_fea[0])+len(val_mapping[0])))

    train_new[:, :len(train_mapping[0])] = norm_one(train_fea, train_mapping)
    val_new[:, :len(val_mapping[0])] = norm_one(val_fea, val_mapping)
    train_new[:, len(train_mapping[0]):len(train_fea[0])+len(train_mapping[0])] = train_fea
    val_new[:, len(val_mapping[0]):len(val_fea[0])+len(val_mapping[0])] = val_fea

    return train_new, val_new


def inference(train_x, train_y, val_x, val_y, batch_size, max_epochs, folder_name, work_dir):
    train_new, val_new = run_mapping(train_x, train_y, val_x, val_y, max_epochs, batch_size, work_dir)
    len_train = len(train_new)
    len_val = len(val_new)
    train_shuffle = np.arange(len_train)
    val_shuffle = np.arange(len_val)
    np.random.shuffle(train_shuffle)
    np.random.shuffle(val_shuffle)
    new_train = train_new[train_shuffle, :]
    train_y = train_y[train_shuffle, :]
    new_val = val_new[val_shuffle, :]
    val_y = val_y[val_shuffle, :]

    new_train = np.expand_dims(new_train, axis=-1)
    new_val = np.expand_dims(new_val, axis=-1)
    Y_test, Y_pred = rcn_model(new_train, train_y, new_val, val_y, max_epochs, batch_size, work_dir)
    characteristic_matrix = confusion_matrix(Y_test, Y_pred)
    class_label = folder_name  # 标签显示
    show_confusion_matrix(work_dir, class_label, characteristic_matrix)
    classification_report_txt = open(work_dir+'/verification_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(Y_test, Y_pred, digits=4))
    classification_report_txt.close()
    print(classification_report(Y_test, Y_pred, digits=4))

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
    
    model_type = doc.createElement('FEA_RELE')
    #model_type.setAttribute('typeID','1')
    root.appendChild(model_type)

    model_item = doc.createElement(project_path+'.trt')
    #model_item.setAttribute('nameID','1')
    model_type.appendChild(model_item)

    model_infos = {
        'Model_Name':str(model_naming),
        'Model_Algorithm':'ATEC',
        'Model_AccuracyOnTrain':'-',
        'Model_AccuracyOnVal':str(args.valAcc),
        'Model_Framework':'Keras',
        'Model_TrainDataset':args.data_dir.split("/")[-1],
        'Model_TrainEpoch':str(args.max_epochs),
        'Model_TrainLR':'0.001',
        'Model_NumClassCategories':str(args.class_number), 
        'Model_Path':os.path.abspath(os.path.join(project_path,model_naming+'.trt')),
        'Model_TrainBatchSize':str(args.batch_size),
        'Model_Note':'-'
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
    params_txt = open(project_path+'/params_save.txt', 'w')
    params_txt.write('project_path: ' + str(project_path) + '\n')
    params_txt.write('model_name: ' + str(model_name) + '\n')
    params_txt.write('max_epochs: ' + str(max_epochs) + '\n')
    params_txt.write('batch_size: ' + str(batch_size) + '\n')
    params_txt.close()


if __name__ == '__main__':

    project_path = args.data_dir  # 工程目录
    # 分割路径，获取文件名
    model_naming = project_path.split('/')[-1]
    model_name = 'ATEC'
    max_epochs = args.max_epochs  # 训练轮数
    batch_size = args.batch_size  # 批处理数量

    file_name = os.listdir(project_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(project_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    args.class_number=len(folder_name)

    save_params()
    train_x, train_y, val_x, val_y, folder_name, folder_file_name, file_class_num = read_project(project_path)
    inference(train_x, train_y, val_x, val_y, batch_size, max_epochs, folder_name, project_path)

    convert_hdf5_to_trt(model_name, project_path, model_naming, '1')
    generator_model_documents(args)
    print("Train Ended:")
