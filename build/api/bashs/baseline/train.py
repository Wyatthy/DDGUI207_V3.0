import os
import re,shutil
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data',default="./db/datasets/local_dir/基于HRRP数据的Baseline_CNN网络")
parser.add_argument('--batch_size', help='the number of batch size',default=32)
parser.add_argument('--max_epochs', help='the number of epochs',default=10)
parser.add_argument('--net', help="network frame", default="DNN")
parser.add_argument('--modeldir', help="model saved path", default="./db/models")
parser.add_argument('--class_number', help="class_number", default="6")
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
        if os.path.isdir(folder_path + '/' + file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序

    # 将指定类别放到首位
    for i in range(0, len(folder_name)):
        if folder_name[i] == 'DT':
            folder_name.insert(0, folder_name.pop(i))
    args.class_number=len(folder_name)
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

        class_data_norm = trans_norm(all_mat_data)  # 归一化处理

        # 设置标签
        label = np.zeros((len(class_data_norm), len(folder_name)))
        label[:, i] = 1

        if i == 0:
            all_class_data = class_data_norm
            all_label = label
        else:
            all_class_data = np.concatenate((all_class_data, class_data_norm))
            all_label = np.concatenate((all_label, label))
    return all_class_data, all_label, folder_name


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


def DNN(path, train_x, train_y, val_x, val_y, epoch, batch_size, work_dir):
    model = keras.models.Sequential([
        keras.layers.Dense(150, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(150, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(150, activation='relu'),
        keras.layers.Dropout(0.2),
        # keras.layers.Dense(50, activation='relu'),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='val_accuracy',
                                                    verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    h = model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, validation_data=(val_x, val_y),
              callbacks=callbacks_list, verbose=2, validation_freq=1)
    h_parameter = h.history
    train_acc(epoch, h_parameter['accuracy'], work_dir)
    val_acc(h_parameter['val_accuracy'], work_dir)
    save_model = keras.models.load_model(path)
    y_val = np.argmax(val_y, axis=1)
    y_pred = np.argmax(save_model.predict(val_x), axis=1)
    args.valAcc=round(max(h_parameter['val_accuracy'])*100,2)
    return y_val, y_pred


def CNN(path, train_x, train_y, val_x, val_y, epoch, batch_size, work_dir):
    model = keras.models.Sequential([
        keras.layers.Conv1D(32, kernel_size=10, strides=1, padding='valid', activation='relu'),
        keras.layers.MaxPool1D(pool_size=2),
        keras.layers.Conv1D(64, kernel_size=10, strides=1, padding='valid', activation='relu'),
        keras.layers.MaxPool1D(pool_size=2),
        keras.layers.Conv1D(128, kernel_size=10, strides=1, padding='valid', activation='relu'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.2),
        # keras.layers.Dense(1024, activation='relu'),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='val_accuracy',
                                                 verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    h = model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, validation_data=(val_x, val_y),
              callbacks=callbacks_list, verbose=2, validation_freq=1)
    h_parameter = h.history
    train_acc(epoch, h_parameter['accuracy'], work_dir)
    val_acc(h_parameter['val_accuracy'], work_dir)
    save_model = keras.models.load_model(path)
    y_val = np.argmax(val_y, axis=1)
    y_pred = np.argmax(save_model.predict(val_x), axis=1)
    args.valAcc=round(max(h_parameter['val_accuracy'])*100,2)
    return y_val, y_pred


def inference_DNN(train_x, train_y, val_x, val_y, folder_name, work_dir, model_naming):
    len_train = len(train_x)
    len_val = len(val_x)
    train_shuffle = np.arange(len_train)
    val_shuffle = np.arange(len_val)
    np.random.shuffle(train_shuffle)
    np.random.shuffle(val_shuffle)
    x_train = train_x[train_shuffle, :]
    y_train = train_y[train_shuffle, :]
    x_val = val_x[val_shuffle, :]
    y_val = val_y[val_shuffle, :]
    save_hdf5_path = work_dir+'/'+model_naming+'.hdf5'
    y_val, y_pred = DNN(save_hdf5_path, x_train, y_train, x_val, y_val, max_epochs, batch_size, work_dir)
    characteristic_matrix = confusion_matrix(y_val, y_pred)
    classification_report_txt = open(work_dir + '/verification_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(y_val, y_pred, digits=4))
    classification_report_txt.close()
    print(classification_report(y_val, y_pred, digits=4))
    labels = folder_name  # 标签显示
    show_confusion_matrix(labels, characteristic_matrix, work_dir)


def inference_CNN(train_x, train_y, val_x, val_y, folder_name, work_dir, model_naming):
    len_train = len(train_x)
    len_val = len(val_x)
    train_shuffle = np.arange(len_train)
    val_shuffle = np.arange(len_val)
    np.random.shuffle(train_shuffle)
    np.random.shuffle(val_shuffle)
    x_train = train_x[train_shuffle, :]
    y_train = train_y[train_shuffle, :]
    x_val = val_x[val_shuffle, :]
    y_val = val_y[val_shuffle, :]
    train_x, val_x = np.expand_dims(x_train, axis=-1), np.expand_dims(x_val, axis=-1)
    save_hdf5_path = work_dir+'/'+model_naming+'.hdf5'
    y_val, y_pred = CNN(save_hdf5_path, train_x, y_train, val_x, y_val, max_epochs, batch_size, work_dir)
    classification_report_txt = open(work_dir + '/verification_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(y_val, y_pred, digits=4))
    classification_report_txt.close()
    print(classification_report(y_val, y_pred, digits=4))
    labels = folder_name  # 标签显示
    characteristic_matrix = confusion_matrix(y_val, y_pred)
    show_confusion_matrix(labels, characteristic_matrix, work_dir)

def generator_model_documents(args):
    from xml.dom.minidom import Document
    doc = Document()  #创建DOM文档对象
    root = doc.createElement('ModelInfo') #创建根元素
    doc.appendChild(root)
    
    model_type = doc.createElement('TRA_DL')
    #model_type.setAttribute('typeID','1')
    root.appendChild(model_type)

    model_item = doc.createElement(model_naming+'.trt')
    #model_item.setAttribute('nameID','1')
    model_type.appendChild(model_item)

    model_infos = {
        'name':str(model_naming),
        'type':'TRA_DL',
        'algorithm':'DenseNet121',
        'framework':'keras',
        'accuracy':str(args.valAcc),
        'trainDataset':project_path.split("/")[-1],
        'trainEpoch':str(args.max_epochs),
        'trainLR':'0.001',
        'class':str(args.class_number),
        'PATH':os.path.abspath(os.path.join(project_path,model_naming+'.trt')),
        'batch':str(args.batch_size),
        'note':'-'
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
    params_txt.write('model_type: ' + str(model_type) + '\n')
    params_txt.write('max_epochs: ' + str(max_epochs) + '\n')
    params_txt.write('batch_size: ' + str(batch_size) + '\n')
    params_txt.write('model_naming: ' + str(model_naming) + '\n')
    params_txt.close()


if __name__ == '__main__':
    project_path = args.data_dir
    model_type = 'baseline'  # 网络类型
    model_naming = project_path.split('/')[-1]
    max_epochs = args.max_epochs
    batch_size = args.batch_size

    x_train, y_train, x_val, y_val, folder_name = read_project(project_path)

    lowProjectPath = model_naming.lower()
    if 'dnn' in lowProjectPath:
        model_name = 'DNN'
        inference_DNN(x_train, y_train, x_val, y_val, folder_name,project_path,model_naming)
    elif 'cnn' in lowProjectPath:
        model_name = 'CNN'
        inference_CNN(x_train, y_train, x_val, y_val, folder_name,project_path,model_naming)
    else:
        assert False, '请在工程目录中包含CNN、DNN中的一个'

    save_params()
    generator_model_documents(args)
    # convert_hdf5_to_trt('HRRP', project_path, args.model_name)
    print("Train Ended:")
