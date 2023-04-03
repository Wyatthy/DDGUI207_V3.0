# -*- coding: UTF-8 -*-
import os
import sys
import re
import abfc_model
import tensorflow as tf
import numpy as np
from utils import BatchCreate
import tensorflow.keras as keras
import tensorflow.compat.v1 as tfv1
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from data_process import show_feature_selection, show_confusion_matrix, read_project, train_acc, val_acc, mycopyfile
from data_process import storage_characteristic_matrix, data_normalization, data_norm_hrrp, show_feature_weights
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import shutil
import argparse

model_num="1"
parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--data_dir', help='the directory of the training data',default="db/datasets/local_dir/基于特征数据的ABFC网络")
parser.add_argument('--batch_size', type=int, help='the number of batch size',default=32)
parser.add_argument('--max_epochs', type=int, help='the number of epochs',default=1)
parser.add_argument('--class_number', type=int, help="class_number", default="6")
parser.add_argument('--fea_num', type=int, help="fea_num", default=128)
parser.add_argument('--fea_start', type=int, help="fea_start", default=16)
parser.add_argument('--fea_step', type=int, help="fea_step", default=16)
parser.add_argument('--data_type', help='the type of the training data', default='HRRP')
args = parser.parse_args()

def test(train_X, train_Y, val_X, val_Y, output_size, fea_num, work_dir):
    train_model = keras.models.Sequential([
        keras.layers.Conv1D(64, kernel_size=1, padding='valid', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(output_size, activation='softmax')])
    train_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    # 建立存储保存后的模型的文件夹
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        # os.makedirs(work_dir + '/model')
    save_model_path = work_dir + '/'+'ABFC_feature_' + str(fea_num) + '.hdf5'
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.99, patience=3, verbose=0, min_lr=0.0001)
    checkpoint = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learn_rate_reduction]
    h = train_model.fit(train_X, train_Y, batch_size=batch_size, epochs=max_epochs, shuffle=True,
                        validation_data=(val_X, val_Y), callbacks=callbacks_list, verbose=0, validation_freq=1)
    h_parameter = h.history
    train_acc(max_epochs, h_parameter['accuracy'], work_dir, fea_num)
    val_acc(h_parameter['val_accuracy'], work_dir, fea_num)
    val_model = keras.models.load_model(save_model_path)
    Y_val = np.argmax(val_Y, axis=1)
    Y_pred = np.argmax(val_model.predict(val_X), axis=1)
    args.valAcc=round(max(h_parameter['val_accuracy'])*100,2)
    return Y_val, Y_pred


def run_test(A, train_X, train_Y, test_X, test_Y, output_size, f_start, f_end, f_interval, work_dir):
    all_characteristic_matrix = []  # 存储特征矩阵
    attention_weight = A.mean(0)
    ABFC_wight_rank = list(np.argsort(attention_weight))[::-1]
    ac_score_list = []
    predicted_label_list = []
    for K in range(f_start, f_end + 1, f_interval):
        use_train_x = np.expand_dims(train_X[:, ABFC_wight_rank[:K]], axis=-1)
        use_test_x = np.expand_dims(test_X[:, ABFC_wight_rank[:K]], axis=-1)
        label_class, predicted_class = test(use_train_x, train_Y, use_test_x, test_Y, output_size, K, work_dir)
        characteristic_matrix, accuracy_every_class, accuracy = storage_characteristic_matrix(predicted_class, label_class, output_size)
        one_predict_label = np.zeros((2, len(predicted_class)))
        one_predict_label[0, :] = predicted_class
        one_predict_label[1, :] = label_class

        print('Using Top {} features| accuracy:{:.4f}'.format(K, accuracy))
        sys.stdout.flush()
        all_characteristic_matrix.append(characteristic_matrix)
        ac_score_list.append(accuracy)
        predicted_label_list.append(one_predict_label)

    return ac_score_list, all_characteristic_matrix, predicted_label_list


def run_train(sess, train_X, train_Y, val_X, val_Y, train_step, batch_size):
    X = tfv1.get_collection('input')[0]
    Y = tfv1.get_collection('output')[0]

    Iterator = BatchCreate(train_X, train_Y)
    for step in range(1, train_step + 1):
        if step % 100 == 0:
            # val_loss, val_accuracy = sess.run(tfv1.get_collection('validate_ops'), feed_dict={X: train_X, Y: train_Y})
            val_loss, val_accuracy = sess.run(tfv1.get_collection('validate_ops'), feed_dict={X: val_X, Y: val_Y})

            print('[%4d] ABFC-loss:%.12f ABFC-accuracy:%.6f' % (step, val_loss, val_accuracy))
            sys.stdout.flush()
        xs, ys = Iterator.next_batch(batch_size)
        _, A = sess.run(tfv1.get_collection('train_ops'), feed_dict={X: xs, Y: ys})
    return A


def inference(train_step, batchsize, f_start, f_end, f_interval, work_dir, data_type):
    train_X, train_Y, val_X, val_Y, class_label = read_project(work_dir)
    if data_type == 'HRRP':
        train_X, val_X = data_norm_hrrp(train_X), data_norm_hrrp(val_X)
    if data_type == 'FEATURE':
        train_X, val_X = data_normalization(train_X), data_normalization(val_X)
    Train_Size = len(train_X)
    total_batch = Train_Size / batchsize
    abfc_model.build(total_batch, len(train_X[0]), len(train_Y[0]))
    with tfv1.Session() as sess:  # 创建上下文
        tfv1.global_variables_initializer().run()  # 初始化模型参数
        print('== Get feature weight by using ABFC ==')
        sys.stdout.flush()
        A = run_train(sess, train_X, train_Y, val_X, val_Y, train_step, batchsize)
    print('== The Evaluation of ABFC ==')
    sys.stdout.flush()
    at = A.mean(0)
    save_A_weight_path = work_dir + '/'+'feature_weights.txt'
    feature_weights = open(save_A_weight_path, 'w', encoding='utf-8')
    for i in range(0, len(at)):
        feature_weights.write(str(at[i])+'\n')
    feature_weights.close()
    # show_feature_weights(at, work_dir)
    A_wight_rank = list(np.argsort(at))[::-1]
    save_A_path = work_dir + '/'+'attention_weight_rank.txt'
    attention_weights = open(save_A_path, 'w', encoding='utf-8')
    for i in range(0, len(A_wight_rank)):
        attention_weights.write(str(A_wight_rank[i])+'\n')
    attention_weights.close()
    ac_score_list, characteristic_matrix_summary, predicted_label_list = run_test(A, train_X, train_Y, val_X, val_Y, len(train_Y[0]), f_start, f_end, f_interval, work_dir)

    show_feature_selection(ac_score_list, f_start, f_end, f_interval, work_dir)
    optimal_result = ac_score_list.index(max(ac_score_list))
    optimal_classification_report = predicted_label_list[optimal_result]
    use_fea_num = f_start + f_interval*optimal_result
    show_feature_weights(at, work_dir, A_wight_rank[:use_fea_num])
    mycopyfile(work_dir+'/train_acc_save/training_accuracy_'+str(use_fea_num)+'.jpg', work_dir, 'training_accuracy.jpg')
    shutil.rmtree(work_dir+'/train_acc_save/')
    mycopyfile(work_dir +'/val_acc_save/verification_accuracy_'+str(use_fea_num)+'.jpg', work_dir, 'verification_accuracy.jpg')
    shutil.rmtree(work_dir +'/val_acc_save/')
    show_confusion_matrix(class_label, characteristic_matrix_summary[optimal_result], work_dir+'/verification_confusion_matrix.jpg')
    classification_report_txt = open(work_dir+'/verification_classification_report.txt', 'w')
    classification_report_txt.write(classification_report(optimal_classification_report[1], optimal_classification_report[0], digits=4))
    classification_report_txt.close()
    print(classification_report(optimal_classification_report[1], optimal_classification_report[0], digits=4))
    print(optimal_result)
    sys.stdout.flush()
    attention_params_save = open(work_dir + '/'+'attention_params.txt', 'w', encoding='utf-8')
    attention_params_save.write(str(ac_score_list.index(max(ac_score_list))+1)+'\n')
    attention_params_save.write(str(f_start)+'\n')
    attention_params_save.write(str(f_interval)+'\n')
    attention_params_save.close()
    sys.stdout.flush()
    global model_num
    model_num = int(f_start) + int(ac_score_list.index(max(ac_score_list))) * int(f_interval)

def generator_model_documents(args):
    from xml.dom.minidom import Document
    doc = Document()  #创建DOM文档对象
    root = doc.createElement('ModelInfo') #创建根元素
    doc.appendChild(root)
    
    model_type = doc.createElement(data_type)
    #model_type.setAttribute('typeID','1')
    root.appendChild(model_type)

    model_item = doc.createElement(model_naming)
    #model_item.setAttribute('nameID','1')
    model_type.appendChild(model_item)

    model_infos = {
        'Model_DataType':str(args.data_type),
        'Model_Name':str(model_naming),
        'Model_Algorithm':'ABFC',
        'Model_AlgorithmType':'特征权重模型',
        'Model_AccuracyOnTrain':'-',
        'Model_AccuracyOnVal':str(args.valAcc),
        'Model_Framework':'Keras',
        'Model_TrainDataset':args.data_dir.split("/")[-1],
        'Model_TrainEpoch':str(args.max_epochs),
        'Model_TrainLR':'0.001',
        'Model_NumClassCategories':str(args.class_number), 
        'Model_Path':os.path.abspath(os.path.join(project_path,model_naming+'.trt')),
        'Model_TrainBatchSize':str(args.batch_size),
        'Model_Note':'-',
        'Model_Type':'ABFC',
        'ProjectType':str(args.data_type)
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
    params_txt.write('feature_number: ' + str(fea_num) + '\n')
    params_txt.write('max_epochs: ' + str(max_epochs) + '\n')
    params_txt.write('batch_size: ' + str(batch_size) + '\n')
    params_txt.write('feature_start: ' + str(fea_start) + '\n')
    params_txt.write('feature_step: ' + str(fea_step) + '\n')
    params_txt.write('data_type: ' + str(data_type) + '\n')
    params_txt.close()


if __name__ == '__main__':
    pid = os.getpid()
    print('pid$',pid,"pid$")
    sys.stdout.flush()
    project_path = args.data_dir  # 工程目录
    # 分割路径，获取文件名
    model_naming = project_path.split('/')[-1]
    model_name = 'ABFC'
    fea_num = args.fea_num  # 单样本包含数值个数
    max_epochs = args.max_epochs  # 训练轮数
    batch_size = args.batch_size  # 批处理数量
    fea_start = args.fea_start  # 首次选择特征的个数
    fea_step = args.fea_step  # 选择特征的步长
    data_type = args.data_type  # 输入数据的类型，'HRRP'或'FEATURE'

    file_name = os.listdir(project_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(project_path+'/'+file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    args.class_number=len(folder_name)

    save_params()
    inference(max_epochs, batch_size, fea_start, fea_num, fea_step, project_path, data_type)
    generator_model_documents(args)
    shutil.copy(project_path + '/ABFC_feature_' + str(model_num) + '.hdf5',project_path + '/'+model_naming+'.hdf5')
    shutil.copy("../sources/modelIMG/ABFC.png",project_path + '/'+model_naming+'.png')
    # cmd="python ./api/bashs/hdf52trt.py --model_type ABFC --work_dir "+ \
    #     project_path+" --model_name "+model_naming+" --abfcmode_Idx " + str(model_num)
    # os.system(cmd)
    print("Train Ended:")
