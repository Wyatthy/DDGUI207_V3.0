# coding=utf-8
import os
import time, sys
from config import log_path
from dataProcess import read_txt, split_test_and_train, prepare_pretrain_data, prepare_increment_data, create_dir, read_mat_new, read_project
from train import Pretrain, IncrementTrain, Evaluation
import argparse
from utils import generator_model_documents

argparser = argparse.ArgumentParser()


argparser.add_argument('--pretrain_epoch', type=int,  help='preTrain epoch number,must biger than 1', default=2)
argparser.add_argument('--increment_epoch', type=int,  help='train new model epoch number,must biger than 1',default=3)
argparser.add_argument('--learning_rate', type=float, help='preTrain learning rate',default=1e-4)
argparser.add_argument('--task_size', type=int, help='number of incremental class',default=1)
argparser.add_argument('--old_class', type=int, help='number of old class',default=5)
argparser.add_argument('--old_class_names', type=str, help='names of old class',default="")
argparser.add_argument('--all_class', type=int, help='number of all class',default=6)
argparser.add_argument('--memory_size', type=int, help='memory size',default=2000)
argparser.add_argument('--batch_size', type=int, help='batch size',default=32)
argparser.add_argument('--bound', type=float, help='up bound of new class weights',default=0.3)
argparser.add_argument('--random_seed', type=int, help='numpy random seed',default=2023)
argparser.add_argument('--reduce_sample', type=float, help='reduce the number of sample to n%',default=1.0)
argparser.add_argument('--data_dimension', type=int,  help='[39, 128, 256]',default=128)
argparser.add_argument('--test_ratio', type=float, help='the ratio of test dataset',default=0.5)
argparser.add_argument('--work_dir', help='the directory of the training data',default='N:/207/GUI207_V3.0/work_dirs/HRRP_Resnet50')
argparser.add_argument('--time', help='the directory of the training data', default="2022-09-21-21-52-17")
argparser.add_argument('--model_name', help='the directory of the training data', default="CILmodel")
argparser.add_argument('--modeldir', help="model saved path")
argparser.add_argument('--train_classname', type=str, help="for label2class", default= "defalutStr;")


args = argparser.parse_args()


# 保存参数
def save_params(project_path):
    params_txt = open(project_path+'/params_save.txt', 'w')
    params_txt.write('project_path: ' + str(project_path) + '\n')
    params_txt.write('oldclass_name_select: ' + str(oldclass_name_select) + '\n')
    params_txt.write('pretrain_epoch: ' + str(args.pretrain_epoch) + '\n')
    params_txt.write('increment_epoch: ' + str(args.increment_epoch) + '\n')
    params_txt.write('learning_rate: ' + str(args.learning_rate) + '\n')
    params_txt.write('task_size: ' + str(args.task_size) + '\n')
    params_txt.write('batch_size: ' + str(args.batch_size) + '\n')
    params_txt.write('data_dimension: ' + str(args.data_dimension) + '\n')
    params_txt.write('memory_size: ' + str(args.memory_size) + '\n')
    params_txt.write('bound: ' + str(args.bound) + '\n')
    params_txt.write('random_seed: ' + str(args.random_seed) + '\n')
    params_txt.write('reduce_sample: ' + str(args.reduce_sample) + '\n')
    params_txt.write('test_ratio: ' + str(args.test_ratio) + '\n')
    params_txt.close()


def inference(args, oldclass_name):
    current_path = os.path.dirname(__file__)
    os.chdir(current_path)
    create_dir()
    args.old_class = len(oldclass_name)
    class_name = read_project(project_path, oldclass_name)
    args.train_classname = ';'.join(class_name)+";";
    args.modeldir = project_path

    # 开始旧类训练
    print('开始旧类训练')
    pretrain_start = time.time()
    if args.pretrain_epoch != 0:
        preTrain = Pretrain(args.old_class, args.all_class, args.memory_size, args.pretrain_epoch, args.batch_size,
                                args.learning_rate, args.data_dimension)
        preTrain.train()
    pretrain_end = time.time()
    print("pretrain_consume_time:", pretrain_end - pretrain_start)
    sys.stdout.flush()

    # 开始增量训练
    print('开始增量训练')
    increment_start = time.time()
    if args.increment_epoch != 0:
        incrementTrain = IncrementTrain(args.memory_size, args.all_class, args.all_class - args.old_class,
                                        args.task_size, \
                                        args.increment_epoch, args.batch_size, args.learning_rate, args.bound,
                                        args.reduce_sample, args.work_dir, class_name, args.data_dimension)
        incrementTrain.train()
    increment_end = time.time()
    print("pretrain_consume_time:", increment_end - increment_start)
    sys.stdout.flush()
    valacc = incrementTrain.result.item()
    args.accuracy = round(valacc, 2)  # 增量训练在测试集上的最高准确率

    # 开始验证
    evaluation = Evaluation(args.all_class, args.all_class - args.old_class, args.batch_size, args.data_dimension)
    old_oa, new_oa, all_oa, metric = evaluation.evaluate()

    timeArray = time.localtime(time.time())
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    logFile = open(args.work_dir + "/verification_classification_report.txt", 'w')
    logFile.write("\n" + str(otherStyleTime) + "\n" +
                  str(args) + "\n" +
                  "pretrain_consume_time: " + str(pretrain_end - pretrain_start) + "\n" +
                  "increment_consume_time: " + str(increment_end - increment_start) + "\n" +
                  "Old_OA:" + str(old_oa) + "\n" +
                  "New_OA:" + str(new_oa) + "\n" +
                  "All_OA:" + str(all_oa) + "\n\n" + str(metric))


if __name__ == '__main__':
    project_path = args.work_dir
    projectName = args.work_dir.split('/')[-1];
    args.model_name = projectName;
    oldclass_name_select = args.old_class_names.split(';')
    folder_path = project_path + '/train'  # 所有文件夹所在路径
    file_name = os.listdir(folder_path)  # 读取所有文件夹，将文件夹名存在列表中
    folder_name = []
    for i in range(0, len(file_name)):
        # 判断文件夹与文件
        if os.path.isdir(folder_path + '/' + file_name[i]):
            folder_name.append(file_name[i])
    folder_name.sort()  # 按文件夹名进行排序
    args.all_class = len(folder_name)

    if(len(oldclass_name_select)==1):#如果没有传旧类
        oldclass_name_select = folder_name[:-1]
    else:
        oldclass_name_select = oldclass_name_select[:-1]

    print("旧类包括：",oldclass_name_select)
    
    save_params(project_path)
    inference(args, oldclass_name_select)
    generator_model_documents(args)

    cmd_onnx2trt="trtexec.exe --explicitBatch --workspace=3072 --minShapes=input:1x1x"+\
                str(args.data_dimension)+"x1 --optShapes=input:20x1x"+\
                str(args.data_dimension)+"x1 --maxShapes=input:512x1x"+\
                str(args.data_dimension)+"x1 --onnx="+args.work_dir + \
                "/incrementModel.onnx "+" --saveEngine="+\
                args.work_dir + "/"+ args.model_name+".trt --fp16"
    os.system(cmd_onnx2trt)

    
    print("Train Ended")
    sys.stdout.flush()