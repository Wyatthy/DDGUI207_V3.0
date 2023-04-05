'''训练旧类'''
import os
import sys
import time
import argparse

from train import Pretrain
from config import data_path
from dataProcess import create_dir, read_project


argparser = argparse.ArgumentParser()

argparser.add_argument('--pretrain_epoch', type=int,  help='preTrain epoch number,must biger than 1',default = 2)
argparser.add_argument('--learning_rate', type=float, help='preTrain learning rate', default = 1e-4)
argparser.add_argument('--old_class', type=int, help='number of old class')
argparser.add_argument('--all_class', type=int, help='number of all class')
argparser.add_argument('--memory_size', type=int, help='memory size',default=2000)
argparser.add_argument('--batch_size', type=int, help='batch size',default=32)
argparser.add_argument('--bound', type=float, help='up bound of new class weights',default=0.3)
argparser.add_argument('--random_seed', type=int, help='numpy random seed',default=2022)
argparser.add_argument('--reduce_sample', type=float, help='reduce the number of sample to n%',default = 1.0)
argparser.add_argument('--data_dimension', type=int,  help='[39, 128, 256]',default = 128)
argparser.add_argument('--test_ratio', type=float, help='the ratio of test dataset',default = 0.5)
argparser.add_argument('--work_dir', help='the directory of the training data',default='D:/基于HRRP的Resnet50网络/增量学习/旧类数据')
argparser.add_argument('--time', help='the directory of the training data', default="2022-09-21-21-52-17")
argparser.add_argument('--model_name', help='the directory of the training data', default="model")
argparser.add_argument('--modeldir', help="model saved path")

args = argparser.parse_args()

# 保存参数
def save_params(project_path):
    params_txt = open(project_path+'/pretrain_params_save.txt', 'w')
    params_txt.write('project_path: ' + str(project_path) + '\n')
    params_txt.write('pretrain_epoch: ' + str(pretrain_epoch) + '\n')
    params_txt.write('learning_rate: ' + str(learning_rate) + '\n')
    params_txt.write('batch_size: ' + str(batch_size) + '\n')
    params_txt.write('data_dimension: ' + str(data_dimension) + '\n')
    params_txt.write('memory_size: ' + str(memory_size) + '\n')
    params_txt.write('bound: ' + str(bound) + '\n')
    params_txt.write('random_seed: ' + str(random_seed) + '\n')
    params_txt.write('reduce_sample: ' + str(reduce_sample) + '\n')
    params_txt.write('test_ratio: ' + str(test_ratio) + '\n')
    params_txt.close()


def inference(args):
    current_path = os.path.dirname(__file__)
    os.chdir(current_path)
    create_dir()
    class_name = read_project(project_path)
    args.old_class = len(class_name)
    args.pretrain_epoch = pretrain_epoch
    args.learning_rate = learning_rate
    args.memory_size = memory_size
    args.bound = bound
    args.random_seed = random_seed
    args.batch_size = batch_size
    args.reduce_sample = reduce_sample
    args.data_dimension = data_dimension
    args.raw_data_path = project_path
    args.work_dir = project_path
    args.test_ratio = test_ratio
    args.modeldir = project_path


    # 存储旧类名称
    oldclass_name_txt = open(data_path + '/oldclass_name.txt', 'w')
    for i in range(0, len(class_name)):
        oldclass_name_txt.write(str(class_name[i]) + '\n')
    oldclass_name_txt.close()

    # 开始旧类训练
    print('开始旧类训练')
    pretrain_start = time.time()
    if args.pretrain_epoch != 0:
        preTrain = Pretrain(args.old_class, args.memory_size, args.pretrain_epoch, args.batch_size,
                                args.learning_rate, args.data_dimension, args.work_dir, class_name)
        preTrain.train()
    pretrain_end = time.time()
    print("pretrain_consume_time:", pretrain_end - pretrain_start)
    sys.stdout.flush()


if __name__ == '__main__':
    project_path = args.work_dir
    projectName = args.work_dir.split('/')[-1]
    print('projectName',projectName)
    args.model_name = projectName
    pretrain_epoch = args.pretrain_epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    data_dimension = args.data_dimension
    memory_size = args.memory_size
    bound = args.bound
    random_seed = args.random_seed
    reduce_sample = args.reduce_sample
    test_ratio = args.test_ratio


    save_params(project_path)
    inference(args)
    cmd_onnx2trt="trtexec.exe --explicitBatch --workspace=3072 --minShapes=input:1x1x"+\
                str(data_dimension)+"x1 --optShapes=input:20x1x"+\
                str(data_dimension)+"x1 --maxShapes=input:512x1x"+\
                str(data_dimension)+"x1 --onnx="+args.work_dir + \
                "/pretrain.onnx "+" --saveEngine="+\
                args.work_dir + "/"+ args.model_name+"_pretrain.trt --fp16"
    os.system(cmd_onnx2trt)
    print("Train Ended:")