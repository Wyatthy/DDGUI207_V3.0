'增量训练'
import os
import time, sys
from config import data_path
from dataProcess import create_dir, read_project_new
from train import Pretrain, IncrementTrain, Evaluation
import argparse
from utils import generator_model_documents

argparser = argparse.ArgumentParser()


argparser.add_argument('--pretrain_epoch', type=int,  help='preTrain epoch number,must biger than 1',default=2)
argparser.add_argument('--increment_epoch', type=int,  help='train new model epoch number,must biger than 1',default=5)
argparser.add_argument('--learning_rate', type=float, help='preTrain learning rate', default=1e-4)
argparser.add_argument('--task_size', type=int, help='number of incremental class')
argparser.add_argument('--old_class', type=int, help='number of old class',default=6)
argparser.add_argument('--all_class', type=int, help='number of all class')
argparser.add_argument('--memory_size', type=int, help='memory size',default=2000)
argparser.add_argument('--batch_size', type=int, help='batch size',default=32)
argparser.add_argument('--bound', type=float, help='up bound of new class weights',default=0.3)
argparser.add_argument('--random_seed', type=int, help='numpy random seed',default=2022)
argparser.add_argument('--reduce_sample', type=float, help='reduce the number of sample to n%',default=1.0)
argparser.add_argument('--data_dimension', type=int,  help='[39, 128, 256]',default=128)
argparser.add_argument('--test_ratio', type=float, help='the ratio of test dataset',default=0.5)
argparser.add_argument('--work_dir', help='the directory of the training data',default='D:/基于HRRP的Resnet50网络/增量学习/新类数据')
argparser.add_argument('--time', help='the directory of the training data', default="2022-09-21-21-52-17")
argparser.add_argument('--model_name', help='the directory of the training data', default="model")
argparser.add_argument('--modeldir', help="model saved path")

args = argparser.parse_args()

# 保存参数
def save_params(project_path):
    params_txt = open(project_path+'/params_save.txt', 'w')
    params_txt.write('project_path: ' + str(project_path) + '\n')
    params_txt.write('oldclass_num: ' + str(oldclass_num) + '\n')
    params_txt.write('increment_epoch: ' + str(increment_epoch) + '\n')
    params_txt.write('learning_rate: ' + str(learning_rate) + '\n')
    # params_txt.write('task_size: ' + str(task_size) + '\n')
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
    args.old_class = oldclass_num
    class_name = read_project_new(args.work_dir, args.old_class)
    args.all_class = len(class_name) + args.old_class
    args.increment_epoch = increment_epoch
    args.learning_rate = learning_rate
    args.task_size = len(class_name)
    args.memory_size = memory_size
    args.bound = bound
    args.random_seed = random_seed
    args.batch_size = batch_size
    args.reduce_sample = reduce_sample
    args.data_dimension = data_dimension
    args.raw_data_path = args.work_dir
    args.work_dir = args.work_dir
    args.test_ratio = test_ratio
    args.modeldir = args.work_dir

    # 合并旧类新类名称
    oldclass_name = open(data_path + '/oldclass_name.txt', 'r', encoding='utf-8').read().splitlines()
    oldclass_name.extend(class_name)
    class_name = oldclass_name

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

    logFile = open(project_path + "/verification_classification_report.txt", 'w')
    logFile.write("\n" + str(otherStyleTime) + "\n" +
                  str(args) + "\n" +
                  "increment_consume_time: " + str(increment_end - increment_start) + "\n" +
                  "Old_OA:" + str(old_oa) + "\n" +
                  "New_OA:" + str(new_oa) + "\n" +
                  "All_OA:" + str(all_oa) + "\n\n" + str(metric))


if __name__ == '__main__':

    project_path = os.path.split(os.path.split(args.work_dir)[0])[0]
    projectName = args.work_dir.split('/')[-3]
    print('projectName',projectName)
    args.model_name = projectName



    oldclass_num = args.old_class
    increment_epoch = args.increment_epoch
    learning_rate = args.learning_rate
    # task_size = args.task_size
    batch_size = args.batch_size
    data_dimension = args.data_dimension
    memory_size = args.memory_size
    bound = args.bound
    random_seed = args.random_seed
    reduce_sample = args.reduce_sample
    test_ratio = args.test_ratio

    save_params(project_path)
    inference(args)

    generator_model_documents(args)

    cmd_onnx2trt="trtexec.exe --explicitBatch --workspace=3072 --minShapes=input:1x1x"+\
                str(data_dimension)+"x1 --optShapes=input:20x1x"+\
                str(data_dimension)+"x1 --maxShapes=input:512x1x"+\
                str(data_dimension)+"x1 --onnx="+project_path + \
                "/incrementModel.onnx "+" --saveEngine="+\
                project_path + "/"+ args.model_name+".trt --fp16"
    os.system(cmd_onnx2trt)

    print("Train Ended:")