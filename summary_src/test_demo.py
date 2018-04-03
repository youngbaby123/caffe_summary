# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import time
import sys
caffe_root = '../' #改成你自己caffe的根路径
sys.path.insert(0, caffe_root + 'python')
import caffe
import argparse
import logging
from collections import OrderedDict
from test_ import test_net


def Get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size', type=int, default=64)
    parse.add_argument('--img_size', type=int, default=224)
    parse.add_argument('--model_deploy', type=str, default="deploy.prototxt")
    parse.add_argument('--model_weights', default="demo_model.caffemodel")
    parse.add_argument('--data_root', default="./data")
    parse.add_argument('--test_list_file', default="./test.txt")
    parse.add_argument('--label_list_file', default="./label.txt")
    parse.add_argument('--GPU', type=bool, default=False)
    parse.add_argument('--score_th', type=float, default=0.0)
    return parse.parse_args()


def single_test():
    opt = Get_opt()
    task_root = "/home/zkyang/Workspace/DL_code/Caffe_code/caffe/data/chebiao"
    # 检测图像根目录
    opt.data_root = os.path.join(task_root, "Data")
    # 检测图像列表
    opt.test_list_file = os.path.join(task_root, "test.txt")
    img_list = open(opt.test_list_file).readlines()
    # 类别及其对应编号列表
    opt.label_list_file = os.path.join(task_root, "label.txt")
    # GPU切换
    opt.GPU = False
    # 网络结构设置
    opt.model_deploy = os.path.join(task_root, "model/deploy.prototxt")
    # 网络参数设置
    opt.model_weights = os.path.join(task_root, "out/car_0129_iter_200000.caffemodel")
    # 输入图像大小
    opt.img_size = 112

    # 使用的模型导入
    test = test_net(opt)

    # 单张图片检测
    for i, line in enumerate(img_list):
        start_time = time.time()
        img_name, index = line.split()
        img_path = os.path.join(opt.data_root, img_name)
        # 单张图片进行检测
        pred, score, load_time = test.single_test_(img_path)
        test_time = time.time() - start_time
        # 检测结果
        print ("图像路径： {}".format(img_path))
        print ("检测结果汇总\n \t所属编号：{}\n\t所属类别：{}\n\t所属分数： {:.3f}".format(pred, test.index_label[pred], score))
        # 运行时间统计
        print (" 总运行时间： {:.3f}ms\n 图片导入及预处理时间： {:.3f}ms\n 模型运算时间： {:.3f}ms".format(test_time * 1000, load_time * 1000,
                                                                                   (test_time - load_time) * 1000))
        print ("----------------" * 4)

def test_from_dir():
    opt = Get_opt()
    task_root = "/home/zkyang/Workspace/task/Densenet_chebiao/50"
    # 检测图像根目录
    opt.data_root = os.path.join(task_root, "outImage")
    # 检测图像列表
    opt.test_list_file = os.path.join(task_root, "test.txt")
    # img_list = os.listdir(opt.data_root)
    # img_list.sort()
    img_list = ["{}.jpg".format(i) for i in range(1,12)]
    # 类别及其对应编号列表
    opt.label_list_file = os.path.join(task_root, "label.txt")
    # GPU切换
    opt.GPU = False
    # 网络结构设置
    opt.model_deploy = os.path.join(task_root, "deploy.prototxt")
    # 网络参数设置
    opt.model_weights = os.path.join(task_root, "Densenet_50_0306_iter_64000.caffemodel")
    # 输入图像大小
    opt.img_size = 224

    # 使用的模型导入
    test = test_net(opt)

    print ("{}\t{}\t{}\t{}\t{}".format("ID","Image_name","pred_label","score","test_time"))
    # 单张图片检测
    for i, img_name in enumerate(img_list):
        start_time = time.time()
        img_path = os.path.join(opt.data_root, img_name)
        # 单张图片进行检测
        pred, score, load_time = test.single_test_(img_path)
        test_time = time.time() - start_time
        # # 检测结果
        # print ("图像路径： {}".format(img_path))
        # print ("检测结果汇总\n \t所属编号：{}\n\t所属类别：{}\n\t所属分数： {:.3f}".format(pred, test.index_label[pred], score[0][0]))
        # # 运行时间统计
        # print (" 总运行时间： {:.3f}ms\n 图片导入及预处理时间： {:.3f}ms\n 模型运算时间： {:.3f}ms".format(test_time * 1000, load_time * 1000,
        #                                                                            (test_time - load_time) * 1000))
        # print ("----------------" * 4)
        print ("{}\t{}\t{}\t{}\t{}".format(i, img_name, test.index_label[pred], score[0][0], test_time))

if __name__ == '__main__':
    test_from_dir()