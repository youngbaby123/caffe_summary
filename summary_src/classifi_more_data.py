# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
caffe_root = '../' #改成你自己caffe的根路径
sys.path.insert(0, caffe_root + 'python')
import caffe
import argparse
from test_ import test_net
import shutil


def Get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size', type=int, default=64)
    parse.add_argument('--img_size', type=int, default=224)
    parse.add_argument('--model_deploy', type=str, default="deploy.prototxt")
    parse.add_argument('--model_weights', default="demo_model.caffemodel")
    parse.add_argument('--data_root', default="./data")
    parse.add_argument('--out_root', default="./out_predict")
    parse.add_argument('--label_list_file', default="./label.txt")
    parse.add_argument('--GPU', type=bool, default=False)
    parse.add_argument('--ifgray', type=bool, default=False)
    parse.add_argument('--score_th', type=float, default=0.0)

    return parse.parse_args()

def load_file(root, rel_path = "", img_list=[], rel_img_list = []):
    if os.path.isfile(root):
        img_list.append(root)
        rel_img_list.append(rel_path)
    elif os.path.isdir(root):
        for path_i in os.listdir(root):
            sub_root = os.path.join(root, path_i)
            sub_rel_path = os.path.join(rel_path, path_i)
            img_list, rel_img_list = load_file(sub_root, sub_rel_path, img_list, rel_img_list)
    return img_list, rel_img_list

def save_img(img_path, label_, out_root):
    save_path = os.path.join(out_root, "{}".format(label_))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy(img_path, save_path)

def load_file_from_list(data_root, data_list):
    img_path_list = []
    img_list = open(data_list).readlines()
    for line in img_list:
        relpath,index = line.split()[:2]
        img_path = os.path.join(data_root, relpath)
        img_path_list.append(img_path)
    return img_path_list

def single_test():
    opt = Get_opt()
    task_root = "/home/zkyang/Workspace/task/chebiao_model_BG/30_label"
    # 检测图像根目录
    # data_root = "/home/zkyang/Workspace/DL_code/Caffe_code/caffe/data/chebiao_30_0110/"
    opt.data_root = os.path.join(task_root, "Data")
    # 检测结果输出路径
    opt.out_root = os.path.join(task_root, "out_predict_0.5")
    # 类别及其对应编号列表
    opt.label_list_file = os.path.join(task_root, "test_model/densenet_v2.2/label.txt")
    # GPU切换
    opt.GPU = True
    # 网络结构设置
    opt.model_deploy = os.path.join(task_root, "test_model/densenet_v2.2/deploy.prototxt")
    # 网络参数设置
    opt.model_weights = os.path.join(task_root, "test_model/densenet_v2.2/densenet_v2.2_0323_iter_208000.caffemodel")
    # 输入图像大小
    opt.img_size = 112
    # score
    opt.score_th = 0.5

    # 使用的模型导入
    test = test_net(opt)

    # 检测图像列表
    img_list, _ = load_file(opt.data_root, rel_path="", img_list=[], rel_img_list=[])
    # data_list = os.path.join(task_root, "test.txt")
    # img_list = load_file_from_list(opt.data_root, data_list)
    print img_list
    img_num = len(img_list)

    # 单张图片检测
    for i, img_path in enumerate(img_list):
        # 单张图片进行检测
        try:
            pred, score, load_time = test.single_test_(img_path)
            # 检测结果
            if score < opt.score_th:
                label_ = "0"
            else:                   
                label_ = test.index_label[pred]
            save_img(img_path, label_, opt.out_root)
        except:
            continue
        if i % 100 == 0:
            print ("完成度： {}/{}".format(i, img_num))

if __name__ == '__main__':
    single_test()
