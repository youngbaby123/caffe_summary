# -*- coding: utf-8 -*-
import os
import test_
import argparse
from collections import OrderedDict
import numpy as np
import time
import summary_
import shutil

def Get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size', type=int, default=64)
    parse.add_argument('--img_size', type=int, default=224)
    parse.add_argument('--model_deploy', type=str, default="deploy.prototxt")
    parse.add_argument('--model_weights', default="demo_model.caffemodel")
    parse.add_argument('--model_name', default="tiny_mobile")
    parse.add_argument('--data_root', default="./data")
    parse.add_argument('--test_list_file', default="./test.txt")
    parse.add_argument('--label_list_file', default="./label.txt")
    parse.add_argument('--GPU', type=bool, default=False)
    parse.add_argument('--score_th', type=float, default=0.0)
    parse.add_argument('--IF_SAVE', type=bool, default=False)
    parse.add_argument('--res_data_root', type=str, default="./out_res_data")
    return parse.parse_args()


def save_res(res, model_name, save_path):
    file = open(save_path, "a+")
    save_res_ = OrderedDict()
    for i in res["tp"]:
        save_res_[i]=[]
        #save_res_[i].append(str(model_name))
        save_res_[i].append(str(i))
    for i in save_res_:
        for summary_label_i in ["label","tp", "fp", "fn", "precision", "recall", "AP"]:
            save_res_[i].append(str(res[summary_label_i][i]))
        for summary_label_i in ["accuracy", "test_speed", "load_speed", "file_size"]:
            save_res_[i].append(str(res[summary_label_i]))
    save_txt = []
    for i in save_res_:
        save_i = "\t".join(save_res_[i])
        save_txt.append(save_i)
    file.write("\n".join(save_txt)+"\n")
    file.close()


def get_FileSize(filePath):
    # filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024)

    return round(fsize,2)

def Save_res_data(img_path, res_data_root, true_label, pred_label, index_label):
    save_set = "True" if true_label == pred_label else "False"
    res_dir = os.path.join(res_data_root, save_set, "{}".format(index_label[pred_label]))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    shutil.copy(img_path, res_dir)


def all_summary():
    opt = Get_opt()
    task_root = "/home/zkyang/Workspace/task/Densenet_chebiao/chebiao_26_all"
    opt.data_root = os.path.join(task_root, "Data")
    opt.test_list_file = os.path.join(task_root, "test.txt")
    opt.label_list_file = os.path.join(task_root, "label.txt")
    # opt.GPU = True
    opt.GPU = False
    # 网络结构设置
    opt.model_deploy = os.path.join(task_root, "model_3/deploy.prototxt")
    # 网络参数设置
    opt.model_weights = os.path.join(task_root, "out_3/tinyDes_26_all_0315_iter_100000.caffemodel")
    # 输入图像大小
    opt.img_size = 112
    # 分数阈值
    # opt.score_th = 0.8

    # 输出结果图片是否剖保存
    #opt.IF_SAVE = True
    opt.res_data_root = os.path.join(task_root, "out_res_tinydes_3")

    # 使用的模型导入
    test = test_.test_net(opt)

    summary_label = ["ID", "label", "tp", "fp", "fn", "precision", "recall", "AP", "accuracy", "test_speed",
                     "load_speed", "file_size"]
    save_file_path = os.path.join(task_root, "summary_CPU_tinydes_3.txt")
    save_file = open(save_file_path, "w+")
    save_file.write("\t".join(summary_label)+"\n")
    save_file.close()

    img_list = open(opt.test_list_file).readlines()
    res = OrderedDict()

    res[opt.model_name] = {}
    #模型的参数个数统计以及模型存储大小
    res[opt.model_name]["file_size"] = get_FileSize(opt.model_weights)

    sum_load_time = 0
    for i, line in enumerate(img_list[:110]):
        if i==10:
            start_time = time.time()
        img_name, index = line.split()
        img_path = os.path.join(opt.data_root, img_name)
        pred, score, load_time = test.single_test_(img_path)
        if i > 9:
            sum_load_time += load_time
    end_time = time.time()

    res[opt.model_name]["test_speed"] = 1.0 * (end_time - start_time) / min(len(img_list)-10,100) * 1000
    res[opt.model_name]["load_speed"] = 1.0 * (sum_load_time) / min(len(img_list)-10,100) * 1000
    print ("Average single test time: {}".format(res[opt.model_name]["test_speed"]))
    print ("Average single data load time: {}".format(res[opt.model_name]["load_speed"]))

    opt.GPU = True
    all_pred = []
    all_score = []
    all_true = []
    for i, line in enumerate(img_list):
        img_name, index = line.split()
        true_label = int(index)
        all_true.append(true_label)
        img_path = os.path.join(opt.data_root, img_name)
        pred, score, load_time = test.single_test_(img_path)
        all_pred.append(pred)
        # print (pred,true_label)
        all_score.append(score)
        if opt.IF_SAVE:
            Save_res_data(img_path, opt.res_data_root, true_label, pred, test.index_label)
        if i%100 == 0:
            print ("完成度： {}/{}".format(i,len(img_list)))

    score_th = None
    summary = summary_.result_summary(np.array(all_pred), np.array(all_true), np.array(all_score), score_th=score_th)
    res[opt.model_name]["label"] = test.index_label
    res[opt.model_name]["tp"] = summary.Get_TP()
    res[opt.model_name]["fp"] = summary.Get_FP()
    res[opt.model_name]["fn"] = summary.Get_FN()
    res[opt.model_name]["precision"] = summary.Get_Precision()
    res[opt.model_name]["recall"] = summary.Get_Recall()
    res[opt.model_name]["accuracy"] = summary.Get_Accuracy()
    res[opt.model_name]["AP"] = summary.Get_AP()
    print (res[opt.model_name])

    save_res(res[opt.model_name], opt.model_name, save_file_path)


if __name__ == '__main__':
    all_summary()