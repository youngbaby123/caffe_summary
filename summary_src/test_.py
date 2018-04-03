# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import time
import sys
caffe_root = '../' #改成你自己caffe的根路径
sys.path.insert(0, caffe_root + 'python')
import caffe
from collections import OrderedDict

class test_net():
    def __init__(self,opt):
        # 超参数(Hyperparameters)
        self.test_batch_size = opt.test_batch_size
        self.img_size = opt.img_size
        self.ifgray = opt.ifgray
        if self.ifgray:
            self.mean_ = np.array([108])
        else:
            self.mean_ = np.array([104, 117, 123])

        # 模型导入, 模型选择
        self.net = self.load_model(opt)
        self.data_root = opt.data_root

#        self.test_list_file = opt.test_list_file

        label_list = open(opt.label_list_file, "r").readlines()
        self.label_index = OrderedDict()
        self.index_label = OrderedDict()
        for line in label_list:
            index, label = line.split()
            index = int(index)
            self.label_index[label] = index
            self.index_label[index] = label
        self.class_num = len(label_list)
        self.label_th = True
        self.score_th = opt.score_th

    def load_model(self,opt):
        if opt.GPU == True:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        model_def = opt.model_deploy
        model_weights = opt.model_weights

        net = caffe.Net(model_def, model_weights, caffe.TEST)

        return net

    def single_data_load(self,img):
        # 数据预处理
        img = img.astype(np.float32, copy=False)
        img -= self.mean_
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img *= 0.017
        if self.ifgray:
            blob = np.zeros((1, self.img_size, self.img_size, 1), dtype=np.float32)
            img = img[:, :, np.newaxis]
        else:
            blob = np.zeros((1, self.img_size, self.img_size, 3), dtype=np.float32)

        channel_swap = (0, 3, 1, 2)

        blob[0, :, :, :] = img
        blob = blob.transpose(channel_swap)
        return blob

    def single_test_(self, img_path):
        load_time_start = time.time()
        if self.ifgray:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)
        self.net.blobs['data'].data[...] = self.single_data_load(img)
        load_time = time.time() - load_time_start

        output = self.net.forward()
        output_prob = output['prob'][0]
        pred = output_prob.argmax()
        score = output_prob[pred]
        return pred, score, load_time