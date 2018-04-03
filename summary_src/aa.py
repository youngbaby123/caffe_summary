# -*- coding: utf-8 -*-
import os
from PIL import Image
import cv2
import random
import argparse
from collections import OrderedDict


def get_label_dict(label_dir):
    label_list = open(label_dir, "r").readlines()
    label_index = OrderedDict()
    index_label = OrderedDict()
    for line in label_list:
        index, label = line.split()
        index = int(index)
        label_index[label] = index
        index_label[index] = label
    return label_index, index_label

def get_label_all():
    label_dir = "/home/longlongaaago/yang/chebiao_model_test/60_label/label.txt"
    label_all_dir = "/home/longlongaaago/yang/chebiao_model_test/60_label/label_all.txt"
    label_list = open(label_dir, "r").readlines()
    label_index = OrderedDict()
    for line in label_list:
        index, label = line.split()
        index = int(index)
        if index == 1:
            continue
        elif index != 0:
            index -= 1
        label_index[label] = index


    res = []
    for label_ in label_index:
        res.append("{} {}".format(label_index[label_], label_))

    open(label_all_dir, "w").write("\n".join(res))

def get_label_index():
    label_dir = "/home/longlongaaago/yang/chebiao_model_test/60_label/label.txt"
    label_all_dir = "/home/longlongaaago/yang/chebiao_model_test/60_label/label_all.txt"
    label_index_file = "/home/longlongaaago/yang/chebiao_model_test/60_label/label_index.txt"

    label_index, index_label = get_label_dict(label_dir)
    all_label_index, all_index_label = get_label_dict(label_all_dir)

    res = []
    for laebl_ in label_index:
        index_ = label_index[laebl_]
        if not all_label_index.has_key(laebl_):
            all_index_ = 0
        else:
            all_index_ = all_label_index[laebl_]
        res.append("{} {}".format(index_, all_index_))
    open(label_index_file, "w").write("\n".join(res))


def main():
    get_label_index()

if __name__ == '__main__':
    main()