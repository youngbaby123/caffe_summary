# -*- coding: utf-8 -*-
import os
from PIL import Image
import cv2
import random
import argparse
from collections import OrderedDict

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Pre-process the classfication dataset')
    parser.add_argument('--data_root', dest='data_root',
                        default=None, type=str,
                        help='Data root to be preprocessed')
    parser.add_argument('--train_ratio', dest='train_ratio',
                        default=0.8, type=float,
                        help='Ratio of the training data')
    parser.add_argument('--val_ratio', dest='val_ratio',
                        default=0.1, type=float,
                        help='Ratio of the validation data')
    parser.add_argument('--test_ratio', dest='test_ratio',
                        default=0.1, type=float,
                        help='Ratio of the test data')
    parser.add_argument('--save_root', dest='save_root',
                        default='./', type=str,
                        help='the train/val/test list path to be saved')
    parser.add_argument('--if_has_bg', dest='if_has_bg',
                        default=False, type=bool,
                        help='IF has background data (the index ).')
    parser.add_argument('--max_num', dest='max_num',
                        default=1000, type=int,
                        help='max number of each label')
    parser.add_argument('--bg_num', dest='max_num',
                        default=3000, type=int,
                        help='max number of background label')
    parser.add_argument('--data_type', dest='data_type',
                        default="train", type=str,
                        help='train or test (just test)')
    parser.add_argument('--label_file', dest='label_file',
                        default="./label.txt", type=str,
                        help='the path of label.txt')
    parser.add_argument('--if_check_img', dest='if_check_img',
                        default=False, type=bool,
                        help='IF need check input image data.')

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def check_img(img_path):
    try:
        img = Image.open(img_path)
        img.verify()
        img = cv2.imread(img_path)
        if not img is None:
            return True
        else:
            return False
    except Exception as e:
        print (e.message)
        return False

def load_file(root, rel_path = "", img_list=[], rel_img_list = [], if_check_img = False):
    if os.path.isfile(root):
        if if_check_img:
            check = check_img(root)
        else:
            check = True
        if check:
            img_list.append(root)
            rel_img_list.append(rel_path)
    elif os.path.isdir(root):
        for path_i in os.listdir(root):
            sub_root = os.path.join(root, path_i)
            sub_rel_path = os.path.join(rel_path, path_i)
            img_list, rel_img_list = load_file(sub_root, sub_rel_path, img_list, rel_img_list)
    return img_list, rel_img_list

def split_data_list(sub_file_list, num_list, label_index, data_dict={}):
    data_dict["train"] += ["{} {}".format(file_i, label_index) for file_i in sub_file_list[: num_list[0]]]
    data_dict["val"] += ["{} {}".format(file_i, label_index) for file_i in sub_file_list[num_list[0] : num_list[1]]]
    data_dict["test"] += ["{} {}".format(file_i, label_index) for file_i in sub_file_list[num_list[1] : num_list[2]]]
    return data_dict

def save_list(all_data_dict, args):
    label_list = []
    data_dict ={"train":[],"val":[],"test":[]}
    for label_ in all_data_dict:
        label_index = all_data_dict[label_]["label_index"]
        sub_file_list = all_data_dict[label_]["file_list"]
        max_num = all_data_dict[label_]["max_num"]
        random.shuffle(sub_file_list)
        num_ = min(max_num, len(sub_file_list))

        if args.data_type == "train" and (label_index == 0 and args.if_has_bg == True):
            num_list = [num_, num_, num_]
            label_list.insert(0, "{} {}".format(0, label_))
        elif args.data_type == "train":
            num_list = [int(num_ * args.train_ratio), int(num_ * (args.train_ratio + args.val_ratio)), num_]
            label_list.append("{} {}".format(label_index, label_))
        else:
            num_list = [0, 0, len(sub_file_list)]

        data_dict = split_data_list(sub_file_list, num_list, label_index, data_dict=data_dict)

    for task_i in data_dict:
        if len(data_dict[task_i]) != 0:
            open(os.path.join(args.save_root, "{}.txt".format(task_i)), "wb+").write("\n".join(data_dict[task_i]))
    if args.data_type != "test":
        open(os.path.join(args.save_root, "label.txt"), "wb+").write("\n".join(label_list))


def get_label_index(label_file):
    res = {}
    label_list = open(label_file, "r").readlines()
    for line_i in label_list:
        index, label = line_i.split()
        res[label] = index
    return res

def Get_data_list(args, bg_list=[]):
    label_name = [label_i for label_i in os.listdir(args.data_root)]
    print (label_name)
    list.sort(label_name)
    if args.data_type == "test":
        test_label_index = get_label_index(args.label_file)

    if args.if_has_bg:
        index_ = 1
    else:
        index_ = 0

    data_set = OrderedDict()
    for label_ in label_name:
        max_num = args.max_num
        if args.data_type == "test":
            if test_label_index.has_key(label_):
                label_index = test_label_index[label_]
            else:
                label_index = 0
        elif label_ in bg_list:
            label_index = 0
            max_num = args.bg_num
        else:
            label_index = index_
            index_ += 1

        sub_data_root = os.path.join(args.data_root, label_)
        data_set[label_] = {}
        _, data_set[label_]["file_list"] = load_file(sub_data_root, rel_path = label_, img_list=[], rel_img_list = [])
        data_set[label_]["label_index"] = label_index
        data_set[label_]["max_num"] = max_num
    return data_set


def main():
    args = parse_args()
    args.data_root = "/home/zkyang/Workspace/task/chebiao_model_test/honghuang/densenet_v2.3_60_v2/Data"
    args.save_root = "/home/zkyang/Workspace/task/chebiao_model_test/honghuang/densenet_v2.3_60_v2"
    args.max_num = 2000
    args.bg_num = 6000

    bg_list = ["0", "background", "fake_background", "Negative", "negative"]
    args.if_check_img = False
    args.if_has_bg = True

    args.data_type = "test"
    args.label_file = "/home/zkyang/Workspace/task/chebiao_model_test/honghuang/densenet_v2.3_60_v2/label_all.txt"


    print ("Start get data list.")
    data_set = Get_data_list(args, bg_list)
    print ("Get done.")
    print ("Start save data list.")
    save_list(data_set, args)
    print ("Save done.")


if __name__ == '__main__':
    main()
