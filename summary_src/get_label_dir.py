# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
import shutil

def load_file(root, rel_path = "", img_list=[], rel_img_list = []):
    if os.path.isfile(root):
        # if check_img(root):
        img_list.append(root)
        rel_img_list.append(rel_path)
    elif os.path.isdir(root):
        for path_i in os.listdir(root):
            sub_root = os.path.join(root, path_i)
            sub_rel_path = os.path.join(rel_path, path_i)
            img_list, rel_img_list = load_file(sub_root, sub_rel_path, img_list, rel_img_list)
    return img_list, rel_img_list

def load_dir(root, root_th = 1, dir_list=[]):
    if root_th <= 0:
        return dir_list
    elif root_th == 1:
        dir_list.append(root)
    elif os.path.isdir(root):
        for path_i in os.listdir(root):
            sub_root = os.path.join(root, path_i)
            if os.path.isfile(sub_root):
                dir_list.append(root)
                break
            else:
                dir_list = load_dir(sub_root, root_th-1, dir_list)
    return dir_list


def main():
    root = "/home/zkyang/Data_set/车标/train/Data/细分"
    out_root = "/home/zkyang/Data_set/车标/train/train_0330"
    sub_root_list = load_dir(root, root_th = 2, dir_list=[])
    for sub_root_i in sub_root_list:
        img_list, _ = load_file(sub_root_i, rel_path = "", img_list=[], rel_img_list = [])
        out_root_i = os.path.join(out_root, os.path.basename(sub_root_i))
        if not os.path.exists(out_root_i):
            os.makedirs(out_root_i)
        for img_i in img_list:
            shutil.copy(img_i, out_root_i)


if __name__ == '__main__':
    main()