from char_stats import CharStats
import random
import os
import sys

sys.path.append("..")
from ppocr.data.imaug import RecAug, operators
import cv2
from ppocr.data.imaug.cut_aug import RandomErasing, LineErasing, SingleErasing, CropErasing

out_path = "./../train_data/tmp/"
gt_file = './../train_data/tmp/labeltrain.txt'
cnt = 0
thread = 2


def augment(data):
    aug = RecAug(True, 0.8)
    data = aug(data)
    return data


def test(data):
    img = data["image"]
    aug_list = ['RandomErasing', 'SingleErasing', 'LineErasing', 'CropErasing']
    idx = random.randint(0, len(aug_list) - 1)
    op = eval(aug_list[idx])
    re = op()
    data["image"] = re(img, 5)
    return data


def parse(c, n, labels, thread):
    d = thread - n
    label_list = list(labels)
    data_list = []
    print(str.format("== parse with {0}, totals: {1} ==", c, d))
    for i in range(d):
        path, label = label_list[random.randint(0, len(label_list) - 1)].strip().split('\t')
        src_path = out_path + path
        if not os.path.exists(src_path):
            raise Exception("{} does not exist!".format(src_path))
        with open(src_path, 'rb') as f:
            img = f.read()
            data = {"image": img, "label": label}
            data = operators.DecodeImage("BGR", False)(data)
            data = augment(data)
            data_list.append(data)
    return data_list


def format_path(cnt):
    cnt = str(cnt)
    return "Train_" + cnt.zfill(6) + ".jpg"


def output(data_list, gt_file):
    global cnt
    with open(gt_file, 'a', encoding='UTF-8-sig') as f:
        for data in data_list:
            path = format_path(cnt)
            cnt += 1
            label = path + "\t" + data["label"]
            f.write(label + '\n')
            cv2.imwrite(out_path + path, data["image"])


def main():
    cs = CharStats(gt_file)
    lows = cs.get_char_list()
    global cnt, thread
    cnt = cs.size
    for each in lows:
        c, n = each[0], each[1]
        labels = cs.char2labels(c)
        if n < thread:
            data_list = parse(c, n, labels, thread)
            output(data_list, gt_file)


if __name__ == "__main__":
    main()
