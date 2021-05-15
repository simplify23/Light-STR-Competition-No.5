import math
import os
import platform
import sys
import matplotlib.pyplot as plt
import numpy as np

if platform.system().lower() == 'windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams["font.family"] = 'Arial Unicode MS'

class CharStats():
    def __init__(self, gt_file):
        self.__gt_file = gt_file
        self.__char_dict = {} #char -> frequence mapping
        self.__gt_list = []
        self.max_len = 0
        self.__char2labels_dict = {} #char -> gt paths  mapping
        self.size = 0
        self.__dict_label()

    def __dict_label(self):
        with open(self.__gt_file, 'r', encoding='UTF-8-sig') as f:
            all = f.readlines()
            self.size = len(all)
            print("== self.size", self.size, len(all))
            for each in all:
                path, text = each.strip().split('\t')
                for i in text:
                    value = self.__char_dict.setdefault(i, 0)
                    value += 1
                    self.__char_dict.update({i: value})
                    #存一下字符所在图片的映射
                    labels = self.__char2labels_dict.setdefault(i, set())
                    labels.add(each)
                self.max_len = len(text) if self.max_len < len(text) else self.max_len
                self.__gt_list.append(text)

    def get_char_list(self):
        return sorted(self.__char_dict.items(), key=lambda x:x[1])

    def show_k_frequent(self, cnt=100, is_top=True):
        list_char = sorted(self.__char_dict.items(), key=lambda x:x[1], reverse=False)[:2000]
        x = []
        y = []
        for i in range(len(list_char)):
            #x.append(list_char[i][0])
            y.append(list_char[i][1])
        print(np.sum(y))
        print(len(y))
        print(np.sum(y) / len(y))
        print(y[math.ceil(len(y)/2)])
        fig = plt.figure(dpi=200)
        plt.tick_params(axis='x', labelsize=4)
        #plt.bar(range(len(list_char)), y, facecolor='#9999ff', edgecolor='white')
        plt.plot(range(len(list_char)), y)
        plt.yticks(np.arange(20, 100, 2))
        fig.tight_layout()
        plt.show()

    def char2labels(self, c):
        return self.__char2labels_dict[c]

def args_parser():
    args = sys.argv
    assert len(args) > 1, "Need file path at least"
    assert os.path.exists(args[1])
    k = 100
    if len(args) > 2 and str.isdigit(args[2]):
        k = min(int(args[2]), k)
    is_top = True
    if args.count("-R") > 0 or args.count("-r") > 0:
        is_top = False

    return args[1], k, is_top

if __name__ == '__main__':
    file_path = "./../train_data/LabelTrain.txt"
    cs = CharStats(file_path)
    cs.show_k_frequent()

    #print(cs.mean())
