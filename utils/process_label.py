import os

if __name__ == '__main__':
    gt_file = '../train_data/ppdataset/train/labeltrain.txt'
    gt_list = []
    with open(gt_file, 'r', encoding='UTF-8-sig') as f:
        all = f.readlines()
        for each in all:
            each = 'train/'+each
            gt_list.append(each)
    print(gt_list)
    with open(gt_file, 'w+', encoding='UTF-8-sig') as f:
        f.writelines(gt_list)