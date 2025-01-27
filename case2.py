# -*- coding: utf-8 -*-
# @Time    : 2024/1/20 11:24
# @Author  : licheng
# @File    : case2.py
from random import sample
import os

xml_path = './datasets/Fatigue_driving_detection/labels/'
file_list = os.listdir(xml_path)
val_file_list = sample(file_list, 200)  # 选择了200张做测试集
line = ''
for i in val_file_list:
    if i.endswith('.txt'):
        line += 'datasets/Fatigue_driving_detection/images/' + i.split('.')[0] + '.png\n'  # datasets/Fatigue_driving_detection/images/ 是yolov7训练使用的
with open('datasets/Fatigue_driving_detection/val.txt', 'w+') as f:
    f.writelines(line)

test_file_list = sample(file_list, 200)
line = ''
for i in test_file_list:
    if i.endswith('.txt'):
        line += 'datasets/Fatigue_driving_detection/images/' + i.split('.')[0] + '.png\n'
with open('./datasets/Fatigue_driving_detection/test.txt', 'w+') as f:
    f.writelines(line)

line = ''
for i in file_list:
    if i not in val_file_list and i not in test_file_list:
        if i.endswith('.txt'):
            line += 'datasets/Fatigue_driving_detection/images/' + i.split('.')[0] + '.png\n'
with open('./datasets/Fatigue_driving_detection/train.txt', 'w+') as f:
    f.writelines(line)
