# -*- coding: utf-8 -*-
# @Time    : 2025/1/26 12:30
# @Author  : licheng
# @File    : ssh_test.py
from my_detect import *
import cv2
import json
import os
import atexit

@atexit.register
def clean():       #ctrl + c 退出程序时清理temp文件夹中的临时内容
    file_list = os.listdir('./temp/')
    for file in file_list:
        file_path = os.path.join('./temp/', file)
        if os.path.isfile(file_path):
            os.remove(file_path)
# 开始运行程序时清空temp文件夹中的内容
file_list = os.listdir('./temp/')
for file in file_list:
    file_path = os.path.join('./temp/', file)
    if os.path.isfile(file_path):
        os.remove(file_path)
# 初始化
fat = fatigue_driving()
fat.create_model()
i = 0
# 死循环读取图片，读取完识别并将数据存为json文件
while True:
    img_filename = './temp/server_ssh_test' + str(i) + '.png'
    flag = True
    # 死循环读取图片，使用try，防止程序报错而直接停止
    while flag:
        try:
            img = cv2.imread(img_filename)
            img = cv2.resize(img, (800, 600))
        except Exception as e:
            flag = True
        else:
            print('receive:' + img_filename)
            flag = False
    # 每读一次删除一次临时文件
    file_list = os.listdir('./temp/')
    for file in file_list:
        file_path = os.path.join('./temp/', file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # 将图片识别数据存为json文件
    my_labels, my_boxs = fat.detect(img)
    data = [my_labels, my_boxs]
    json_filename = './temp/server_ssh_test' + str(i) + '.json'
    i = i + 1
    with open(json_filename, 'w') as f:
        json.dump(data, f)