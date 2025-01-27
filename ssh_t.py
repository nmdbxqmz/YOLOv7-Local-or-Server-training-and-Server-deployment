# -*- coding: utf-8 -*-
# @Time    : 2025/1/26 9:12
# @Author  : licheng
# @File    : ssh_t.py
import os
import paramiko
import cv2
import json

import atexit

@atexit.register
def clean():        #ctrl + c 退出程序时清理temp文件夹中的临时内容同时关闭ssh
    file_list = os.listdir('./temp/')
    for file in file_list:
        file_path = os.path.join('./temp/', file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # 关闭SSH连接
    sftp.close()
    ssh.close()
# 开始运行程序时清空temp文件夹中的内容
file_list = os.listdir('./temp/')
for file in file_list:
    file_path = os.path.join('./temp/', file)
    if os.path.isfile(file_path):
        os.remove(file_path)
# 连接笔记本摄像头
cap = cv2.VideoCapture(0)
# 设置SSH连接参数
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='connect.nmb1.seetacloud.com', port='30201', username='root', password='BGR+vtWFsLiN')
# 使用SFTP传输文件
sftp = ssh.open_sftp()

i = 0

while True:
    # 发送
    flag_t = True
    ret, img = cap.read()
    if ret:
        filename_t = './temp/server_ssh_test' + str(i) + '.png'
        targetname_t = '/root/yolov7-main/temp/server_ssh_test' + str(i) + '.png'
        img = cv2.resize(img, (800, 600))       # 压缩图像，减少传输时间
        cv2.imwrite(filename_t, img)
        # 设置源文件路径和目标路径
        source_file = filename_t
        target_folder = targetname_t
        # 死循环上传，使用try，防止程序报错而直接停止
        while flag_t:
            try:
                sftp.put(source_file, target_folder)  #目标文件名可以按需更改
            except Exception as e:
                    flag_t = True
            else:
                print("文件上传成功")
                flag_t = False
        # 接收
        flag_r = True
        filename_r = '/root/yolov7-main/temp/server_ssh_test' + str(i) + '.json'
        targetname_r = './temp/server_ssh_test' + str(i) + '.json'
        # 设置源文件路径和目标路径
        source_file = filename_r
        target_folder = targetname_r
        # 死循环接收，使用try，防止程序报错而直接停止
        while flag_r:
            try:
                sftp.get(source_file, target_folder)
            except Exception as e:
                flag_r = True
            else:
                print("文件下载成功")
                flag_r = False
        # 打开文件，获取图片识别的结果
        with open(target_folder, 'r') as file:
            loaded_data = json.load(file)
            print(loaded_data)
        my_labels, my_boxs = loaded_data
        # 进行标注
        if my_labels != 0:
            j = 0
            for box in my_boxs:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.putText(img, my_labels[j], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                j = j + 1
        # 显示
        windowname = 'img' + str(i)
        i = i + 1
        cv2.imshow(windowname, img)
        # 每进行完一轮操作就删除一次临时文件
        file_list = os.listdir('./temp/')
        for file in file_list:
            file_path = os.path.join('./temp/', file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # 手动关闭窗口来进行下一轮的识别
        cv2.waitKey(0)
