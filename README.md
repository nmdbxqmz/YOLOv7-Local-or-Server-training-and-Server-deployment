# YOLOv7-Local-or-Server-training-and-Server-deployment
* 因为大部分文件都是开源的，所以本仓库不附带文件，但是会在相应位置给出下载链接

## REANDME目录
* 配置环境
* 标注数据
* yolov7参数修改
* 本地训练
* 服务器训练
* 服务器部署

## 配置环境
### anaconda新建环境
* 因为yolo对环境安装的第三包的版本要求比较严格，所以建议使用anaconda来创建一个yolo专用的环境
* 安装anaconda下面的链接已经讲的很详细了，不过多赘述

  anaconda安装教程：
  >https://blog.csdn.net/qq_44000789/article/details/142214660
* 打开anaconda自带的anaconda prompt，输入以下指令来创建一个python版本为3.7.4的新环境
  ```
  conda create -n yolo python==3.7.4 //yolo为该环境的名称，你也可以改为其他的，不过需要注意之后的部分指令也需要做相应的更改
  ```
  此时会在anaconda安装目录下的envs生成一个叫yolo的文件，如下图所示：
  
